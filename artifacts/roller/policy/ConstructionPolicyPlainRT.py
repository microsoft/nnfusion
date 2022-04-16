from config import *
from cost_model import *
from .PolicyBase import *
import math
import numpy

log_regular_tile_found_key = "regular_tile_found"

def size_of(dtype):
    if dtype == "float":
        return 4
    if dtype == "half":
        return 2

def lcm(x, y):
    return x * y // math.gcd(x, y)

def num_tiles(large_tile, base_tile):
    ret = 1
    for d1, d2 in zip(large_tile, base_tile):
        ret *= math.ceil(d1 / d2)
    return ret

def eligible(op, arch, rprog, mem_level, tile_tensor = "output"):
    # a tile size is not eligible if
    # 1, the # of used registers exceed the capacity
    # 2, the size of memory used at certain level exceeds the capacity
    if mem_level == 1: # reg tiling
        # check register usage
        regTile = rprog.GetTile(mem_level)
        reg_usage_2 = op.RegUsage(regTile, tile_tensor)
        if reg_usage_2 > arch._reg_cap(1):
            return False
    if mem_level == 0: # smem tiling
        num_threads = rprog.GetParallelism(mem_level + 1)
        regTile = rprog.GetTile(mem_level + 1)
        # check register usage
        reg_usage_2 = op.RegUsage(regTile, tile_tensor)
        reg_usage_1 = reg_usage_2 * num_threads
        if reg_usage_1 > arch._reg_cap(0):
            return False
        # check shared memory usage
        smemTile = rprog.GetTile(mem_level)
        if op.MemFootprint(smemTile, tile_tensor) >= arch.mem_cap(0):
            return False
        # remove over sized warps
        if op.use_tc:
            num_threads *= 32
        return num_threads <= 512
    return True

def divisible(large_tile, base_tile):
    for d1, d2 in zip(large_tile, base_tile):
        if d1 % d2 > 0:
            return False
    return True

def Prod(tile):
    ret = 1
    for d in tile:
        ret *= d
    return ret

def RewriteSche_BankSize(schedule,smem_bank_size):
    '''
    bank size change here
    TODO: data type = float, 4B by default
    TODO: most inside axis is base_tile[1]
    '''
    level2_tile,level2_redu = schedule.get_tile(2)
    level1_tile,level1_redu = schedule.get_tile(1)
    merge_number = smem_bank_size // 4
    # inner_axis_id = len(level2_tile) - 1

    if level2_tile[-1] < merge_number:
        expand = merge_number // level2_tile[-1]
        new_level2_tile = list(level2_tile[:-1]) + [merge_number]
        schedule.update_tile(mem_level=2,dim=new_level2_tile,reduction_dict=level2_redu)

        if level1_tile[-1] >= expand:
            new_level1_tile = list(level1_tile[:-1]) + [level1_tile[-1] // expand]
            schedule.update_tile(mem_level=1,dim=new_level1_tile,reduction_dict=level1_redu)
        else:
            # TODO redundant computation in this branch, report or other solution?
            new_level1_tile = list(level1_tile[:-1]) + [1]
            schedule.update_tile(mem_level=1,dim=new_level1_tile,reduction_dict=level1_redu)
    return schedule


class ConstructionPolicyPlainRT(PolicyBase):
    """
        Constructing tiling schedules using DFS for sizes
        expand tiling sizes for alignment requirement
    """
    def __init__(self, op, arch,
                 smem_tiling=False,
                 reg_tiling=False,
                 st_align=False,
                 padding_threshold_cap=1.0,
                 shrink_tiny=False,
                 tile_tensor="output"):
        self.op = op
        self.arch = arch
        self.tile_tensor = tile_tensor
        self.num_level = arch.num_level
        expr_out = self.op.expr(self.op.Dimensions())
        outputs = expr_out[1]
        self.th_cap = padding_threshold_cap
        self.shrink_tiny = shrink_tiny

        self.saxis, self.raxis = get_axis_names(outputs[0])
        self.tile_dim = len(self.saxis)
        self.step_dim = len(self.raxis)
        
        self.raw_rprogs = []
        self.all_results = []
        self.in_results = set()

        self.ConstructionLog = {}
        self.ConstructionLog[log_regular_tile_found_key] = False
        self.border_rprogs = [[] for _ in range(self.num_level)]


    def AlignedToMemory(self, rtile, mem_level):
        input_subtensors = rtile.GetInputDataTiles()
        output_subtensors = rtile.GetOutputDataTiles()
        subtensors = input_subtensors + output_subtensors

        full_input_tensors = self.op.input_tensors
        full_output_tensors = self.op.output_tensors
        full_tensors = full_input_tensors + full_output_tensors

        for subtensor_shape, full_tensor in zip(subtensors, full_tensors):
            st_aligned = False
            #st_dim = subtensors[tensor_name]
            full_dim = full_tensor.shape
            if subtensor_shape[-1] >= full_dim[-1]:
                st_aligned = True
            base_transaction_size = self.arch.transaction_size[mem_level] // self.op.InputTypeSize()
            if subtensor_shape[-1] % base_transaction_size == 0:
                st_aligned = True
            if not st_aligned:
                return False
        return True


    def GetAlignedSteps(self, base_rtile, mem_level):
        """
        returns
            steps: list of possible step choices for each dimension
            new_base_rtile: the basic tile size, inferred based on both arithmetic and architecture info
        """
        # hardcode for two-level tiling now
        full_dim = self.op.Dimensions()
        full_sdim = self.op.SDimensions()
        dim = len(full_dim)
        sdim = len(full_sdim)
        steps_arch = []
        base_sdims = base_rtile.SDimensions()
        base_rdims = base_rtile.RDimensions()

        if mem_level == 1:
            # reg level, using power of 2 tiles
            for d in range(sdim):
                steps_arch.append([x * base_sdims[d] for x in [1, 2, 4, 8, 16, 32]])
            for rd in base_rdims:
                steps_arch.append([rd])

        new_base_sdims = list(base_sdims)
        new_base_rdims = list(base_rdims)
        if mem_level == 0:
            # smem level
            # last level aligned to transaction size
            new_base_sdims[-1] = lcm(new_base_sdims[-1], self.arch.transaction_size[mem_level] // self.op.InputTypeSize())
            for d in range(sdim):
                steps_arch.append([new_base_sdims[d] * (i + 1) for i in range(512)])

            # hardcode: each reduction axis aligned to axis length or transaction size
            if len(self.raxis) > 0:
                for base_rdim, raxis_name in zip(new_base_rdims, self.raxis):
                    base_transaction_size = self.arch.transaction_size[0] // self.op.InputTypeSize() * 4
                    rlen_cap = min(self.op.GetAxisLen(raxis_name), 128)
                    if self.op.use_tc:
                        base_transaction_size = 64
                        rlen_cap = 128
                    new_base_rdim = lcm(base_rdim, base_transaction_size)
                    steps_arch.append([])
                    while True:
                        steps_arch[-1].append(min(new_base_rdim, rlen_cap))
                        if new_base_rdim >= rlen_cap:
                            break
                        new_base_rdim *= 2

        # scan through all steps, remove the ones with too much padding
        steps = []
        for d in range(sdim):
            steps.append([])
            for s in steps_arch[d]:
                padded_dim = math.ceil(full_dim[d] / s) * s
                if padded_dim <= full_dim[d] * (1 + self.padding_threshold):
                    steps[-1].append(s)
        for d in range(sdim, dim):
            steps.append(steps_arch[d])
        for step in steps:
            if len(step) == 0:
                return steps, None
        new_base_dim = [steps[d][0] for d in range(len(steps))]
        new_base_rtile = rTile(self.op.expr, new_base_dim, self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
        return steps, new_base_rtile


    def AlignedToCU(self, rprog, mem_level):
        if mem_level == 1:
            return True
        if mem_level == 0:
            block_size = rprog.GetParallelism(1)
            if self.op.use_tc:
                block_size *= 32
            return block_size % self.arch.warp_size == 0
        # scale out
        if mem_level == -1:
            grid_size = rprog.GetParallelism(0)
            return grid_size >= 2 * self.arch.compute_max_core[0]


    def EnlargeTile(self, last_rprog, rprog, steps, mem_level):
        key = rprog.Dump()
        if key in self.visited:
            return
        self.visited.add(key)
        if len(self.top_results) == self.TOPK:
            return

        def one_level_down(rprog, this_level):
            # print("going down {}".format(schedule.dump_to_string()))
            base_rtile = rprog.GetTile(this_level)
            steps, base_rtile = self.GetAlignedSteps(base_rtile, this_level - 1)
            # valid = True
            # for step in steps:
            #     if len(step) == 0:
            #         valid = False
            # if valid:
            if base_rtile:
                rprog.AddTile(mem_level - 1, base_rtile)
                self.EnlargeTile(rprog, rprog, steps, mem_level - 1)
                rprog.DeleteTile(mem_level - 1)

        # exit if current schedule exceeds memory capacity
        if not eligible(self.op, self.arch, rprog, mem_level):
            #if mem_level == 0:
            #    print(last_schedule.dump_to_string(), last_schedule.subtile_count(0, 1))
            if self.AlignedToCU(last_rprog, mem_level):
                self.border_rprogs[mem_level].append(last_rprog)
                #print("border case")
                #print(last_schedule.dump_to_string())
            if mem_level > 0 and not self.ConstructionLog[log_regular_tile_found_key]:
                one_level_down(last_rprog, mem_level)
                #print("border case")
                #print(last_schedule.dump_to_string())
            return

        # check if current schedule is compute saturated and is compute intensive
        # scale-up after scale-out to favor large tiles
        aligned_to_cu = self.AlignedToCU(rprog, mem_level)
        reg_size = rprog.GetTile(1).Size()
        rtile = rprog.GetTile(mem_level)

        if aligned_to_cu and (reg_size >= 2):
            if mem_level == 0:
                if self.AlignedToMemory(rtile, mem_level):
                    self.top_results.append(rprog)
                    if len(self.top_results) == self.TOPK:
                        return
            else:
                # going one level down
                one_level_down(rprog, mem_level)

        # expand the current level tiles
        # caluate data reuse scores
        # enumerate from most inner dimension
        rtile = rprog.GetTile(mem_level)
        shape = rtile.Dimensions()

        for d in range(len(self.op.Dimensions()), 0, -1):
        #for d in range(len(self.op.dims[self.tile_tensor]), 0, -1):
            if len(steps[d - 1]) <= 1:
                continue
            new_rprog = rprog.copy()
            new_shape = shape.copy()
            old_step = steps[d - 1].pop(0)
            new_shape[d - 1] = steps[d - 1][0]
            new_rtile = rTile(rprog.expr, new_shape, self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
            new_rprog.AddTile(mem_level, new_rtile)

            self.EnlargeTile(rprog, new_rprog, steps, mem_level)
            steps[d - 1].insert(0, old_step)

    def emit_raw_configs(self, padding_threshold=0):
        """
            using BFS to iteratively search for all eligible candidates level by level
        """
        # initialize uni tiles
        mem_level = self.num_level - 1
        uniTile = rTile(self.op.expr, [1 for _ in range(self.tile_dim + self.step_dim)], self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
        uniProg = rProg(self.arch.num_level, self.op)
        uniProg.AddTile(self.num_level, uniTile)
        uniProg.AddTile(self.num_level - 1, uniTile)

        # initialize key hyperparameters
        self.padding_threshold = padding_threshold
        steps, _ = self.GetAlignedSteps(uniTile, 1)
        self.top_results = []

        self.visited = set()
        self.EnlargeTile(uniProg, uniProg, steps, mem_level)
        
        #for schedule in self.top_results:
        #    schedule = self.expand_reduce_axis(schedule, "k", 0)

        return self.top_results


    def try_shrink(self, rprog):
        sdim = len(rprog.saxis)
        while True:
            smem_tile_dim = rprog.GetTile(0).Dimensions()
            # validate grid size, return if grid size >= # of CUs
            grid_size = rprog.GetParallelism(0)
            if grid_size >= self.arch.compute_max_core[0]:
                return rprog
            
            # otherwise, shrink dimentions based on data reuse
            #print("try shrink small config: {}".format(schedule.dump_to_string()))
            # r_scores = DataReuseScore(self.op, rprog, 0)
            r_scores = self.DataReuseScore(rprog, 0)[:sdim]
            reversed_score_np = numpy.array([-s for s in r_scores])
            dim_order = numpy.argsort(reversed_score_np)
            for d in dim_order:
                if smem_tile_dim[d] == 1:
                    continue
                for l in range(self.arch.num_level):
                    tile = rprog.GetTile(l)
                    tile_sdim = tile.SDimensions()
                    tile_sdim[d] = math.ceil(tile_sdim[d] / 2)
                    tile_rdim = tile.RDimensions()
                    new_tile = rTile(tile.expr, tile_sdim + tile_rdim, self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
                    rprog.UpdateTile(new_tile, l)             
            # print("config after shrinking: {}".format(rprog.Dump()))
    

    def emit_config_without_trails(self, topk):
        # directly compute the theoretical performance for each raw configs and pick the optimal k configs
        # will call self.emit_raw_configs()
        self.TOPK = topk
        th = 0

        while th <= self.th_cap and len(self.all_results) < topk:
            print("threshold {}".format(th))
            self.emit_raw_configs(th)
            # take border cases if no IO intensity satisfied configs
            if len(self.top_results) == 0:
                self.top_results = self.border_rprogs[0][:self.TOPK]
            if len(self.top_results) == 0:
                print("failed to find results with padding threshold {}".format(th))
            else:
                print("found {} results in first round with threshold {}".format(len(self.top_results), th))
                # add current results to all
                for result in self.top_results:
                    key = result.Dump()
                    if key not in self.in_results:
                        self.in_results.add(key)
                        self.all_results.append(result)
            th += 0.1
        
        # handling small configs
        if self.shrink_tiny:
            for config in self.all_results:
                config = self.try_shrink(config)
        return self.all_results[:self.TOPK]

        output_results = []
        for schedule in self.all_results[:self.TOPK]:
            # print('init schedule:', schedule.dump_to_string())
            new_sche = RewriteSche_BankSize(schedule,self.arch.smem_bank_size)
            # print('updated schedule:', schedule.dump_to_string()) 
            output_results.append(new_sche)

        return output_results
