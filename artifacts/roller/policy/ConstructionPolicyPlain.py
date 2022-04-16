from config import *
from cost_model import *
from .PolicyBase import *
import math
import numpy

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

def eligible(op, arch, schedule, mem_level, check_smem=True, tile_tensor = "output"):
    # a tile size is not eligible if
    # 1, the # of used registers exceed the capacity
    # 2, the size of memory used at certain level exceeds the capacity
    dim, reduc = schedule.get_tile(mem_level)
    if mem_level == 1: # reg tiling
        # check register usage
        _, reg_usage_2 = op.reg_usage(dim, tile_tensor)
        if reg_usage_2 > arch._reg_cap(1):
            return False
    if mem_level == 0: # smem tiling
        dim1, reduc1 = schedule.get_tile(mem_level + 1)
        # check register usage
        _, reg_usage_2 = op.reg_usage(dim1, tile_tensor)
        reg_usage_1 = reg_usage_2 * schedule.subtile_count(mem_level, mem_level + 1)
        if reg_usage_1 > arch._reg_cap(0):
            return False
        # check shared memory usage
        if check_smem and op.memory_footprint(dim, reduc, mem_level, tile_tensor) >= arch.mem_cap(0):
            return False
        # check num threads
        num_threads = num_tiles(dim, dim1)
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

class ActiveBlockDB():
    def __init__(self):
        self.store = {}
        self.set_store = set()

    def hashkey(self, reg_tile, smem_tile, reduction_size):
        hash_key = str(reg_tile) + str(smem_tile) + str(reduction_size)
        return hash_key

    def insert(self, reg_tile, smem_tile, reduction_size, perf):
        key = self.hashkey(reg_tile, smem_tile, reduction_size)
        self.store[key] = perf
        self.set_store.add((tuple(reg_tile), tuple(smem_tile), tuple(reduction_size)))

    def lookup(self, reg_tile, smem_tile, reduction_size):
        key = self.hashkey(reg_tile, smem_tile, reduction_size)
        if key in self.store:
            return self.store[key]
        return None

    def lookup_schedule(self, config):
        reg_tile, _ = config.get_tile(1)
        smem_tile, rdict = config.get_tile(0)
        reduction_size = [rdict["k"], 1]
        return self.lookup(list(reg_tile), list(smem_tile), reduction_size)


class SmallGlbmemDB():
    def __init__(self):
        self.store = {}

    def hashkey(self, warp_num, grid_size):
        hash_key = str(warp_num) + "," + str(grid_size)
        return hash_key

    def insert(self, warp_num, grid_size, perf):
        key = self.hashkey(warp_num, grid_size)
        self.store[key] = perf

    def lookup(self, warp_num, grid_size):
        key = self.hashkey(warp_num, grid_size)
        return self.store[key]


class ComputeDB():
    def __init__(self):
        self.store = {}
        self.lst_store = []

    def hashkey(self, reg_size, warp_num, reduction):
        hash_key = str(reg_size) + "," + str(warp_num) + "," + str(reduction)
        return hash_key

    def insert(self, reg_size, warp_num, reduction, perf):
        key = self.hashkey(reg_size, warp_num, reduction)
        self.store[key] = perf
        self.lst_store.append([reg_size, warp_num, reduction])

    def lookup(self, reg_size, warp_num, reduction):
        key = self.hashkey(reg_size, warp_num, reduction)
        return self.store[key]


class ConstructionPolicyPlain(PolicyBase):
    """
        Constructing tiling schedules using DFS for sizes
        expand tiling sizes for alignment requirement
    """
    def __init__(self, op, arch, saxis_names, raxis_names, data_type="float", smem_tiling=False, tile_tensor="output"):
        self.op = op
        self.tile_tensor = tile_tensor
        self.arch = arch
        self.dim_size = len(self.op.dims[self.tile_tensor])
        self.num_level = arch.num_level
        self.smem_tiling = smem_tiling
        # TODO: load basic building blocks as well as any profiling infomation
        #       initialize the cost model with more infomation if necessary
        self.cost_model = WarpBasedCostModel(op, arch)
        self.saxis_names = saxis_names
        self.raxis_names = raxis_names
        self.small_op_sets = [[] for _ in range(self.num_level)]
        
        # active block per sm database
        self.activeblock_db = ActiveBlockDB()
        activeblock_filename = 'policy/activeblock_matmul.csv'
        with open(activeblock_filename, 'r') as fin:
            activeblock_db_str = fin.readlines()
            fin.close()
        for config_str in activeblock_db_str:
            if config_str.find('reg') != -1: continue
            config_list = config_str.split(',')
            config = list(map(int,config_list[:-1]))
            activeblocks = int(config_list[-1])
            if activeblocks > 0 and config[4] == config[5]:
                reg_tile = [config[0], config[1]]
                smem_tile = [config[0] * config[2], config[1] * config[3]]
                # k_reduction_size = [config[5], config[4], 1]
                reduction_size = [config[4], 1]
                self.activeblock_db.insert(reg_tile, smem_tile, reduction_size, activeblocks)
        # print(self.activeblock_db.set_store)

        # small global memory database
        self.small_glbmem_db = SmallGlbmemDB()
        glbmem_lookup_filename = 'policy/glbmem_small_lookup.csv'
        with open(glbmem_lookup_filename, 'r') as fin:
            glbmem_lookup_str = fin.readlines()
            fin.close()  
        for config_str in glbmem_lookup_str:
            if config_str.find('warp') != -1: continue
            config_list = config_str.split(',')
            throughput = float(config_list[-1])
            self.small_glbmem_db.insert(config_list[0], config_list[1], throughput)
        
        # computation database
        self.compute_db = ComputeDB()
        compute_filename = 'policy/basicbuildingblk_compute_mm_v2.csv'
        with open(compute_filename, 'r') as fin:
            compute_db_str = fin.readlines()
            fin.close()
        for config_str in compute_db_str:
            if config_str.find('warp') != -1: continue
            config_list = config_str.split(',')
            throughput = float(config_list[-1])
            self.compute_db.insert(config_list[0], config_list[1], config_list[2], throughput)
        
        self.raw_configs = []
        self.all_results = []
        self.in_results = set()

        #self.tolerate_small_tile = False # switch on to construct tiles for small ops
        self.border_configs = [[] for _ in range(self.num_level)]


    def AlignedToMemory(self, tile_dim, reduction_size, mem_level):
        subtensors = self.op.subtensor_dim(tile_dim, reduction_size, mem_level, self.tile_tensor)
        for tensor_name in subtensors:
            st_aligned = False
            st_dim = subtensors[tensor_name]
            full_dim = self.op.dims[tensor_name]
            if st_dim[-1] >= full_dim[-1]:
                st_aligned = True
            if st_dim[-1] % (self.arch.transaction_size[0] // size_of(self.op.input_type)) == 0:
                st_aligned = True
            if not st_aligned:
                return False
        return True


    def GetAlignedSteps(self, base_tile, mem_level):
        """
        returns
            steps: list for each tensor dimensions
            base_tile: the basic tile size, inferred based on both arithmetic and architecture info
            rstep: the aligned reduction step, hardcode for transaction size for now
        """
        # hardcode for two-level tiling now
        full_dim = self.op.dims[self.tile_tensor]
        dim = len(full_dim)
        steps_arch = [[] for _ in range(dim)]
        rstep = None

        if mem_level == 1:
            # reg level, using power of 2 tiles
            for d in range(dim):
                steps_arch[d] = [x * base_tile[d] for x in [1, 2, 4, 8, 16, 32]]
            rstep = {raxis : 1 for raxis in self.raxis_names}

        new_base_tile = list(base_tile)
        if mem_level == 0:
            # smem level
            # last level aligned to transaction size
            new_base_tile[-1] = lcm(new_base_tile[-1], self.arch.transaction_size[0] // size_of(self.op.input_type))
            for d in range(dim):
                steps_arch[d] = [new_base_tile[d] * (i + 1) for i in range(512)]
            rstep = {raxis : 1 for raxis in self.raxis_names}
            # hardcode for now
            if len(rstep) > 0:
                rstep[self.raxis_names[-1]] = min(self.op.reduction_axis_len(), self.arch.transaction_size[0] // size_of(self.op.input_type))
                if self.op.use_tc:
                    rstep[self.raxis_names[-1]] = 64
        
        # scan through all steps, remove the ones with too much padding
        steps = [[] for _ in range(dim)]
        for d in range(dim):
            for s in steps_arch[d]:
                #step = min(full_dim[d], s)
                step = s
                padded_dim = math.ceil(full_dim[d] / step) * step
                if padded_dim <= full_dim[d] * (1 + self.padding_threshold):
                    steps[d].append(step)
        return steps, new_base_tile, rstep


    def AlignedToCU(self, schedule, mem_level):
        reg_tile, _ = schedule.get_tile(1)
        if mem_level == 1:
            return True
        if mem_level == 0:
            smem_tile, _ = schedule.get_tile(0)
            num_threads = num_tiles(smem_tile, reg_tile)
            if self.op.use_tc:
                num_threads *= 32
            return num_threads % 32 == 0
        # scale out
        if mem_level == -1:
            full_dim = self.op.dims[self.tile_tensor]
            smem_tile, _ = schedule.get_tile(0)
            return num_tiles(full_dim, smem_tile) >= 2 * self.arch.compute_max_core[0]


    def DFS_tile(self, last_schedule, schedule, steps, mem_level):
        key = schedule.dump_to_string()
        if key in self.visited:
            return
        self.visited.add(key)
        if len(self.top_results) == self.TOPK:
            return

        def one_level_down(schedule, this_level):
            # print("going down {}".format(schedule.dump_to_string()))
            base_tile, _ = schedule.get_tile(this_level)
            steps, base_tile, rstep = self.GetAlignedSteps(base_tile, this_level - 1)
            valid = True
            for step in steps:
                if len(step) == 0:
                    valid = False
            if valid:
                schedule.add_tile(mem_level - 1, base_tile, rstep)
                self.DFS_tile(schedule, schedule, steps, mem_level - 1)
                schedule.delete_tile(mem_level - 1)

        # exit if current schedule exceeds memory capacity
        if not eligible(self.op, self.arch, schedule, mem_level, check_smem=self.smem_tiling):
            #if mem_level == 0:
            #    print(last_schedule.dump_to_string(), last_schedule.subtile_count(0, 1))
            if self.AlignedToCU(last_schedule, mem_level):
                self.border_configs[mem_level].append(last_schedule)
                #print("border case")
                #print(last_schedule.dump_to_string())
            if mem_level > 0:
                one_level_down(last_schedule, mem_level)
            return

        # check if current schedule is compute saturated and is compute intensive
        # scale-up after scale-out to favor large tiles
        aligned_to_cu = self.AlignedToCU(schedule, mem_level)
        reg_size = Prod(schedule.get_tile(1)[0])
        tile_dim, rstep = schedule.get_tile(mem_level)

        if aligned_to_cu and (reg_size >= 2):
            if mem_level == 0: 
                if self.AlignedToMemory(tile_dim, rstep, mem_level):
                    self.top_results.append(schedule)
                    if len(self.top_results) == self.TOPK:
                        return
            else:
                # going one level down
                one_level_down(schedule, mem_level)

        # expand the current level tiles
        # caluate data reuse scores
        # enumerate from most inner dimension
        for d in range(len(self.op.dims[self.tile_tensor]), 0, -1):
            if len(steps[d - 1]) <= 1:
                continue
            new_schedule = schedule.copy()
            old_step = steps[d - 1].pop(0)
            new_tile = list(tile_dim)
            new_tile[d - 1] = steps[d - 1][0]
            new_schedule.update_tile(mem_level, dim=new_tile, reduction_dict=rstep.copy())

            self.DFS_tile(schedule, new_schedule, steps, mem_level)
            steps[d - 1].insert(0, old_step)

    def expand_reduce_axis(self, config, reduction_axis_name, mem_level):
        #rstep = config._reduction_size[mem_level][reduction_axis_name]
        #config._reduction_size[mem_level][reduction_axis_name] = 32
        limit = min(64, self.op.reduction_axis_len())
        while config._reduction_size[mem_level][reduction_axis_name] < limit:
            last_r = config._reduction_size[mem_level][reduction_axis_name]
            config._reduction_size[mem_level][reduction_axis_name] = min(last_r * 2, limit)
            if not eligible(self.op, self.arch, config, mem_level, check_smem=self.smem_tiling):
                config._reduction_size[mem_level][reduction_axis_name] = last_r
                break
        return config

    def emit_raw_configs(self, padding_threshold=0):
        """
            using BFS to iteratively search for all eligible candidates level by level
        """
        # initialize uni tiles
        mem_level = self.num_level - 1
        uni_tile, uni_step = self.op.uni_schedule(self.saxis_names, self.raxis_names)
        
        uni_schedule = Schedule(self.dim_size, self.saxis_names, self.raxis_names)
        uni_schedule.add_tile(mem_level + 1, uni_tile, uni_step)
        uni_schedule.add_tile(mem_level, uni_tile, uni_step)

        # initialize key hyperparameters
        self.TOPK = 50 # stop search after finding TOPK schedules
        self.padding_threshold = padding_threshold
        steps, _, _ = self.GetAlignedSteps(uni_tile, 1)
        self.top_results = []

        self.visited = set()
        self.DFS_tile(uni_schedule, uni_schedule, steps, mem_level)
        
        # expand reduce axis
        if len(self.raxis_names) > 0:
            for i in range(len(self.top_results)):
                self.top_results[i] = self.expand_reduce_axis(self.top_results[i], "k", 0)

        return self.top_results


    def try_shrink(self, schedule):
        while True:
            smem_tile, _ = schedule.get_tile(0)
            # validate grid size, return if grid size >= # of CUs
            if self.op.use_tc:
                grid_x, grid_y = self.op.get_grid_size(smem_tile)
                grid_size = grid_x * grid_y
            else:
                grid_size = self.op.get_grid_size(smem_tile)
            if grid_size >= self.arch.compute_max_core[0]:
                return schedule
            
            # otherwise, shrink dimentions based on data reuse
            #print("try shrink small config: {}".format(schedule.dump_to_string()))
            for d in range(len(smem_tile)):
                if smem_tile[d] == 1:
                    continue
                for l in range(self.arch.num_level):
                    tile = list(schedule.get_tile(l)[0])
                    tile[d] = tile[d] // 2
                    schedule.update_tile(l, tile, reduction_dict=None)
            #print("config after shrinking: {}".format(schedule.dump_to_string()))
        return schedule
                    

    def emit_config_without_trails(self, topk):
        # directly compute the theoretical performance for each raw configs and pick the optimal k configs
        # will call self.emit_raw_configs()
        self.TOPK = topk
        th = 0

        while th <= 1 and len(self.all_results) <= topk:
            self.emit_raw_configs(th)
            # take border cases if no IO intensity satisfied configs
            if len(self.top_results) == 0:
                self.top_results = self.border_configs[0][:self.TOPK]
            if len(self.top_results) == 0:
                pass
                # print("failed to find results with padding threshold {}".format(th))
            else:
                # print("found {} results in first round with threshold {}".format(len(self.top_results), th))
                # add current results to all
                for result in self.top_results:
                    key = result.dump_to_string()
                    if key not in self.in_results:
                        self.in_results.add(key)
                        self.all_results.append(result)
            th += 0.2
        
        # handling small configs
        for config in self.all_results:
            config = self.try_shrink(config)

        output_results = []
        for schedule in self.all_results[:self.TOPK]:
            # print('init schedule:', schedule.dump_to_string())
            new_sche = RewriteSche_BankSize(schedule,self.arch.smem_bank_size)
            # print('updated schedule:', schedule.dump_to_string()) 
            output_results.append(new_sche)

        return output_results
