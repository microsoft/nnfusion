from config import *
from cost_model import *
from .PolicyBase import *
import math
import numpy

log_memory_bound_key = "is_memory_bound"
log_reg_tile_found_key = "reg_tile_found"
log_small_op_key = "is_small_op"

def lcm(x, y):
    return x * y // math.gcd(x, y)

def eligible(op, arch, schedule, mem_level, tile_tensor = "output"):
    # a tile size is not eligible if
    # 1, the # of used registers exceed the capacity
    # 2, the size of memory used at certain level exceeds the capacity
    dim, reduc = schedule.get_tile(mem_level)
    if mem_level == 1: # reg tiling
        _, reg_usage_2 = op.reg_usage(dim, tile_tensor)
        if reg_usage_2 > arch.reg_cap(1):
            return False
    if mem_level == 0: # smem tiling
        dim1, reduc1 = schedule.get_tile(mem_level + 1)
        _, reg_usage_2 = op.reg_usage(dim1, tile_tensor)
        reg_usage_1 = reg_usage_2 * schedule.subtile_count(mem_level, mem_level + 1)
        if reg_usage_1 > arch.reg_cap(0):
            return False
        if op.memory_footprint(dim, reduc, mem_level, tile_tensor) > arch.mem_cap(0):
            return False
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

def num_tiles(large_tile, base_tile):
    ret = 1
    for d1, d2 in zip(large_tile, base_tile):
        ret *= math.ceil(d1 / d2)
    return ret

def DataReuseScore(op, schedule, mem_level, tile_tensor="output"):
    """
    return a list of scores on each dimension
    """
    tile_dim, reduction_dict = schedule.get_tile(mem_level)
    ret_list = []
    tile_dim = list(tile_dim)
    for d in range(len(op.dims[tile_tensor])):
        tile_dim[d] += 1
        compute_workload = op.compute_workload(tile_dim, reduction_dict)
        memory_tensors = op.memory_workload(tile_dim, reduction_dict, mem_level)
        memory_workload = sum([memory_tensors[tensor_name] for tensor_name in memory_tensors])
        ret_list.append(-compute_workload / memory_workload)
        tile_dim[d] -= 1
    return ret_list


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
        return self.store[key]


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


class ConstructionPolicyV1(PolicyBase):
    """
        Constructing tiling schedules using DFS, hardcode reduction step size for now
    """
    def __init__(self, op, arch, saxis_names, raxis_names, tile_tensor="output", data_type="float"):
        self.op = op
        self.tile_tensor = tile_tensor
        self.arch = arch
        self.dim_size = len(self.op.dims[self.tile_tensor])
        self.num_level = arch.num_level
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
        
        if data_type == "float":
            self.size_of_type = 4
        self.raw_configs = []

        self.ConstructionLog = {}
        self.ConstructionLog[log_memory_bound_key] = False
        self.ConstructionLog[log_reg_tile_found_key] = False
        self.ConstructionLog[log_small_op_key] = True

        self.pure_memory = False # switch on to construct tiles for memory-bound ops
        self.tolerate_small_tile = False # switch on to construct tiles for small ops


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

        if mem_level == 1:
            # reg level, using power of 2 tiles
            for d in range(dim):
                steps_arch[d] = [1, 2, 4, 8, 16, 32]
            rstep = {raxis : 1 for raxis in self.raxis_names}

        new_base_tile = list(base_tile)
        if mem_level == 0:
            # smem level
            # last level aligned to transaction size
            new_base_tile[-1] = lcm(new_base_tile[-1], self.arch.transaction_size[0] // self.size_of_type)
            for d in range(dim):
                steps_arch[d] = [new_base_tile[d] * (i + 1) for i in range(32)]
            rstep = {raxis : 1 for raxis in self.raxis_names}
            # hardcode for now
            rstep[self.raxis_names[-1]] = min(self.op.reduction_axis_len(), self.arch.transaction_size[0] // self.size_of_type)
        
        # scan through all steps, remove the ones with too much padding
        steps = [[] for _ in range(dim)]
        for d in range(dim):
            for s in steps_arch[d]:
                padded_dim = math.ceil(full_dim[d] / s) * s
                if padded_dim <= full_dim[d] * (1 + self.padding_threshold):
                    steps[d].append(s)
        return steps, new_base_tile, rstep


    def IsComputeIntensive(self, schedule, mem_level):
        """
        return True if the schedule is compute intensive
        """
        reg_size = Prod(schedule.get_tile(self.arch.num_level - 1)[0])
        tile_dim, rdict = schedule.get_tile(mem_level)
        
        max_warp = min(32, 512 // reg_size)
        compute_workload = self.op.compute_workload(tile_dim, rdict)
        compute_throughput = self.compute_db.lookup(64,8,8) * self.arch.compute_max_core[0]
        #compute_throughput = self.arch.peak_flops# / self.arch.compute_max_core[0]
        compute_latency = compute_workload / compute_throughput
        if self.tolerate_small_tile:
            compute_latency *= 4

        memory_tensors = self.op.memory_workload(tile_dim, rdict, mem_level)
        memory_workload = sum([memory_tensors[tensor_name] for tensor_name in memory_tensors])

        memory_throughput = self.arch.memory_bw(mem_level)# / self.arch.mem_max_core[0]
        memory_latency = memory_workload / memory_throughput
        return compute_latency > memory_latency
        

    def IsPeakComputeTile(self, schedule, mem_level):
        reg_tile, _ = schedule.get_tile(1)
        if mem_level == 1:
            reg_size = Prod(reg_tile)
            if self.pure_memory:
                return reg_size >= 2
            if self.tolerate_small_tile:
                return reg_size >= 8
            return reg_size >= 32
        if mem_level == 0:
            smem_tile, _ = schedule.get_tile(0)
            num_threads = num_tiles(smem_tile, reg_tile)
            return num_threads % 128 == 0 and num_threads <= 512
        # scale out
        if mem_level == -1:
            full_dim = self.op.dims[self.tile_tensor]
            smem_tile, _ = schedule.get_tile(0)
            return num_tiles(full_dim, smem_tile) >= 2 * self.arch.compute_max_core[0]


    def DFS_tile(self, schedule, base_tile, steps, mem_level):
        key = schedule.dump_to_string()
        if key in self.visited:
            return
        self.visited.add(key)
        if len(self.top_results) == self.TOPK:
            return

        if mem_level == 0:
            self.ConstructionLog[log_reg_tile_found_key] = True

        # exit if current schedule exceeds memory capacity
        if not eligible(self.op, self.arch, schedule, mem_level):
            if not self.pure_memory:
                is_compute_intensive = self.IsComputeIntensive(schedule, mem_level)
                if not is_compute_intensive and mem_level == 0:
                    self.ConstructionLog[log_memory_bound_key] = True
            return

        # check if current schedule is compute saturated and is compute intensive
        # scale-up after scale-out to favor large tiles
        is_peak_compute = self.IsPeakComputeTile(schedule, mem_level)
        is_compute_intensive = self.pure_memory or self.IsComputeIntensive(schedule, mem_level)
        #print(is_peak_compute, is_compute_intensive)

        #print(schedule.dump_to_string())
        #if mem_level == 0:
        #    print("thread size {}".format(schedule.subtile_count(0, 1)))
        #print(is_peak_compute, is_compute_intensive)
        if is_peak_compute and is_compute_intensive:
            if mem_level == 0:
                self.top_results.append(schedule)
                if self.IsPeakComputeTile(schedule, -1):
                    self.ConstructionLog[log_small_op_key] = False
                if len(self.top_results) == self.TOPK:
                    return
            else:
                # going one level down
                base_tile, _ = schedule.get_tile(mem_level)
                steps, base_tile, rstep = self.GetAlignedSteps(base_tile, mem_level - 1)
                schedule.update_tile(mem_level - 1, base_tile, rstep)
                self.DFS_tile(schedule, base_tile, steps, mem_level - 1)

        # expand the current level tiles
        # caluate data reuse scores
        r_scores = []
        dim = len(self.op.dims[self.tile_tensor])
        tile_dim, reduction_size = schedule.get_tile(mem_level)
        r_scores = DataReuseScore(self.op, schedule, mem_level)
        x = numpy.array(r_scores)
        dim_order = numpy.argsort(x)
        # enumerate from dimensions with highest scores
        for d in dim_order:
            if len(steps[d]) <= 1:
                continue
            new_schedule = schedule.copy()
            old_step = steps[d].pop(0)
            new_tile = list(tile_dim)
            new_tile[d] = steps[d][0]
            new_schedule.update_tile(mem_level, dim=new_tile, reduction_dict=None)

            self.DFS_tile(new_schedule, base_tile, steps, mem_level)
            steps[d].insert(0, old_step)

    def expand_reduce_axis(self, config, reduction_axis_name, mem_level):
        #rstep = config._reduction_size[mem_level][reduction_axis_name]
        #config._reduction_size[mem_level][reduction_axis_name] = 32
        limit = min(32, self.op.reduction_axis_len())
        last_r = config._reduction_size[mem_level][reduction_axis_name]
        #last_r = base_r
        while config._reduction_size[mem_level][reduction_axis_name] < limit:
            last_r = config._reduction_size[mem_level][reduction_axis_name]
            config._reduction_size[mem_level][reduction_axis_name] = min(last_r * 2, limit)
            if not eligible(self.op, self.arch, config, mem_level):
                config._reduction_size[mem_level][reduction_axis_name] = last_r
                break
        return config

    def emit_raw_configs(self, padding_threshold=0):
        """
            using BFS to iteratively search for all eligible candidates level by level
        """
        # initialize uni tiles
        mem_level = self.num_level - 1
        base_tile = Schedule(self.dim_size, self.saxis_names, self.raxis_names)
        base_tile.add_tile(mem_level + 1, [1 for _ in self.op.dims[self.tile_tensor]], {name: 1 for name in self.raxis_names})
        uni_schedule = Schedule(self.dim_size, self.saxis_names, self.raxis_names)
        uni_schedule.add_tile(mem_level + 1, [1 for _ in self.op.dims[self.tile_tensor]], {name: 1 for name in self.raxis_names})
        uni_schedule.add_tile(mem_level, [1 for _ in self.op.dims[self.tile_tensor]], {name: 1 for name in self.raxis_names})

        # initialize key hyperparameters
        self.TOPK = 50 # stop search after finding TOPK schedules
        self.padding_threshold = padding_threshold
        steps, _, _ = self.GetAlignedSteps([1 for _ in self.op.dims[self.tile_tensor]], 1)
        self.top_results = []

        self.visited = set()
        self.DFS_tile(uni_schedule, base_tile, steps, mem_level)
        
        for schedule in self.top_results:
            schedule = self.expand_reduce_axis(schedule, "k", 0)

        for config in self.top_results:
            reg_tile, _ = config.get_tile(1)
            smem_tile, rdict = config.get_tile(0)
            smem_tile_scale = [x // y for x, y in zip(smem_tile, reg_tile)]
            reduction_size = [rdict["k"], 1]
            try:
                config.compute_peak_performance = self.cost_model.get_compute_peak_performance(config, self.compute_db)
                config.glbmem_bandwidth = self.cost_model.get_glbmem_bandwidth(config, self.small_glbmem_db)
                config.smem_bandwidth = self.cost_model.get_smem_bandwidth(config)
                config.active_blocks_per_sm = self.activeblock_db.lookup(list(reg_tile), list(smem_tile), reduction_size)
            except:
                print("not found {}".format(config.dump_to_string()))
                # print("config: {} not found in database!", config_wh)
                continue
        return self.top_results


    def emit_config_without_trails(self, topk):
        # directly compute the theoretical performance for each raw configs and pick the optimal k configs
        # will call self.emit_raw_configs()
        self.TOPK = topk
        th = 0
        if self.op.reduction_axis_len() < 50:
            self.pure_memory = True
        self.emit_raw_configs(th)
        while len(self.top_results) == 0:
            print("failed to find results with padding threshold {}".format(th))
            # check if it is memory bound
            print(self.pure_memory)
            print(self.ConstructionLog[log_memory_bound_key])
            if not self.pure_memory and self.ConstructionLog[log_memory_bound_key]:
                print("this is a memory-bound op")
                self.pure_memory = True
                #th = 0
                #return self.top_results
            else:
                th += 0.01
            self.emit_raw_configs(th)
        
        # handling small op
        if self.ConstructionLog[log_small_op_key] and not self.tolerate_small_tile:
            print("is small op")
            self.tolerate_small_tile = True
            self.top_results = []
            return self.emit_config_without_trails(topk)
        print("{} raw configs generated with threshold = {}".format(len(self.top_results), th))
        return self.top_results

        perf_config = []
        for config in raw_configs:
            theo_perf = self.cost_model.Theoretical_Perf(config)
            perf_config.append((theo_perf, config))

        def sort_key(a):
            return a[0]
        perf_config.sort(key=sort_key)
        
        return [x for (_, x) in perf_config[:topk]]

