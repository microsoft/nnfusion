from config import *
from cost_model import *
from .PolicyBase import *
import math

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

visited = set()

def DFS_tile(cur_tile, base_tile, lower_limit, upper_limit, ret_list):
    size = num_tiles(cur_tile, base_tile)
    if size > upper_limit:
        return
    if tuple(cur_tile) in visited:
        return
    visited.add(tuple(cur_tile))
    if lower_limit <= size and size <= upper_limit:
        ret_list.append(cur_tile)
    for d in range(len(cur_tile)):
        next_tile = cur_tile.copy()
        #next_tile[d] = cur_tile[d] + base_tile[d]
        next_tile[d] = cur_tile[d] + cur_tile[d]
        DFS_tile(next_tile, base_tile, lower_limit, upper_limit, ret_list)


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


class ConstructionPolicy(PolicyBase):
    """
        Enumerating basic building blocks for given operators
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

    def BFS_Tiles(self, base_schedule, stride_spatial, stride_reduce, mem_level, Tiling = "all", tile_tensor = "output"):
        # Run BFS and return all configs that satisfy memory limit
        # stride_spatial: the strides of enumerating configurations
        # stride_reduce: the reduction step sizes for all schedules
        
        dim_size = len(self.op.dims[tile_tensor])
        candidates = [] # list of Schedule()

        # using tile from last level as a base tile
        dim_queue = [stride_spatial]
        visited = set()
        while len(dim_queue) > 0:
            cur_dim = dim_queue.pop(0)
            if tuple(cur_dim) in visited:
                continue
            visited.add(tuple(cur_dim))

            new_schedule = base_schedule.copy()
            new_schedule.add_tile(mem_level, cur_dim, stride_reduce)
            if not eligible(self.op, self.arch, new_schedule, mem_level):
                continue
            
            candidates.append(new_schedule)

            for d in range(dim_size, 0, -1):
                if cur_dim[d - 1] >= self.op.dims[tile_tensor][d - 1]:
                    continue
                next_dim = list(cur_dim).copy()
                if Tiling == "regular":
                    next_dim[d - 1] = min(next_dim[d - 1] + cur_dim[d - 1], self.op.dims[tile_tensor][d - 1])
                    #next_dim[d - 1] = next_dim[d - 1] + cur_dim[d - 1]
                elif Tiling == "all":
                    next_dim[d - 1] = min(next_dim[d - 1] + stride_spatial[d - 1], self.op.dims[tile_tensor][d - 1])
                    #next_dim[d - 1] = next_dim[d - 1] + stride_spatial[d - 1]
                dim_queue.append(next_dim)
        return candidates


    def divide_schedules(self, raw_schedules, mem_level):
        # divide the schedules into two sets, based on comparison between memory and compute
        compute_intensive_set, memory_intensive_set = [], []
        for schedule in raw_schedules:
            reg_size = Prod(schedule.get_tile(self.arch.num_level - 1)[0])
            tile_dim, rdict = schedule.get_tile(mem_level)
            reduction_size = rdict["k"]

            max_warp = min(32, 512 // reg_size)
            compute_workload = self.op.compute_workload(tile_dim, rdict)
            compute_throughput = self.compute_db.lookup(64,8,1) * self.arch._compute_max_core[0]
            #compute_throughput = self.arch._peak_flops# / self.arch._compute_max_core[0]
            compute_latency = compute_workload / compute_throughput

            memory_tensors = self.op.memory_workload(tile_dim, rdict, mem_level)
            memory_workload = sum([memory_tensors[tensor_name] for tensor_name in memory_tensors])

            memory_throughput = self.arch.memory_bw(mem_level)# / self.arch._mem_max_core[0]
            memory_latency = memory_workload / memory_throughput
            
            #print(schedule.dump_to_string())
            #print(compute_workload, memory_workload)
            #print(compute_latency, memory_latency)
            if compute_latency > memory_latency:
                compute_intensive_set.append(schedule)
            else:
                memory_intensive_set.append(schedule)
        return compute_intensive_set, memory_intensive_set

    def prune_and_find_small_configs(self, raw_schedules, mem_level):
        """
        initialize small ops
        returns
            schedules: pruned schedules
        """
        if mem_level == 0:
            schedules = []
            for schedule in raw_schedules:
                reg_tile = schedule.get_tile(1)[0]
                smem_tile = schedule.get_tile(0)[0]
                pruned = False
                # warp size and warp schedulers check
                num_threads = num_tiles(smem_tile, reg_tile)
                if num_threads % 128 > 0 or num_threads > 512:
                    pruned = True
                if not pruned:
                    schedules.append(schedule)
        else:
            schedules = raw_schedules
        #return schedules
        compute_intensive_set, memory_intensive_set = self.divide_schedules(schedules, mem_level)
        self.small_op_sets[mem_level] = memory_intensive_set
        #for config in compute_intensive_set:
        #    print(config.dump_to_string())
        if len(compute_intensive_set) > 0:
            schedules = compute_intensive_set
        else:
            schedules = memory_intensive_set

        # op dimension division check
        def try_prune_non_divisible(schedules, mem_level):
            divisible_schedules = []
            full_dim = self.op.dims[self.tile_tensor]
            for schedule in schedules:
                tile_dim, _ = schedule.get_tile(mem_level)
                if divisible(full_dim, tile_dim):
                    divisible_schedules.append(schedule)
            if len(divisible_schedules) > 0:
                return divisible_schedules
            return schedules
        schedules = try_prune_non_divisible(schedules, mem_level)
        return schedules


    def emit_raw_candidates_level(self, ll_schedules, mem_level):
        stride_spatial_arch = [1 for _ in range(len(self.saxis_names))]
        stride_reduce_arch = {raxis: 1 for raxis in self.raxis_names}
        # hardcode for (M, K) * (K, N) matmul for now
        if mem_level == 0:
            stride_reduce_arch['k'] = self.arch._transaction_size[mem_level] // self.size_of_type
            #stride_reduce_arch['k'] = 32
            stride_spatial_arch[-1] = self.arch._transaction_size[mem_level] // self.size_of_type
        for raxis in self.raxis_names:
            stride_reduce_arch['k'] = min(stride_reduce_arch['k'], self.op.reduction_axis_len())

        schedules = []
        for ll_schedule in ll_schedules:
            # get stride by GCD between last level schedule and architecture constraints
            base_stile, base_rstep = ll_schedule.get_tile(mem_level + 1)
            stride_spatial = [lcm(x, y) for x, y in zip(base_stile, stride_spatial_arch)]
            stride_reduce = {k: lcm(base_rstep[k], stride_reduce_arch[k]) for k in base_rstep}
            if mem_level == 1:
                raw_schedules = self.BFS_Tiles(ll_schedule, stride_spatial, stride_reduce, mem_level, Tiling = "regular", tile_tensor = "output")
            else:
                raw_schedules = self.BFS_Tiles(ll_schedule, stride_spatial, stride_reduce, mem_level, Tiling = "all", tile_tensor = "output")
            schedules.extend(raw_schedules)
        return schedules

    def expand_reduce_axis(self, config, reduction_axis_name, mem_level):
        #rstep = config._reduction_size[mem_level][reduction_axis_name]
        config._reduction_size[mem_level][reduction_axis_name] = 32
        while True:
            if eligible(self.op, self.arch, config, mem_level):
                return config
            config._reduction_size[mem_level][reduction_axis_name] = config._reduction_size[mem_level][reduction_axis_name] // 2

    def emit_raw_configs(self):
        """
            using BFS to iteratively search for all eligible candidates level by level
        """
        mem_level = self.num_level - 1
        uni_tile = Schedule(self.dim_size, self.saxis_names, self.raxis_names)
        uni_tile.add_tile(mem_level + 1, [1 for _ in self.op.dims[self.tile_tensor]], {name: 1 for name in self.raxis_names})
        schedule_candidates = [uni_tile]

        # top-down search iterations
        while mem_level >= 0:
            last_level_schedules = schedule_candidates.copy()
            schedule_candidates = self.emit_raw_candidates_level(last_level_schedules, mem_level)
            schedule_candidates = self.prune_and_find_small_configs(schedule_candidates, mem_level)
            mem_level -= 1

        # check small op
        def try_scale_out(schedules):
            for schedule in schedules:
                TB_tile, _ = schedule.get_tile(0)
                parallelism = self.op.get_grid_size(TB_tile)
                # hardcode for v100
                if parallelism > 4 * self.arch._compute_max_core[0]:
                    return True
            return False

        if not try_scale_out(schedule_candidates):
            # is small op, fallback to last level
            print("is small op")
            for config in self.small_op_sets[1]:
                print(config.dump_to_string())
            schedule_candidates = self.emit_raw_candidates_level(self.small_op_sets[1], 0)
            schedule_candidates = self.prune_and_find_small_configs(schedule_candidates, 0)
            schedule_candidates = self.small_op_sets[0]

            for schedule in schedule_candidates:
                schedule = self.expand_reduce_axis(schedule, "k", 0)

        print(len(schedule_candidates))
        for config in schedule_candidates:
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

        print("---------------------------------------------------")
        return schedule_candidates


    def emit_config_without_trails(self, topk):
        # directly compute the theoretical performance for each raw configs and pick the optimal k configs
        # will call self.emit_raw_configs()
        raw_configs = self.emit_raw_configs()
        print("{} raw configs generated after pruning".format(len(raw_configs)))
        perf_config = []
        for config in raw_configs:
            theo_perf = self.cost_model.Theoretical_Perf(config)
            
            perf_config.append((theo_perf, config))

        def sort_key(a):
            return a[0]
        perf_config.sort(key=sort_key)
            
        return [x for (_, x) in perf_config[:topk]]

