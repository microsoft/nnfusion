from config import *
from cost_model import *
from .PolicyBase import *

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


def num_tiles(large_tile, base_tile):
    ret = 1
    for d1, d2 in zip(large_tile, base_tile):
        ret *= d1 // d2
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


class BuildingBlockPolicy(PolicyBase):
    """
        Enumerating basic building blocks for given operators
    """
    def __init__(self, op, arch, saxis_names, raxis_names, tile_tensor="output"):
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


    def emit_raw_configs(self):
        # TODO: return a list of all tiling schedules:
        # 1, load basic building block configs and stats
        # 2, convert blocks to Sokoban tiling configs
        # currently hard coded for matrix multiplication
        if len(self.raw_configs) > 0:
            return self.raw_configs

        config_wh = set()
        out_shape = self.op.dims["output"]
        r_len = self.op.reduction_axis_len()

        # TODO: for each op, emit eligible configs   
        for (reg_tile, smem_tile, reduction_size) in self.activeblock_db.set_store:
            reg_tile = list(reg_tile)
            smem_tile = list(smem_tile)
            reduction_size = list(reduction_size)

            # matmul specific
            if reduction_size[0] > r_len or reg_tile[0] >= out_shape[0] * 2 or reg_tile[1] >= out_shape[1] * 2:
                continue
            if smem_tile[0] >= out_shape[0] * 2 or smem_tile[1] >= out_shape[1] * 2:
                continue
        
            config_sche = Schedule(dim_size=2, spatial_axis=self.saxis_names, reduce_axis=self.raxis_names)
            config_sche.add_tile(mem_level=2, dim=[1,1], reduction_dict={'k':1})
            config_sche.add_tile(mem_level=1, dim=reg_tile, reduction_dict={'k':1})
            config_sche.add_tile(mem_level=0, dim=smem_tile, reduction_dict={'k':reduction_size[0]})

            wh_key = config_sche.dump_to_string()
            if wh_key in config_wh:
                continue
            config_wh.add(wh_key)
            
            try:
                config_sche.compute_peak_performance = self.cost_model.get_compute_peak_performance(config_sche, self.compute_db)
                config_sche.glbmem_bandwidth = self.cost_model.get_glbmem_bandwidth(config_sche, self.small_glbmem_db)
                config_sche.smem_bandwidth = self.cost_model.get_smem_bandwidth(config_sche)
                config_sche.active_blocks_per_sm = self.activeblock_db.lookup(reg_tile, smem_tile, reduction_size)
            except:
                # print("config: {} not found in database!", config_wh)
                continue

            if eligible(self.op, self.arch, config_sche, 0):
                # matmul specific
                # TODO: if we do 2^n padding, twopower_k's code can be removed.
                #twopower_k = 1
                #while twopower_k < self.op._K:
                #    twopower_k = twopower_k << 1
                #if twopower_k > self.op._K:
                #    twopower_k = twopower_k >> 1
                #k_reduction_size = [twopower_k] + reduction_size
                #config_sche.active_blocks_per_sm = self.activeblock_db.lookup(reg_tile, smem_tile, k_reduction_size)
                self.raw_configs.append(config_sche)
                    
        print("total {} raw configs".format(len(self.raw_configs)))
        return self.raw_configs


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

