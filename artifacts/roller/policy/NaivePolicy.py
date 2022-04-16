from config import *
from cost_model import *
from .PolicyBase import *

Tiling = "regular" # 'all' or 'regular'

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
        if op.memory_footprint(dim, reduc, tile_tensor) > arch.mem_cap(0):
            return False
    return True

def Area(tile):
    ret = 1
    for dim in tile:
        ret *= dim
    return ret

class NaivePolicy(PolicyBase):
    """
        A naive policy implementation:
            1, enumerate all raw candidates
            2, prune some candidates with very straight-forward strategies
            3, calculate the theoretical performance of all configs
    """
    def __init__(self, op, arch, saxis_names, raxis_names, enum_reduce=False, tile_tensor="output"):
        self.op = op
        self.tile_tensor = tile_tensor
        self.arch = arch
        self.dim_size = len(self.op.dims[self.tile_tensor])
        self.num_level = arch.num_level
        self.saxis_names = saxis_names
        self.raxis_names = raxis_names
        self.enum_reduce = enum_reduce
        self.cost_model = WarpBasedCostModel(op, arch)

    def BFS_Tiles(self, last_level_schedules, mem_level, tile_tensor = "output"):
        # Run BFS and return all configs that satisfy memory limit
        # last_level_schedules: the base dim lists
        
        dim_size = len(self.op.dims[tile_tensor])
        candidates = [] # list of Schedule()

        for ll_schedule in last_level_schedules:
            # using tile from last level as a base tile
            base_dim, _ = list(ll_schedule.get_tile(mem_level + 1))
            dim_queue = [base_dim]
            visited = set()
            while len(dim_queue) > 0:
                cur_dim = dim_queue.pop(0)
                if tuple(cur_dim) in visited:
                    continue
                visited.add(tuple(cur_dim))
                reduction_dict = {name: 1 for name in self.raxis_names}
                reduction_dict_choices = [reduction_dict]

                if self.enum_reduce:
                    for rlen in [2, 4, 8, 16, 32, 64, 128]:
                        rdict = {name: 1 for name in self.raxis_names}
                        if len(self.raxis_names) > 0:
                            rdict[self.raxis_names[0]] = min(rlen, self.op.reduction_axis_len())
                        reduction_dict_choices.append(rdict)
                        if rlen > self.op.reduction_axis_len():
                            break

                for rdict in reduction_dict_choices:
                    new_schedule = ll_schedule.copy()
                    new_schedule.add_tile(mem_level, cur_dim, rdict)
                    if not eligible(self.op, self.arch, new_schedule, mem_level):
                        continue
                    candidates.append(new_schedule)

                for d in range(dim_size, 0, -1):
                    if cur_dim[d - 1] >= self.op.dims[tile_tensor][d - 1]:
                        continue
                    next_dim = list(cur_dim).copy()
                    if Tiling == "regular":
                        next_dim[d - 1] = min(next_dim[d - 1] + cur_dim[d - 1], self.op.dims[tile_tensor][d - 1])
                    elif Tiling == "all":
                        next_dim[d - 1] = min(next_dim[d - 1] + base_dim[d - 1], self.op.dims[tile_tensor][d - 1])
                    dim_queue.append(next_dim)
        return candidates

    def emit_raw_candidates_level(self, last_level_schedules, mem_level):
        # enumerate all raw candidates using BFS given a level
        # return all candidates that satisfy the memory footprint constraints
        raw_candidates = self.BFS_Tiles(last_level_schedules, mem_level, self.tile_tensor)
        return raw_candidates

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
            schedule_candidates = self.prune(schedule_candidates, mem_level)
            #print(len(schedule_candidates))
            #for c in schedule_candidates:
            #    print(c.dump_to_string())
            #print("---------------------------------------------------")
            mem_level -= 1

        return schedule_candidates

    def prune(self, raw_candidates, mem_level):
        """
            simple pruning strategy:
            1, discard configs with more than 256 or less than 64 threads per smem tile
            2, discard reg tiles that are too small or too large
        """
        new_schedule = []
        full_dim = self.op.dims["output"]
        for candidate in raw_candidates:
            Pruned = False
            # discard candidates with too small reg tile sizes
            last_tile, _ = candidate.get_tile(mem_level + 1)
            this_tile, _ = candidate.get_tile(mem_level)
            for d in range(len(last_tile)):
                if this_tile[d] % last_tile[d] > 0:
                    Pruned = True
            if mem_level == 1:
                reg_tile, _ = candidate.get_tile(1)
                if Area(reg_tile) < 1 or Area(reg_tile) > 32:
                    Pruned = True
            # discard candidates with <64 or >256 block size
            if mem_level == 0:
                block_size = candidate.subtile_count(0, 1)
                if block_size <= 32 or block_size > 512:
                    Pruned = True
                #if block_size % 32 > 0:
                #    Pruned = True
            if not Pruned:
                new_schedule.append(candidate)
        return new_schedule


    def emit_config_without_trails(self, topk):
        # directly compute the theoretical performance for each raw configs and pick the optimal k configs
        # will call self.emit_raw_configs()
        raw_configs = self.emit_raw_configs()
        print("{} raw configs generated after pruning".format(len(raw_configs)))
        perf_config = []
        for config in raw_configs:
            theo_perf = self.cost_model.Theoretical_Perf(config, pure_memory=True)
            perf_config.append((theo_perf, config))

        def sort_key(a):
            return a[0]
        perf_config.sort(key=sort_key)
            
        return [x for (_, x) in perf_config[:topk]]

