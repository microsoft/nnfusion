from roller.config import *
from roller.cost_model import *
from .policy_base import *
import math
import numpy
from roller.utils import *
import time
# from compute_db import *


def lcm(x, y):
    return x * y // math.gcd(x, y)

def num_tiles(large_tile, base_tile):
    ret = 1
    for d1, d2 in zip(large_tile, base_tile):
        ret *= math.ceil(d1 / d2)
    return ret

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

class PolicyRTV2(PolicyBase):
    """
        Constructing tiling schedules using DFS, hardcode reduction step size for now
    """
    def __init__(self, op, arch, smem_tiling=False, reg_tiling=False, st_align=False, padding_threshold_cap=1.0,
               shrink_tiny=True, tile_tensor="output"):
        self.op = op
        self.arch = arch
        self.tile_tensor = tile_tensor
        self.num_level = arch.num_level
        self.th_cap = padding_threshold_cap
        self.shrink_tiny = shrink_tiny

        # for storage align
        self.smem_tiling = smem_tiling
        self.reg_tiling = reg_tiling
        self.st_align = st_align

        # self.computedb = ComputeDB(self.op.tvm_op, self.arch)


        self.choosen_results = set()
        self.allow_peak = True
    def DataReuseScore(self, rprog, aligned_size, mem_level):
        """
        return a list of scores on each dimension
        """
        score = []
        rtile = rprog.GetTile(mem_level)

        def traffic_and_footprint(op, rtile, rprog, mem_level):
            rprog.UpdateTile(rtile, mem_level)
            self.update_rtile_storage_padding(rprog, self.arch, mem_level, smem_tiling=self.smem_tiling, reg_tiling=self.reg_tiling, st_align=self.st_align)           
            memory_workload = op.MemWorkload(rtile)
            tranffic = sum(memory_workload[0]) / sum(memory_workload[1])
            footprint = op.MemFootprint(rtile)
            return tranffic, footprint
        
        Q, F = traffic_and_footprint(self.op, rtile, rprog, mem_level)  
        dims = rtile.Dimensions()
        for d in range(len(dims)):
            new_dims = dims.copy()
            new_dims[d] = aligned_size[d] 
            new_rtile = rTile(rtile.tvm_op, new_dims, self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
            new_Q, new_F = traffic_and_footprint(self.op, new_rtile, rprog, mem_level) 
            if Q == new_Q or F == new_F:
                s = 0
            else:
                s = (Q - new_Q)/(new_F - F)
            score.append(s)

        rprog.UpdateTile(rtile, mem_level)
        return score

    def GetNextAlignedSteps(self, rprog, mem_level, init):
        base_rtile = rprog.GetTile(mem_level)
        # hardcode for two-level tiling now
        full_sdim = self.op.SDimensions()
        full_rdim = self.op.RDimensions()
        base_sdims = base_rtile.SDimensions()
        base_rdims = base_rtile.RDimensions()

        base_saxis = base_rtile.SAxis()
        base_raxis = base_rtile.RAxis()
        ins_inner_most_axis = set()
        for ian in base_rtile.InputAxis():
            ins_inner_most_axis.add(ian[-1])

        def align(axis_name, dim_size):
            if mem_level == 0:
                # align to reg_tile size
                reg_tile = rprog.GetTile(mem_level + 1)
                reg_dim_size = reg_tile.GetAxisLen(axis_name)
                aligned_d = math.ceil(dim_size / reg_dim_size) * reg_dim_size
                # align to transaction size
                if axis_name in ins_inner_most_axis:
                    transaction_size = self.arch.transaction_size[0] // self.op.InputTypeSize()
                    if self.op.use_tc:
                        transaction_size = 64
                    aligned_d = lcm(dim_size, transaction_size)
                # print(axis_name, dim_size, aligned_d)
                return aligned_d
            else:
                return dim_size

        def tolerate_pad(tile_d, op_d):
            padded_dim = math.ceil(op_d / tile_d) * tile_d
            return padded_dim <= op_d * (1 + self.padding_threshold)

        # reg level, using power of 2 tiles
        def get_aligned_size(is_reduction):
            aligned_axis_size = []
            # base_dims = base_sdims if not is_reduction else base_rdims
            base_axis = base_saxis if not is_reduction else base_raxis
            full_dim = full_sdim if not is_reduction else full_rdim
            for i in range(len(base_axis)):
                new_d_candidate = ()
                a = base_axis[i]
                d = base_rtile.GetAxisLen(a)
                full_d = full_dim[i]
                if mem_level == 1:
                    d_cap = min(full_d, 32) # cap 32
                elif mem_level == 0:
                    d_cap = full_d
                    if is_reduction:
                        d_cap = min(full_d, 128)
                if init:
                    new_d = min(align(a, d), d_cap)
                    aligned_axis_size.append(new_d)
                else:
                    new_d1 = d + 1
                    find = False
                    while new_d1 <= d_cap and not find:#
                        new_d1 = align(a, new_d1)
                        if tolerate_pad(new_d1, full_d):
                            new_d_candidate = new_d_candidate + (new_d1,)
                            find = True
                        else:
                            new_d1 += 1
                    mul = 2
                    new_d2 = d * mul
                    find = False
                    while new_d2 <= d_cap and not find:
                        new_d2 = align(a, new_d2)
                        if tolerate_pad(new_d2, full_d):
                            new_d_candidate = new_d_candidate + (new_d2,)
                            find = True
                        else:
                            mul += 1
                            new_d2 = d * mul
                    if len(new_d_candidate) == 0:
                        aligned_axis_size.append(d_cap)
                    else:
                        aligned_axis_size.append(min(new_d_candidate))

            return aligned_axis_size
            
        aligned_saxis_size = get_aligned_size(False) 
        aligned_raxis_size = get_aligned_size(True)
        return aligned_saxis_size + aligned_raxis_size

    def IsComputeIntensive(self, rprog, mem_level):
        """
        return True if the schedule is compute intensive
        """
        regTile = rprog.GetTile(self.arch.num_level - 1)
        thisTile = rprog.GetTile(mem_level)
        
        compute_workload = self.op.ComputeWorkload(thisTile)
        # compute_throughput = self.compute_db.lookup(64,8,8) * self.arch.compute_max_core[0]
        compute_throughput = self.arch.peak_flops# / self.arch.compute_max_core[0]
        if self.op.use_tc:
            compute_throughput = self.arch.peak_tc_flops
        compute_latency = compute_workload / compute_throughput

        memory_tensors = self.op.MemWorkload(thisTile)
        memory_workload = sum(memory_tensors[0]) + sum(memory_tensors[1])

        memory_throughput = self.arch.memory_bw(mem_level)# / self.arch.mem_max_core[0]
        memory_latency = memory_workload / memory_throughput
        return compute_latency > memory_latency
        
    def IsThroughputSatisfied(self, rprog, mem_level):
        # if mem_level == 1:
        #     return False
        throughput = self.Throughput(rprog, mem_level)
        if throughput == -1:
            return False  
        satisfied = self.min_throughput < throughput
        # print("throughput satisified:", satisfied, self.min_throughput, throughput, rprog.Dump())
        return satisfied

    def Throughput(self, rprog, mem_level):
        compute_perf = self.computedb.ComputePerf(rprog)
        if compute_perf == -1:
            return -1
        compute_throughput = 32 * compute_perf # reg tile number/ ms
        reg_tile = rprog.GetTile(1)
        reg_throughput = self.arch.memory_bw(1) * 1024 * 1024 * 1024 / self.arch.mem_max_core[0] /self.op.MemFootprint(reg_tile) / 1000
        if mem_level == 1:
            cur_min_throughput = min(compute_throughput, reg_throughput)
        elif mem_level == 0:
            sm_tile = rprog.GetTile(0)
            num_reg_per_sm = rprog.GetParallelism(1)
            sm_throughput = self.arch.memory_bw(0) * 1024 * 1024 * 1024 / self.arch.mem_max_core[0] / self.op.MemFootprint(sm_tile) * num_reg_per_sm / 1000
            cur_min_throughput = min(compute_throughput, reg_throughput, sm_throughput)
        return cur_min_throughput
        
    def IsComputeAligned(self, rprog, mem_level):
        if mem_level == 1:
            return True
        if mem_level == 0:
            num_threads = rprog.GetParallelism(1)
            if self.op.use_tc:
                num_threads *= 32       
            return num_threads % (self.arch.warp_size * self.arch.compute_sm_partition[1]) == 0
            print(num_threads)
            return num_threads >= (self.arch.warp_size * self.arch.compute_sm_partition[1]) and num_threads % 32 == 0

    def EnlargeTile(self, rprog, mem_level, init=False):
        key = rprog.Dump()
        # print(key)
        if mem_level == -1:
            if key not in self.in_results and len(self.in_results) < self.TOPK and key not in self.choosen_results:
                self.in_results.add(key)
                # print(key)
                # print("******************************************")
                self.top_results.append(rprog.copy())
            return
        if len(self.in_results) >= self.TOPK:
            return

        if key in self.visited:

            return
        self.visited.add(key)

        r_scores = []
        rtile = rprog.GetTile(mem_level)
        if init:
            aligned_sizes = self.GetNextAlignedSteps(rprog, mem_level, True)
            shape = rtile.Dimensions().copy()
            for i in range(len(aligned_sizes)):
                shape[i] = aligned_sizes[i]
            rtile = rTile(rtile.tvm_op, shape, self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
            rprog.UpdateTile(rtile, mem_level)
        aligned_sizes = self.GetNextAlignedSteps(rprog, mem_level, False)
        # if mem_level == 0:
        #     print(aligned_sizes)

        shape = rtile.Dimensions()
        r_scores = self.DataReuseScore(rprog, aligned_sizes, mem_level)
        x = numpy.array(r_scores)
        dim_order = numpy.argsort(x)

        # enumerate from dimensions with highest scores
        for d in reversed(dim_order): 
            new_rprog = rprog.copy()      
            new_shape = shape.copy()
            new_shape[d] = aligned_sizes[d]
            new_rtile = rTile(rtile.tvm_op, new_shape, self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
            new_rprog.UpdateTile(new_rtile, mem_level)
            self.update_rtile_storage_padding(new_rprog, self.arch, mem_level, self.smem_tiling, self.reg_tiling, self.st_align)

            if not self.Eligible(new_rprog, mem_level):
                # print("aaaaaaaaaaaaaa")
                if self.IsComputeAligned(rprog, mem_level) and self.allow_peak:
                    # print("aaaaaaaaaaaaaa111111111111111")
                    rprog.AddTile(mem_level - 1, rTile(rtile.tvm_op, rtile.Dimensions(), self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor()))
                    self.EnlargeTile(rprog, mem_level - 1, True)
                    rprog.DeleteTile(mem_level - 1)

            # elif self.IsThroughputSatisfied(new_rprog, mem_level) and self.IsComputeAligned(new_rprog, mem_level):
            if self.IsComputeIntensive(new_rprog, mem_level) and self.IsComputeAligned(new_rprog, mem_level):
                # print("bbbbbbbbbbbbb")
                new_rprog.AddTile(mem_level - 1, rTile(new_rtile.tvm_op, new_rtile.Dimensions(), self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor()))
                self.EnlargeTile(new_rprog, mem_level - 1, True)
                new_rprog.DeleteTile(mem_level - 1)
          
            else:
                # print("ccccccccccc")
                self.EnlargeTile(new_rprog, mem_level, False)

    def Eligible(self, rprog, mem_level, tile_tensor = "output"):
        # a tile size is not Eligible if
        # 1, the # of used registers exceed the capacity
        # 2, the size of memory used at certain level exceeds the capacity
        if mem_level == 1: # reg tiling
            # check register usage
            regTile = rprog.GetTile(mem_level)
            reg_usage_2 = self.op.RegUsage(regTile, tile_tensor)
            if reg_usage_2 > self.arch._reg_cap(1):
                return False
        if mem_level == 0: # smem tiling
            num_threads = rprog.GetParallelism(mem_level + 1)
            regTile = rprog.GetTile(mem_level + 1)
            # check register usage
            reg_usage_2 = self.op.RegUsage(regTile, tile_tensor)
            reg_usage_1 = reg_usage_2 * num_threads
            if reg_usage_1 > self.arch._reg_cap(0):
                return False
            # check shared memory usage
            smemTile = rprog.GetTile(mem_level)
            if self.op.MemFootprint(smemTile, tile_tensor) >= self.arch.mem_cap(0):
                return False
            # remove over sized warps
            if self.op.use_tc:
                num_threads *= 32
            return num_threads <= 512
        return True

    def try_shrink(self, rprog):
        sdim = len(rprog.saxis)
        pre_grid_size = 0
        while True:
            smem_tile_dim = rprog.GetTile(0).Dimensions()
            # validate grid size, return if grid size >= # of CUs
            grid_size = rprog.GetParallelism(0)
            print("grid_size: ", grid_size)
            print(rprog.Dump())

            if grid_size >= self.arch.compute_max_core[0] or grid_size == pre_grid_size:
                return rprog

            # otherwise, shrink dimentions based on data reuse
            #print("try shrink small config: {}".format(schedule.dump_to_string()))
            
            # r_scores = self.DataReuseScore(rprog, 0)[:sdim]
            aligned_sizes = self.GetNextAlignedSteps(rprog, 0, False)
            r_scores = self.DataReuseScore(rprog, aligned_sizes, 0)[:sdim]
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
                new_tile = rTile(tile.tvm_op, tile_sdim + tile_rdim, self.op.SAxis(),
                                self.op.RAxis(), self.op.GetTvmOutTensor())
                rprog.UpdateTile(new_tile, l)
            pre_grid_size = grid_size
        # print("config after shrinking: {}".format(rprog.Dump()))

    def emit_raw_configs(self, padding_threshold=0):
        """
            using BFS to iteratively search for all Eligible candidates level by level
        """
        # initialize uni tiles
        mem_level = self.num_level - 1
        # uniTile = rTile(self.op.tvm_op, [1 for _ in range(self.tile_dim + self.step_dim)], self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
        uniTile = rTile(self.op.tvm_op, self.op.GetUniSchedule(), self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor()) 
        uniProg = rProg(self.arch.num_level, self.op)
        uniProg.AddTile(self.num_level, uniTile)
        uniProg.AddTile(self.num_level - 1, uniTile)

        # initialize key hyperparameters
        self.padding_threshold = padding_threshold
        self.top_results = []
        self.visited = set()
        self.EnlargeTile(uniProg, mem_level, False)
        return self.top_results

    def emit_config_without_trails(self, topk):
        # directly compute the theoretical performance for each raw configs and pick the optimal k configs
        # will call self.emit_raw_configs()
        self.all_results = []
        self.in_results = set()
        self.TOPK = topk
        th = 0           
        round = 0 
        start = time.time()
        while th < self.th_cap and len(self.all_results) < topk:
            self.emit_raw_configs(th)
            round += 1
            if len(self.top_results) == 0:
                print("failed to find results with padding threshold {}".format(th))
                pass
            else:
                print("found {} results in {} round with threshold {}".format(len(self.top_results), round, th))
            # add current results to all
            for result in self.top_results:
                self.all_results.append(result)
            th += 0.2 
            if time.time() - start > 60:
                break
        
        # return self.all_results[:self.TOPK]

        count = 1
        # for rprog in self.all_results:
        #     print(count, "###", rprog.Dump())
        #     count +=1

        # handling small configs
        if self.shrink_tiny:
            for rprog in self.all_results:
                if count == 1:
                    rprog = self.try_shrink(rprog)
                count += 1
                # print(rprog.Dump())
        return self.all_results[:self.TOPK]

    def emit_config(self, topk, min_parallism=80):
        th = 0.9
        self.min_throughput = 1e10
        self.choosen_results = set()
        self.allow_peak = True
        results = self.emit_config_without_trails(topk)
        if len(results) == 0:
            return results
        self.allow_peak = False
        final_results = results
        chosen_rprog = None
        parallism = 1e10
        for rprog in results:
            throughput = self.Throughput(rprog, 0) 
            if throughput != -1:
                chosen_rprog = rprog
                self.min_throughput = throughput
                parallism = chosen_rprog.GetParallelism(0)
                self.choosen_results.add(chosen_rprog.Dump())
                break
        while parallism < min_parallism and chosen_rprog:
            print("cur parallism: ", parallism, "min_throughput: ", self.min_throughput)
            results = self.emit_config_without_trails(topk)
            chosen_rprog = None
            for rprog in results:
                throughput = self.Throughput(rprog, 0) 
                if throughput != -1:
                    # print(rprog.Dump(), throughput)
                    chosen_rprog = rprog
                    self.min_throughput = min(self.min_throughput, throughput) * th
                    parallism = chosen_rprog.GetParallelism(0)
                    self.choosen_results.add(chosen_rprog.Dump())
                    final_results = results
                    break
        return final_results      
            
        





