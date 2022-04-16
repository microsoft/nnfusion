from config import *
from cost_model import *
from .PolicyBase import *
import math
import numpy
from utils import *
from compute_db import *


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

class ConstructionPolicyPlainRTV2(PolicyBase):
    """
        Constructing tiling schedules using DFS for sizes
        expand tiling sizes for alignment requirement
    """
    def __init__(self, op, arch, smem_tiling=False, reg_tiling=False, st_align=False, data_type="float", tile_tensor="output"):
        self.op = op
        self.arch = arch
        self.tile_tensor = tile_tensor
        self.num_level = arch.num_level        

        self.smem_tiling = smem_tiling
        self.reg_tiling = reg_tiling
        self.st_align = st_align


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
                    while new_d1 <= d_cap and not find:
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

    def IsComputeAligned(self, rprog, mem_level):
        if mem_level == 1:
            return True
        if mem_level == 0:
            num_threads = rprog.GetParallelism(1)
            if self.op.use_tc:
                num_threads *= 32
            return num_threads % 32 == 0 and num_threads >= (self.arch.warp_size * self.arch.compute_sm_partition[1])

    def EnlargeTile(self, rprog, mem_level, init=False):
        key = rprog.Dump()
        if mem_level == -1:
            if key not in self.in_results and len(self.in_results) < self.TOPK:
                self.in_results.add(key)
                self.top_results.append(rprog)
            return
        if len(self.in_results) >= self.TOPK:
            return

        if key in self.visited:
            return
        self.visited.add(key)
       
        rtile = rprog.GetTile(mem_level)
        if init:
            aligned_sizes = self.GetNextAlignedSteps(rprog, mem_level, True)
            shape = rtile.Dimensions().copy()
            for i in range(len(aligned_sizes)):
                shape[i] = aligned_sizes[i]
            rtile = rTile(rtile.expr, shape, self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
            rprog.UpdateTile(rtile, mem_level)
        aligned_sizes = self.GetNextAlignedSteps(rprog, mem_level, False)
        shape = rtile.Dimensions()
   
        for d in range(len(self.op.Dimensions()) - 1, -1, -1):
            new_rprog = rprog.copy()      
            new_shape = shape.copy()
            new_shape[d] = aligned_sizes[d]
            new_rtile = rTile(rtile.expr, new_shape, self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
            new_rprog.UpdateTile(new_rtile, mem_level)
            self.update_rtile_storage_padding(new_rprog, self.arch, mem_level, self.smem_tiling, self.reg_tiling, self.st_align)
            if not self.Eligible(new_rprog, mem_level):
                rprog.AddTile(mem_level - 1, rTile(rtile.expr, rtile.Dimensions(), self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor()))
                self.EnlargeTile(rprog, mem_level - 1, True)
                rprog.DeleteTile(mem_level - 1)

            elif self.IsComputeAligned(new_rprog, mem_level):
                new_rprog.AddTile(mem_level - 1, rTile(new_rtile.expr, new_rtile.Dimensions(), self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor()))
                self.EnlargeTile(new_rprog, mem_level - 1, True)
                new_rprog.DeleteTile(mem_level - 1)
            
            else:
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

    def emit_raw_configs(self, padding_threshold=0):
        """
            using BFS to iteratively search for all Eligible candidates level by level
        """
        # initialize uni tiles
        mem_level = self.num_level - 1
        uniTile = rTile(self.op.expr, [1 for _ in range(len(self.op.Dimensions()))], self.op.SAxis(), self.op.RAxis(), self.op.GetTvmOutTensor())
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
        while th <= 1 and len(self.all_results) < topk:
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
        
        return self.all_results[:self.TOPK]
    
    def emit_config(self, topk, min_parallism=80):
        return self.emit_config_without_trails(topk)
        
                