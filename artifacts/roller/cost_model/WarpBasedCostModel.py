from config import *
from op import *
from arch import *
from .CostModelBase import *
from tvm import te
import math
from .glbmem import DRAM_latency


class WarpBasedCostModel(CostModelBase):
    """
        The cost model based on profiled warp-level performance
        The compute estimation:
            1, estimating kernel compute latency based on profiled single-warp FLOPS
            2, scale up compute FLOPS for each tiling level
        TODO: add descriptions for memory estimations
        The memory estimation:
            1, 
    """
    def __init__(self, op, arch):
        self.op = op
        self.arch = arch
    
    def tile_size(self, dim_list):
        ret = 1
        for d in dim_list:
            ret *= d
        return ret
    
    def num_tiles(self, large_tile, base_tile):
        ret = 1
        for d1, d2 in zip(large_tile, base_tile):
            ret *= d1 // d2
        return ret
    
    def block_warps_num(self, reg_tile, smem_tile):
        # round up to a multiple of arch_warp_size
        return int(math.ceil(self.num_tiles(smem_tile, reg_tile) / self.arch._warp_size))
    
    def get_glbmem_bandwidth(self, schedule, small_glbmem_db, tile_tensor="output"):
        """
            If parallelism is small, look up the database, otherwise the peak bandwidth
        """
        smem_tile = schedule.get_tile(0)[0]
        reg_tile = schedule.get_tile(1)[0]
        warps_num = self.block_warps_num(reg_tile, smem_tile)
        grid_size = self.op.get_grid_size(smem_tile, tile_tensor)

        arch_sm_num = self.arch._glbmem_sm_partition[0]
        arch_glm_saturate_warps = self.arch._glbmem_sm_partition[1]
        # If parallelism is insufficient, lookuup throughput
        if (grid_size * warps_num) < (arch_sm_num * arch_glm_saturate_warps):
            bandwidth = small_glbmem_db.lookup(warps_num, grid_size)
        else:
            bandwidth = self.arch.memory_bw(0)
        return bandwidth
    
    def get_compute_peak_performance(self, schedule, compute_db):
        """
            Estimate computation peak performance by a thread-block performance
        """
        smem_tile, reduction_size = schedule.get_tile(0)
        reg_tile = schedule.get_tile(1)[0]
        reg_size = self.tile_size(reg_tile)
        warps_num = self.block_warps_num(reg_tile, smem_tile)
        compute_pf = compute_db.lookup(reg_size, warps_num, reduction_size["k"])

        arch_sm_num = self.arch._compute_sm_partition[0]
        arch_sm_partition = self.arch._compute_sm_partition[1]
        if warps_num < arch_sm_partition:
            active_warps = warps_num
        else:
            active_warps = arch_sm_partition
        peak_performance = compute_pf / active_warps * arch_sm_num * arch_sm_partition
        return peak_performance
    

    def get_smem_bandwidth(self, schedule):
        """
            Estimate the shared memory bandwidth by bank conflicts
        """
        warp_size = self.arch._warp_size
        bank_size = self.arch._transaction_size[1] // 4
        base_bandwidth = self.arch.memory_bw(1)
        reg_tile, _ = schedule.get_tile(1)
        smem_tile, reduction_size = schedule.get_tile(0)
        sy, sx = smem_tile[0] // reg_tile[0], smem_tile[1] // reg_tile[1]
        total_ld_inst = reg_tile[0] + reg_tile[1]
        
        # load 32/x elements from a (?, k) subtensor
        # addr: [0, k, 2k, ...]
        num_ele = warp_size // min(sx, warp_size)
        bc_A = max(1, num_ele / (bank_size / reduction_size["k"]))
        bc_B = 1

        tp_penalty = total_ld_inst / (reg_tile[0] * bc_A + reg_tile[1] * bc_B)
        smem_bandwidth = base_bandwidth * tp_penalty
        return smem_bandwidth


    def compute_estimate(self, schedule, tile_tensor = "output"):
        """
            estimate the latency of compute
        """  

        if self.arch.num_level == 2:
            smem_tile, reduction_size = schedule.get_tile(0)
            reg_tile = schedule.get_tile(1)[0]
            # print(smem_tile, reg_tile)

            arch_warp_size = self.arch._warp_size
            arch_sm_num = self.arch._compute_sm_partition[0]
            arch_sm_partition = self.arch._compute_sm_partition[1]
            arch_block_schedule_way = self.arch._compute_block_schedule_way
            active_blocks_per_sm = schedule.active_blocks_per_sm
            # print(arch_warp_size, arch_sm_num, arch_sm_partition, arch_block_schedule_way, active_blocks_per_sm)

            block_warps_num =  self.block_warps_num(reg_tile, smem_tile)
            block_size = block_warps_num * arch_warp_size
            grid_size = self.op.get_grid_size(smem_tile, tile_tensor)
            # print("warp num:", block_warps_num, "block size:", block_size, "grid size:", grid_size)

            peak_performance = schedule.compute_peak_performance
            # print(peak_performance)
            
            # determine the block schedule way
            # always warp schedule
            if len(arch_block_schedule_way) == 1:
                block_schedule_unit = arch_block_schedule_way[0]
            # warp schedule (reach active blocks) -> active block schedule
            elif len(arch_block_schedule_way) == 2:
                if grid_size <= (active_blocks_per_sm * arch_sm_num):
                    block_schedule_unit = arch_block_schedule_way[0]
                else:
                    block_schedule_unit = arch_block_schedule_way[1]
            else:
                raise NotImplementedError
            # print(block_schedule_unit)
            
            if block_schedule_unit == "warp":
                if block_warps_num < arch_sm_partition:
                    active_warps = block_warps_num
                else:
                    active_warps = arch_sm_partition
                sche_units_num = grid_size * active_warps
                sche_units_total = arch_sm_num * arch_sm_partition

            elif block_schedule_unit == "active block":
                sche_units_num = grid_size
                sche_units_total = arch_sm_num * active_blocks_per_sm
            
            else:
                raise NotImplementedError
            # print(sche_units_num, sche_units_total)
            
            compute_kernel_tp = peak_performance * sche_units_num / (math.ceil(sche_units_num / sche_units_total) * sche_units_total)

            compute_workload_total = self.op.compute_workload(reg_tile, reduction_size, tile_tensor) * block_size * grid_size  # / 1000000000 to GFLOPS
            compute_time_ns = compute_workload_total / compute_kernel_tp  # * 1000000000 to ns
            # print(compute_kernel_tp, compute_workload_total)
            return compute_time_ns
        
        else:
            raise NotImplementedError


    def memory_estimate(self, schedule, mem_level, tile_tensor = "output"):
        """
            estimate the latency of memory latency for a given memory level
        """

        # shared memory level
        if mem_level == 1 and self.arch.num_level == 2:
            smem_tile, reduction_size = schedule.get_tile(0)
            reg_tile = schedule.get_tile(1)[0]
            # print(smem_tile, reg_tile)

            arch_warp_size = self.arch._warp_size
            arch_sm_num = self.arch._smem_sm_partition[0]
            arch_sm_partition = self.arch._smem_sm_partition[1]
            arch_block_schedule_way = self.arch._smem_block_schedule_way
            active_blocks_per_sm = schedule.active_blocks_per_sm
            # print(arch_warp_size, arch_sm_num, arch_sm_partition, arch_block_schedule_way, active_blocks_per_sm)

            block_warps_num =self.block_warps_num(reg_tile, smem_tile)
            block_size = block_warps_num * arch_warp_size
            grid_size = self.op.get_grid_size(smem_tile, tile_tensor)
            # print("warp num:", block_warps_num, "block size:", block_size, "grid size:", grid_size)

            peak_performance = schedule.smem_bandwidth
            # print(peak_performance)
                
            # determine the block schedule way
            # always warp schedule
            if len(arch_block_schedule_way) == 1:
                block_schedule_unit = arch_block_schedule_way[0]
            # warp schedule (reach active blocks) -> active block schedule
            elif len(arch_block_schedule_way) == 2:
                if grid_size <= (active_blocks_per_sm * arch_sm_num):
                    block_schedule_unit = arch_block_schedule_way[0]
                else:
                    block_schedule_unit = arch_block_schedule_way[1]
            else:
                raise NotImplementedError
            # print(block_schedule_unit)
                
            if block_schedule_unit == "warp":
                if block_warps_num < arch_sm_partition:
                    active_warps = block_warps_num
                else:
                    active_warps = arch_sm_partition
                sche_units_num = grid_size * active_warps
                sche_units_total = arch_sm_num * arch_sm_partition

            elif block_schedule_unit == "active block":
                sche_units_num = grid_size
                sche_units_total = arch_sm_num * active_blocks_per_sm
                
            else:
                raise NotImplementedError
            # print(sche_units_num, sche_units_total)
            smem_workload = self.op.memory_workload(reg_tile, reduction_size, 1, tile_tensor)  # from SMEM to REGS
            smem_workload_total = sum([smem_workload[tensor_name] for tensor_name in smem_workload])
            
            smem_kernel_tp = peak_performance * sche_units_num / (math.ceil(sche_units_num / sche_units_total) * sche_units_total)
            smem_bytes = smem_workload_total * block_size * grid_size 
            smem_time_ns = smem_bytes / smem_kernel_tp * 1000000000 / (1024 * 1024 * 1024)
            # print(smem_workload, smem_workload_total, smem_bytes, smem_kernel_tp, smem_time_ns)
            return smem_time_ns
        
        elif mem_level == 0:
            reg_tile_dim, _ = schedule.get_tile(1)
            smem_tile_dim, reduction_size = schedule.get_tile(0)
            return DRAM_latency(self.op,
                                schedule.glbmem_bandwidth,
                                self.arch._transaction_size[0] // 4,
                                self.arch._warp_size,
                                reg_tile_dim,
                                smem_tile_dim,
                                reduction_size
                                )
            
        else:
            raise NotImplementedError