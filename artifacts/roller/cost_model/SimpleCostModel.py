from config import *
from op import *
from .CostModelBase import *

def _align(x, unit_size):
    return ((x - 1) // unit_size + 1) * unit_size

def _num_tiles(large, small):
    ret = 1
    for i in range(len(large)):
        ret = ret * ((large[i] - 1) // small[i] + 1)
    return ret

def _fused_inner_axis(full_dim, tile_dim):
    dim_size = len(tile_dim)
    d = dim_size - 1
    fused_dim = 1
    while d >= 0:
        fused_dim *= tile_dim[d]
        d -= 1
        if tile_dim[d + 1] != full_dim[d + 1]:
            break
    return fused_dim

def _fix_reduction_dimension(subtensor, reduction_size):
    for raxis in reduction_size:
        for i in range(len(subtensor["axis"])):
            if subtensor["axis"][i] == raxis:
                subtensor["dim"][i] = reduction_size[raxis]
    return subtensor


class SimpleCostModel(CostModelBase):
    """
        A very simple implementation of the cost model
        The compute estimation:
            1, assumes that we can always achieve peak FLOPS if 
            2, estimates the penalty due to non-aligned warps
        The memory estimation:
            1, assumes that we can always achieve peak throughput if all transactions are utilized
            2, estimates the penalty due to non-aligned transactions
            3, does not consider bank conflicts
    """
    def __init__(self, op, arch):
        self.op = op
        self.arch = arch

    def compute_estimate(self, schedule, tile_tensor = "output"):
        """
            estimate the latency of compute
        """
        full_dim = self.op.dims[tile_tensor]
        smem_tile_dim, _ = schedule.get_tile(0)
        reg_tile_dim, _ = schedule.get_tile(1)
        grid_size = _num_tiles(full_dim, smem_tile_dim)
        block_size = _num_tiles(smem_tile_dim, reg_tile_dim)
        compute_throughput = self.arch.peak_flops()

        # Compute throughput penalized due to non-aligned warps
        aligned_threads = _align(block_size, 32)
        compute_penalty = aligned_threads / block_size

        # Calculate compute latency
        compute_workload_thread = self.op.compute_workload(reg_tile_dim, tile_tensor)
        compute_workload_raw = compute_workload_thread * grid_size * block_size
        
        return compute_workload_raw * compute_penalty / compute_throughput


    def memory_estimate(self, schedule, mem_level, tile_tensor = "output"):
        """
            estimate the latency of memory latency for a given memory level
        """
        full_dim = self.op.dims[tile_tensor]
        tile_dim, reduction_size = schedule.get_tile(mem_level)
        num_tiles = _num_tiles(full_dim, tile_dim)
        memory_latency_tile = 0
        
        workloads = self.op.memory_workload(tile_dim, tile_tensor, mem_level)
        mem_transaction_size = self.arch.Transaction_size[mem_level]
        # the dimension of subtensors
        tile_subtensors = self.op.subtensor_size(tile_dim, tile_tensor)
        # Calculate bank conflict when estimating smem-reg throughput given dram-smem tile is ready
        #if mem_level == 0 and level == 1:
        #    bc_penalty = Bank_Conflict_Penalty(schedule, self)
        
        for tensor_name in workloads:
            rw_bytes = workloads[tensor_name]
            # Memory bandwidth penalized due to non-aligned transactions
            memory_penalty = 1
            if mem_transaction_size > 1:
                full_subtensor = self.op.dims[tensor_name]
                tile_subtensor = tile_subtensors[tensor_name]
                tile_subtensor = _fix_reduction_dimension(tile_subtensor, reduction_size)
                inner_axis = _fused_inner_axis(full_subtensor, tile_subtensor["dim"])
                aligned_inner_axis = _align(inner_axis, mem_transaction_size)
                memory_penalty = aligned_inner_axis / inner_axis
            memory_latency_tensor = rw_bytes * memory_penalty / self.arch.memory_bw(mem_level)
            # Memory bandwidth penalized due to bank conflicts
            #if mem_level == 0 and level == 1 and tensor_name != "output":
            #    memory_latency_tensor *= bc_penalty[tensor_name]
                #print("bc penalty {}, non-aligned penalty {}".format(bc_penalty[tensor_name], memory_penalty))
            memory_latency_tile += memory_latency_tensor

        return memory_latency_tile * num_tiles
