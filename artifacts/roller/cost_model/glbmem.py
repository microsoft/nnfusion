import math

def _align(x, unit_size):
    return ((x - 1) // unit_size + 1) * unit_size

def _fuse_inner_axis(full_dim, tile_dim):
    """
        the product of lengths of continuous inner axes that are full
    """
    assert len(full_dim) == len(tile_dim)
    dim_size = len(tile_dim)
    d = dim_size - 1
    fused_dim = 1
    while d >= 0:
        fused_dim *= tile_dim[d]
        d -= 1
        if tile_dim[d + 1] != full_dim[d + 1]:
            break
    return fused_dim

def _num_tiles(large, small):
    ret = 1
    for i in range(len(large)):
        ret = ret * ((large[i] - 1) // small[i] + 1)
    return ret

def _replace_reduction_dimension(subtensor, reduction_size):
    """
        replace the lengths of reduction axes with reduction step sizes
    """
    for raxis in reduction_size:
        for i in range(len(subtensor["axis"])):
            if subtensor["axis"][i] == raxis:
                subtensor["dim"][i] = reduction_size[raxis]
    return subtensor

def get_warp_dim(warp_size, smem_tile_dim, reg_tile_dim):
    warp_dim = []
    for d in range(len(smem_tile_dim), 0, -1):
        thread_dim = min(warp_size, smem_tile_dim[d - 1] // reg_tile_dim[d - 1])
        warp_dim.insert(0, thread_dim)
        warp_size = math.ceil(warp_size / thread_dim)
    return warp_dim

def DRAM_latency(op, # op.OpBase
                 bandwidth, # global memory bandwidth in GBps
                 transaction_size, # length of a memory transaction in bytes
                 warp_size,
                 reg_tile_dim, # [int]
                 smem_tile_dim, # [int]
                 reduction_size, # {str: reduction_axis_name -> int: reduction step size}
                 tile_tensor="output",
                 data_bytes=4 # set to 4 for using float type by default
    ):
    """
    return the time (in ns) to load/store data from/to DRAM given the tile dimensions & reduction size
    considering the following factors for estimating DRAM latency:
        1, transaction size
    """
    # the total bytes for each subtensor, memory level set to 0 for estimating DRAM
    full_dim = op.dims[tile_tensor]
    num_tiles = _num_tiles(full_dim, smem_tile_dim)
    # print(full_dim, smem_tile_dim, num_tiles)

    memory_workloads = op.memory_workload(smem_tile_dim, reduction_size, 0, tile_tensor)
    # the dimension for each subtensor
    subtensors = op.subtensor_dim(smem_tile_dim, reduction_size, 0, tile_tensor)
    memory_latency_tile = 0

    for tensor_name in memory_workloads:
        # total ld/st bytes 
        rw_bytes = memory_workloads[tensor_name]
        full_st_dim = op.dims[tensor_name]
        subtensor_dim = subtensors[tensor_name]
        # fuse inner axes that are all full, they are continuous in physical memory address
        # print(full_st_dim, subtensor_dim)
        inner_axis = _fuse_inner_axis(full_st_dim, subtensor_dim)
        aligned_inner_axis = _align(inner_axis, transaction_size)
        memory_penalty = aligned_inner_axis / inner_axis
        #print(tensor_name, aligned_inner_axis, inner_axis, rw_bytes, memory_penalty, num_tiles, bandwidth)

        #print(tensor_name, full_st_dim, subtensor_dim)
        #print(rw_bytes, memory_penalty, num_tiles)
        # estimating write memory penalty due to warp transaction
        if tensor_name == "output":
            #print(min(warp_size, smem_tile_dim[1] // reg_tile_dim[1]))
            # get warp shape
            warp_dim = get_warp_dim(warp_size, smem_tile_dim, reg_tile_dim)
            inner_axis = _fuse_inner_axis(full_st_dim, warp_dim)
            aligned_inner_axis = _align(inner_axis, transaction_size)
            # print(aligned_inner_axis / inner_axis)
            memory_penalty *= aligned_inner_axis / inner_axis

        memory_latency_tensor = rw_bytes * memory_penalty / bandwidth * (1000000000) / (1024 * 1024 * 1024)
        memory_latency_tile += memory_latency_tensor
        
    return memory_latency_tile * num_tiles
