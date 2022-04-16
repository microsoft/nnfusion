from config import *

def Area(tile):
    ret = 1
    for d in tile:
        ret *= d
    return ret

"""
def Addresses_To_Delay(flattened_warp_addr, bank_size):
    # returns the highest visted bank
    counts = [0 for _ in range(bank_size)]
    for flattened_addr in flattened_warp_addr:
        counts[flattened_addr % bank_size] += 1
    max_count = 0
    for count in counts:
        max_count = max(max_count, count)
    different_banks = 0
    for i in range(bank_size):
        if counts[i] > 0:
            different_banks += 1
    return min(different_banks, max_count)
"""

def Access_To_Delay(conflict):
    # when doubling the number of threads accessing the different locations in the same bank
    # this is empirical results of v100
    return conflict
    #return (conflict * 2 + 16) / 16


def Bank_Count(flattened_warp_addr, bank_size):
    # returns the highest visted bank
    #print(flattened_warp_addr)
    counts = [0 for _ in range(bank_size)]
    deduplicated_addr = []
    for flattened_addr in flattened_warp_addr:
        if flattened_addr not in deduplicated_addr:
            deduplicated_addr.append(flattened_addr)
    #print(flattened_warp_addr)
    #print(deduplicated_addr)
    for addr in deduplicated_addr:
        counts[addr % bank_size] += 1
    ret = 0
    for count in counts:
        ret = max(ret, count)
    return ret


def Address_Iterates(full_tile, base_tile, total = -1):
    # filling full_tile with base_tile
    # generate a list of addresses of tile2 corners
    dim_size = len(base_tile)
    addrs = [[0 for _ in range(dim_size)]]
    while (total == -1) or (len(addrs) < total):
        last_addr = addrs[-1]
        d = dim_size - 1
        new_addr = last_addr.copy()
        while d >= 0:
            if last_addr[d] + 1 < full_tile[d] // base_tile[d]:
                new_addr[d] += 1
                break
            new_addr[d] = 0
            d -= 1
        if d < 0:
            break
        addrs.append(new_addr)
    return addrs


def Bank_Conflict_Penalty(schedule, tiling):
    # For each tensors, returns how much slower memory bandwidth is when loading data, due to bank conflict
    # For now we only calculate the bank conflict in SMEM level
    #print("====================================")
    reg_tile = schedule.get_tile(1)
    smem_tile = schedule.get_tile(0)
    # returns if there is no smem_tile
    if smem_tile == None:
        ret = {}
        for tensor_name in tiling.op.dims:
            ret[tensor_name] = 1
        return ret
    # create a sub_op with smem tile being the full size
    sub_op = tiling.op.sub_op(smem_tile, tiling.tile_tensor)
    # generate all the first loaded addresses of all threads in one smem tile
    dim_size = len(sub_op.dims[tiling.tile_tensor])
    warp_size = 32
    addrs = Address_Iterates(smem_tile, reg_tile, warp_size)
    
    # For each tensor, estimate bank conflict by calculating the addresses of first loaded element in a warp
    ret = {}
    bank_size = 32
    for tensor_name in sub_op.dims:
        if tensor_name == tiling.tile_tensor:
            continue
        flatten_addrs = []
        for addr in addrs:
            deps = sub_op.dep_base(addr, tiling.tile_tensor)
            flatten_addrs.append(sub_op.flatten_addr(deps[tensor_name], tensor_name))
            #print(tensor_name, addr, deps[tensor_name], sub_op.flatten_addr(deps[tensor_name], tensor_name))
        ret[tensor_name] = Access_To_Delay(Bank_Count(flatten_addrs, bank_size))
    return ret

    # compute bank conflicts for each warp on each tensor
    for tensor_name in sub_op.dims:
        if tensor_name == tiling.tile_tensor:
            continue
        ret[tensor_name] = 0
        warp_count = 0
        tid = 0
        for tid in range(0, len(addrs[tensor_name]), warp_size):
            tid_r = min(tid + warp_size, len(addrs[tensor_name]))
            flattened_warp_addrs = [sub_op.flatten_addr(addr, tensor_name) for addr in addrs[tensor_name][tid: tid_r]]
            ret[tensor_name] += Access_To_Delay(Bank_Count(flattened_warp_addrs, bank_size))
        #ret[tensor_name] += Addresses_To_Delay(flattened_warp_addrs, bank_size)
        warp_count += 1
        ret[tensor_name] = ret[tensor_name] / warp_count
    #print("------------------------------------")

    return ret
