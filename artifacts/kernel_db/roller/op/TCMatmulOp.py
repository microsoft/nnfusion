from .OpBase import *
import math

class TCMatmulOp(OpBase):
    def __init__(self, M, K, N, wmma_m, wmma_k, wmma_n, input_type, output_type):
        self._M = M
        self._K = K
        self._N = N

        self._wmma_m = wmma_m
        self._wmma_k = wmma_k
        self._wmma_n = wmma_n

        self.dims = {}
        self.dims["input1"] = [M, K]
        self.dims["input2"] = [K, N]
        self.dims["output"] = [M, N]

        self.axis = {}
        self.axis["M"] = M
        self.axis["k"] = K
        self.axis["N"] = N

        self.input_type = input_type
        self.output_type = output_type
        self.use_tc = True

    def reduction_axis_len(self):
        return self._K

    def compute_workload(self, tile_dim, reduction_size, tile_tensor = "output"):
        # given a tile size, returns the number of FLOPS involved in this tile
        # for now tiling is only on output
        if tile_tensor == "output":
            m, n = tile_dim
            aligned_K = math.ceil(self.reduction_axis_len() / reduction_size["k"]) * reduction_size["k"]
            return m * aligned_K * n * size_of(self.input_type) // 2

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        ret = {}
        if tile_tensor == "output":
            m, n = tile_dim
            ret["input1"] = [m, reduction_size["k"]]
            ret["input2"] = [reduction_size["k"], n]
            ret["output"] = [m, n]
        return ret

    def reg_usage(self, warp_tile_dim, tile_tensor = "output"):
        # given the dimension of a warp tile, return the register usage
        ret = {}
        if tile_tensor == "output":
            m, n = warp_tile_dim
            ret["input1"] = m // self._wmma_m * 4
            ret["input2"] = n // self._wmma_n * 4
            ret["output"] = m * n // self._wmma_m // self._wmma_n * 4
        return ret, ret["input1"] + ret["input2"] + ret["output"]

    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        if tile_tensor == "output":
            m, n = tile_dim
            aligned_K = math.ceil(self.reduction_axis_len() / reduction_size["k"]) * reduction_size["k"]
            ret["input1"] = m * aligned_K * size_of(self.input_type)
            ret["input2"] = aligned_K * n * size_of(self.input_type)
            ret["output"] = m * n * size_of(self.output_type)
        return ret

    def flatten_addr(self, addr, tile_tensor):
        # Given an address in full dimension, return the flattened 1D address
        if tile_tensor == "input1":
            m, k = addr[0], addr[1]
            return m * self._K + k
        if tile_tensor == "input2":
            k, n = addr[0], addr[1]
            return k * self._N + n
        if tile_tensor == "output":
            # output tensor
            pass

    def uni_schedule(self, saxis_names, raxis_names):
        tile = [self._wmma_m, self._wmma_n]
        rstep = {raxis_names[0]: self._wmma_k}
        return tile, rstep

    def get_block_size(self, block_tile, warp_tile):
        block_x, block_y, block_z = 32, block_tile[0] // warp_tile[0], block_tile[1] // warp_tile[1]
        return block_x, block_y, block_z
    
    def get_grid_size(self, block_tile):
        m = math.ceil(self._M / block_tile[0])
        n = math.ceil(self._N / block_tile[1])
        return m, n
