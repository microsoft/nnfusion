from .OpBase import *
import math

class BatchMatmulOp(OpBase):
    def __init__(self, B, M, K, N):
        # 3D Batch NT (M, K) x (K, N) matmul
        self._B = B
        self._M = M
        self._K = K
        self._N = N

        self.dims = {}
        self.dims["input1"] = [B, M, K]
        self.dims["input2"] = [B, K, N]
        self.dims["output"] = [B, M, N]

        self.axis = {}
        self.axis["B"] = B
        self.axis["M"] = M
        self.axis["k"] = K
        self.axis["N"] = N

        self.use_tc = False
        self.input_type = "float"
        self.output_type = "float"

    def reduction_axis_len(self):
        return self._K

    def compute_workload(self, tile_dim, reduction_size, tile_tensor = "output"):
        # given a tile size, returns the number of FLOPS involved in this tile
        # for now tiling is only on output
        if tile_tensor == "output":
            b, m, n = tile_dim
            aligned_K = math.ceil(self.reduction_axis_len() / reduction_size["k"]) * reduction_size["k"]
            return b * m * aligned_K * n * size_of(self.input_type) // 2

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        ret = {}
        if tile_tensor == "output":
            b, m, n = tile_dim
            ret["input1"] = [b, m, reduction_size["k"]]
            ret["input2"] = [b, reduction_size["k"], n]
            ret["output"] = [b, m, n]
        return ret

    def reg_usage(self, tile_dim, tile_tensor = "output"):
        # given a register tiling size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        if tile_tensor == "output":
            b, m, n = tile_dim
            ret["input1"] = b * m
            ret["input2"] = b * n
            ret["output"] = b * m * n
        return ret, ret["input1"] + ret["input2"] + ret["output"]

    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        if tile_tensor == "output":
            b, m, n = tile_dim
            aligned_K = math.ceil(self.reduction_axis_len() / reduction_size["k"]) * reduction_size["k"]
            ret["input1"] = b * m * aligned_K * size_of(self.input_type)
            ret["input2"] = b * n * aligned_K * size_of(self.input_type)
            ret["output"] = b * m * n * size_of(self.output_type)
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if tile_tensor == "output":
            b, m, n = tile_dim
            grid_b = int((self._B + (b - 1)) // b)
            grid_m = int((self._M + (m - 1)) // m)
            grid_n = int((self._N + (n - 1)) // n)
            return grid_b * grid_m * grid_n
    
