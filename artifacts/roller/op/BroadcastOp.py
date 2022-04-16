from .OpBase import *
import math

def Prod(dim):
    ret = 1
    for d in dim:
        ret *= d
    return ret

class BroadcastOp(OpBase):
    def __init__(self, M, B, N):
        # input[M, N] -> output[M, B, N]
        self.dims = {}
        self.dims["input"] = [M, N]
        self.dims["output"] = [M, B, N]

        self._M = M
        self._B = B
        self._N = N
        
        self.use_tc = False
        self.input_type = "float"
        self.output_type = "float"

    def compute_workload(self, tile_dim):
        return 0

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        ret = {}
        if tile_tensor == "output":
            m, b, n = tile_dim
            ret["input"] = [m, n]
            ret["output"] = [m, b, n]
        return ret

    def reg_usage(self, tile_dim, tile_tensor = "output"):
        # given a register tiling size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        if tile_tensor == "output":
            m, b, n = tile_dim
            ret["input"] = m * n
            ret["output"] = m * b * n
        return ret, ret["input"] + ret["output"]

    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        if tile_tensor == "output":
            m, b, n = tile_dim
            ret["input"] = m * n * size_of(self.input_type)
            ret["output"] = m * b * n * size_of(self.output_type)
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if tile_tensor == "output":
            m, b, n = tile_dim
            grid_m = int((self._M + (m - 1)) // m)
            grid_b = int((self._B + (b - 1)) // b)
            grid_n = int((self._N + (n - 1)) // n)
            return grid_m * grid_b * grid_n
    