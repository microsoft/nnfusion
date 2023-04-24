from .OpBase import *

class Transpose2dOp(OpBase):
    def __init__(self, M, N):
        # input[M, N] -> output[N, M]
        self._M = M
        self._N = N
        self.dims = {}
        self.dims["input"] = [M, N]
        self.dims["output"] = [N, M]
        self.use_tc = False
        self.input_type = "float"
        self.output_type = "float"

    def compute_workload(self, tile_dim):
        return 0

    def memory_workload(self, tile_dim, tile_tensor, mem_level):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        m, n = tile_dim[0]
        ret["input"] = m * n * 4
        ret["output"] = m * n * 4
        return ret

    def reg_usage(self, tile_dim, tile_tensor):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        m, n = tile_dim[0], tile_dim[1]
        ret["input"] = m * n
        ret["output"] = m * n
        return ret, ret["input"] + ret["output"]
    
    def subtensor_size(self, tile_dim, tile_tensor = "output"):
        ret = {}
        if tile_tensor == "output":
            n, m = tile_dim
            ret["input"] = {}
            ret["input"]["dim"] = [m, n]
            ret["input"]["axis"] = ["M", "N"]
            ret["output"] = {}
            ret["output"]["dim"] = [n, m]
            ret["output"]["axis"] = ["N", "M"]
        if tile_tensor == "input":
            m, n = tile_dim
            ret["input"] = {}
            ret["input"]["dim"] = [m, n]
            ret["input"]["axis"] = ["M", "N"]
            ret["output"] = {}
            ret["output"]["dim"] = [n, m]
            ret["output"]["axis"] = ["N", "M"]
        return ret

    def constant_ratio(self):
        return True

