from .OpBase import *
import math

def Prod(Dims):
    ret = 1
    for dim in Dims:
        ret *= dim
    return ret

class ReduceOp(OpBase):
    """
    Tensor[M, N] -> Tensor[M]
    """
    def __init__(self, M, N):
        # Reduced: the number of consecutive inner axes fused
        self.dims = {}
        self.dims["input"] = [M, N]
        self.dims["output"] = [M]

        self._M = M
        self._N = N
        self.use_tc = False
        self.input_type = "float"
        self.output_type = "float"

    def compute_workload(self, tile_dim, tile_tensor):
        if isinstance(tile_dim, int):
            m = tile_dim
        else:
            m = tile_dim[0]
        return m * self._N * 2

    def reduction_axis_len(self):
        return self._N

    def reg_usage(self, tile_dim, tile_tensor):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        if isinstance(tile_dim, int):
            m = tile_dim
        else:
            m = tile_dim[0]
        ret = {}
        ret["input"] = m
        ret["output"] = m
        return ret, ret["input"] + ret["output"]
    
    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        if isinstance(tile_dim, int):
            m = tile_dim
        else:
            m = tile_dim[0]
        k = reduction_size["k"]
        N = math.ceil(self._N / k) * k
        ret = {}
        ret["input"] = m * N * 4
        ret["output"] = m * 4
        return ret

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor):
        if isinstance(tile_dim, int):
            m = tile_dim
        else:
            m = tile_dim[0]
        k = reduction_size["k"]
        ret = {}
        if tile_tensor == "output":
            ret["input"] = [m, k]
            ret["output"] = [m]
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if isinstance(tile_dim, int):
            m = tile_dim
        else:
            m = tile_dim[0]
        if tile_tensor == "output":
            return math.ceil(self._M / m)


"""
class ReduceOp(OpBase):
    def __init__(self, M, N):
        # Reduced: the number of consecutive inner axes fused
        self.dims = {}
        self.dims["input"] = [Dims]
        self.dims["output"] = [Dims[i] for i in range(len(Dims) - Reduced)]

        self.reduce_keys = ["k{}".format(k) for k in range(Reduced)]

        self.thread_workload = Prod(self.dims["input"]) // Prod(self.dims["output"])
        self.constant_ratio = True

    def compute_workload(self, tile_dim, tile_tensor):
        return self.thread_workload * Prod(tile_dim)

    def reg_usage(self, tile_dim, tile_tensor):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        ret["input"] = Prod(tile_dim)
        ret["output"] = Prod(tile_dim)
        return ret, ret["input"] + ret["output"]
    
    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        N = Prod(tile_dim)
        ret["input"] = N * thread_workload * 4
        ret["output"] = N * 4
        return ret

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor):
        ret = {}
        if tile_tensor == "output":
            ret["input"] = tile_dim.copy()
            for rk in self.reduce_keys:
                ret["input"].append(reduction_size[rk])
            ret["output"] = tile_dim
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if tile_tensor == "output":
            ret = 1
            for dout, dtile in zip(self.dims["output"], tile_dim):
                ret *= math.ceil(dout / dtile)
            return ret
"""

