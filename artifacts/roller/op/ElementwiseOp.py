from .OpBase import *

def Area(tile):
    ret = 1
    for d in tile:
        ret *= d
    return ret

class UnitaryOp(OpBase):
    def __init__(self, dim, fuse=True):
        # dim: the input dimensions, in list
        if fuse:
            dim = [Area(dim)]
        self._dim = dim
        self._dim_size = len(dim)
        self.dims = {}
        self.dims["input"] = dim
        self.dims["output"] = dim

        self.use_tc = False
        self.input_type = "float"
        self.output_type = "float"
        # self.axis_name = []
        # for i in range(self._dim_size):
        #     self.axis_name.append("N" + str(i + 1))

    def compute_workload(self, tile_dim, reduction_size, tile_tensor="output"):
        return Area(tile_dim)

    def reg_usage(self, tile_dim, tile_tensor="output"):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        size = Area(tile_dim)
        ret["input"] = size
        ret["output"] = size
        return ret, ret["input"] + ret["output"]
    
    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        size = Area(tile_dim)
        ret["input"] = size * size_of(self.input_type)
        ret["output"] = size * size_of(self.input_type)
        return ret

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        ret = {}
        if tile_tensor == "output" or tile_tensor == "input":
            ret["input"] = tile_dim
            ret["output"] = tile_dim
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        ret = 1
        for d in range(self._dim_size):
            ret *= int((self._dim[d] + (tile_dim[d] - 1)) // tile_dim[d])
        return ret


class BinaryOp(OpBase):
    def __init__(self, dim, fuse=True):
        # dim: the input dimensions, in list
        if fuse:
            dim = [Area(dim)]
        self._dim = dim
        self._dim_size = len(dim)
        self.dims = {}
        self.dims["input1"] = dim
        self.dims["input2"] = dim
        self.dims["output"] = dim

    def compute_workload(self, tile_dim, reduction_size, tile_tensor="output"):
        return Area(tile_dim)

    def reg_usage(self, tile_dim, tile_tensor="output"):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        size = Area(tile_dim)
        ret["input1"] = size
        ret["input2"] = size
        ret["output"] = size
        return ret, ret["input1"] + ret["input2"] + ret["output"]
    
    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        size = Area(tile_dim)
        ret["input1"] = size * 4
        ret["input2"] = size * 4
        ret["output"] = size * 4
        return ret

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        ret = {}
        if tile_tensor == "output" or tile_tensor == "input":
            ret["input1"] = tile_dim
            ret["input2"] = tile_dim
            ret["output"] = tile_dim
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        ret = 1
        for d in range(self._dim_size):
            ret *= int((self._dim[d] + (tile_dim[d] - 1)) // tile_dim[d])
        return ret


class BiasAddOp(OpBase):
    def __init__(self, M, N):
        # dim: the input dimensions, in list
        self._M = M
        self._N = N

        self.dims = {}
        self.dims["input1"] = [M, N]
        self.dims["input2"] = [N]
        self.dims["output"] = [M, N]

        # self.axis = {}
        # self.axis["M"] = M
        # self.axis["N"] = N

        self.constant_ratio = True

    def compute_workload(self, tile_dim, reduction_size, tile_tensor="output"):
        return Area(tile_dim)

    def reg_usage(self, tile_dim, tile_tensor="output"):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        m, n = tile_dim[0], tile_dim[1]
        ret["input1"] = m * n
        ret["input2"] = n
        ret["output"] = m * n
        return ret, ret["input1"] + ret["input2"] + ret["output"]
    
    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        m, n = tile_dim[0], tile_dim[1]
        ret["input1"] = m * n * 4
        ret["input2"] = n * 4
        ret["output"] = m * n * 4
        return ret

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        m, n = tile_dim[0], tile_dim[1]
        ret = {}
        ret["input1"] = tile_dim
        ret["input2"] = [n]
        ret["output"] = tile_dim
        return ret

    def constant_ratio(self):
        return True
    
    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        m, n = tile_dim[0], tile_dim[1]
        grid_m = int((self._M + (m - 1)) // m)
        grid_n = int((self._N + (n - 1)) // n)
        return grid_m * grid_n

