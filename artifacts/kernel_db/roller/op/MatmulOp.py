from .OpBase import *
import math

class MatmulOp(OpBase):
    def __init__(self, M, K, N):
        self._M = M
        self._K = K
        self._N = N

        self.dims = {}
        self.dims["input1"] = [M, K]
        self.dims["input2"] = [K, N]
        self.dims["output"] = [M, N]

        self.axis = {}
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
            m, n = tile_dim
            aligned_K = math.ceil(self.reduction_axis_len() / reduction_size["k"]) * reduction_size["k"]
            return m * aligned_K * n * size_of(self.input_type) // 2

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor = "output", padding = 0):
        ret = {}
        if tile_tensor == "output":
            m, n = tile_dim
            ret["input1"] = [m, reduction_size["k"] + padding]
            ret["input2"] = [reduction_size["k"], n]
            ret["output"] = [m, n]
        return ret

    def reg_usage(self, tile_dim, tile_tensor = "output"):
        # given a register tiling size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        if tile_tensor == "output":
            m, n = tile_dim
            ret["input1"] = m
            ret["input2"] = n
            ret["output"] = m * n
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

    def sub_op(self, sub_dim, tile_tensor = "output"):
        # return a sub op with the size of tile_tensor being sub_dim
        if tile_tensor == "output":
            K = self._K
            M, N = sub_dim[0], sub_dim[1]
            return MatmulOp(M, K, N)

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

    def unflatten_addr(self, addr, tile_tensor = "output"):
        # Given an flattened 1D address, return the address in full dimension
        ret = []
        dim = self.dims[tile_tensor].copy()
        dim.reverse()
        for d in dim:
            ret.append(addr % d)
            addr = addr // d
        ret.reverse()
        return ret

    def dep_base(self, addr, tile_tensor = "output"):
        # Given an address in full dimension, return the address of the first loaded element
        if tile_tensor == "output":
            ret = {}
            m, n = addr[0], addr[1]
            ret["input1"] = [m, 0]
            ret["input2"] = [0, n]
            return ret
    
    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if tile_tensor == "output":
            m, n = tile_dim
            grid_m = int((self._M + (m - 1)) // m)
            grid_n = int((self._N + (n - 1)) // n)
            return grid_m * grid_n
    
    
