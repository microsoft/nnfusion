from .OpBase import *

class TransposeMdOp(OpBase):
    def __init__(self, shape, perm):
        self.input_shape = shape
        self.in_to_out_perm = perm
        self.out_to_in_perm = [0 for _ in perm]
        for i in range(len(perm)):
            self.out_to_in_perm[perm[i]] = i
        self.nd = len(shape)

        self.dims = {}
        self.dims["input"] = list(self.input_shape)
        self.dims["output"] = [self.input_shape[p] for p in perm]

        self.use_tc = False
        self.input_type = "float"
        self.output_type = "float"

    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        size = 1
        for d in tile_dim:
            size *= d
        ret["input"] = size * 4
        ret["output"] = size * 4
        return ret

    def reg_usage(self, tile_dim, tile_tensor):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        size = 1
        for d in tile_dim:
            size *= d
        ret["input"] = size
        ret["output"] = size
        return ret, ret["input"] + ret["output"]

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        ret = {}
        if tile_tensor == "output":
            ret["output"] = list(tile_dim)
            ret["input"] = [tile_dim[p] for p in self.in_to_out_perm]
        return ret
    
    def subtensor_size(self, tile_dim, tile_tensor = "output"):
        ret = {}
        ret["input"] = {}
        ret["input"]["dim"] = list(tile_dim)
        ret["input"]["axis"] = ["N" + str(i) for i in range(self.nd)]
        ret["output"] = {}
        ret["output"]["dim"] = []
        ret["output"]["axis"] = []
        for p in self.perm:
            ret["output"]["dim"].append(tile_dim[p])
            ret["output"]["axis"].append("N" + str(p))
       
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if tile_tensor == "output":
            total = 1
            for i in range(self.nd):
                grid_i = int((self.input_shape[i] + (tile_dim[i] - 1)) // tile_dim[i])
                total *= grid_i
            return total
