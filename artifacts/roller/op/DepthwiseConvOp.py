from .OpBase import *
import math

def get_pad_tuple(padding, kernel):
    """Common code to get the pad option
    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    kernel : tuple of int
        Conv kernel size
    Returns
    -------
    pad_top : int
        Padding size on top
    pad_left : int
        Padding size on left
    pad_down : int
        Padding size on down.
    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


class DepthwiseConvOp(OpBase):
    def __init__(self, N, C, K, S, H, W, D, P, M = 1):
        self._N = N
        self._C = C
        self._K = K
        self._H = H
        self._W = W
        self._M = M
        
        self._S = S
        self._P = P

        pt, pl, pd, pr = get_pad_tuple(P, (K, K))
        self._HI = H + pt + pd
        self._WI = W + pl + pr
        self._HO = (self._HI - ((K - 1) * D + 1)) // S + 1
        self._WO = (self._WI - ((K - 1) * D + 1)) // S + 1

        self.dims = {}
        self.dims["input1"] = [N, C, H, W] # data 4d
        self.dims["input2"] = [C, M, K, K] # kernel 4d, M dimension is set as 1
        self.dims["output"] = [N, C, self._HO, self._WO]

        self.use_tc = False
        self.input_type = "float"
        self.output_type = "float"

    def reduction_axis_len(self):
        return self._K * self._K

    def compute_workload(self, tile_dim, reduction_size, tile_tensor = "output"):
        # given a tile size, returns the number of FLOPS involved in this tile
        # for now tiling is only on output
        if tile_tensor == "output":
            n, c, ho, wo = tile_dim
            aligned_K = math.ceil(self.reduction_axis_len() / reduction_size["k"] + 1) * reduction_size["k"]
            return n * c * ho * wo * aligned_K * 2

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        # given a tile size, returns the memory footprints of related subtensors
        ret = {}
        if tile_tensor == "output":
            n, c, ho, wo = tile_dim
            h_load = self._K + self._S * (ho - 1)
            w_load = self._K + self._S * (wo - 1)
            ret["input1"] = [n, c, h_load, w_load]
            ret["input2"] = [c, 1, self._K, self._K]
            ret["output"] = [n, c, ho, wo]
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if tile_tensor == "output":
            n, c, ho, wo = tile_dim
            grid_n = math.ceil(self._N / n)
            grid_c = math.ceil(self._C / c)
            grid_h = math.ceil(self._HO / ho)
            grid_w = math.ceil(self._WO / wo)
            return grid_n * grid_c * grid_h * grid_w
    
    def compute_workload_total(self):
        return self.compute_workload(self.dims["output"], "output")

    def reg_usage(self, tile_dim, tile_tensor = "output"):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        if tile_tensor == "output":
            n, c, ho, wo = tile_dim
            ret["input1"] = n * ho * wo
            ret["input2"] = c
            ret["output"] = n * c * ho * wo
        return ret, ret["input1"] + ret["input2"] + ret["output"]
    
    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor = "output"):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        if tile_tensor == "output":
            n, c, ho, wo = tile_dim
            # the amount of bytes loaded from DRAM to SMEM
            # estimate the overhead caused by non-divisible dimensions
            overhead = 1
            for i in range(len(self.dims["output"])):
                N = math.ceil(self.dims["output"][i] / tile_dim[i])
                overhead *= N * tile_dim[i] / self.dims["output"][i]
            h_load = self._K + self._S * (ho - 1)
            w_load = self._K + self._S * (wo - 1)
            ret["input1"] = n * h_load * w_load * c * 4 * overhead
            ret["input2"] = self._K * self._K * c * 4 * overhead
            ret["output"] = n * c * ho * wo * 4 * overhead
        return ret
