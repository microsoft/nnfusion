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


class Pooling2dOp(OpBase):
    def __init__(self, N, C, K, S, H, W, D, P):
        self._B = N * C # fuse N & C axis
        self._K = K
        self._H = H
        self._W = W
        
        self._S = S
        self._P = P

        pt, pl, pd, pr = get_pad_tuple(P, (K, K))
        self._HI = H + pt + pd
        self._WI = W + pl + pr
        self._HO = (self._HI - ((K - 1) * D + 1)) // S + 1
        self._WO = (self._WI - ((K - 1) * D + 1)) // S + 1

        self.dims = {}
        self.dims["input1"] = [self._B, self._HI, self._WI] # data 4d
        self.dims["output"] = [self._B, self._HO, self._WO]

        self.use_tc = False
        self.input_type = "float"
        self.output_type = "float"

    def reduction_axis_len(self):
        return self._K * self._K

    def compute_workload(self, tile_dim, tile_tensor):
        # given a tile size, returns the number of FLOPS involved in this tile
        # for now tiling is only on output
        if tile_tensor == "output":
            b, ho, wo = tile_dim
            return b * ho * wo * self._K * self._K * 2

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor):
        # given a tile size, returns the memory footprints of related subtensors
        ret = {}
        if tile_tensor == "output":
            b, ho, wo = tile_dim
            h_load = self._K + self._S * (ho - 1)
            w_load = self._K + self._S * (wo - 1)
            ret["input1"] = [b, h_load, w_load]
            ret["output"] = [b, ho, wo]
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if tile_tensor == "output":
            b, ho, wo = tile_dim
            grid_b = math.ceil(self._B / b)
            grid_h = math.ceil(self._HO / ho)
            grid_w = math.ceil(self._WO / wo)
            return grid_b * grid_h * grid_w
    
    def compute_workload_total(self):
        return self.compute_workload(self.dims["output"], "output")

    def reg_usage(self, tile_dim, tile_tensor):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        if tile_tensor == "output":
            b, ho, wo = tile_dim
            ret["input1"] = b * ho * wo * self._K * self._K
            ret["output"] = b * ho * wo
        return ret, ret["input1"] + ret["output"]
    
    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        if tile_tensor == "output":
            b, ho, wo = tile_dim
            # the amount of bytes loaded from DRAM to SMEM
            h_load = self._K + self._S * (ho - 1)
            w_load = self._K + self._S * (wo - 1)
            ret["input1"] = b * h_load * w_load * 4
            ret["output"] = b * ho * wo * 4
        return ret
