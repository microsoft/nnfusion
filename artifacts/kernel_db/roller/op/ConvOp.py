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


class ConvOp(OpBase):
    def __init__(self, N, C, F, K, S, H, W, D, P):
        self._N = N
        self._C = C
        self._F = F
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
        self.dims["input1"] = [N, C, self._HI, self._WI] # 2d data
        self.dims["input2"] = [F, C, K, K] # 2d kernel
        #self.dims["input1"] = [N, C, H + T + D, W + L + R] # 4d data
        #self.dims["input2"] = [F, C, K, K] # 4d kernel
        self.dims["output"] = [N, F, self._HO, self._WO]

        self.axis = {}
        self.axis["n"] = N
        self.axis["c"] = C
        self.axis["f"] = F
        self.axis["s"] = S
        self.axis["h"] = H
        self.axis["w"] = W
        self.axis["k"] = K

        self.use_tc = False
        self.input_type = "float"
        self.output_type = "float"

    def reduction_axis_len(self):
        return self._C * self._K * self._K

    def compute_workload(self, tile_dim, reduction_size, tile_tensor="output"):
        # given a tile size, returns the number of FLOPS involved in this tile
        # for now tiling is only on output
        if tile_tensor == "output":
            n, f, ho, wo = tile_dim
            aligned_K = math.ceil(self.reduction_axis_len() / reduction_size["k"]) * reduction_size["k"]
            return n * f * ho * wo * aligned_K * 2

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        # given a tile size, returns the memory footprints of related subtensors
        # for now support only k=1
        ret = {}
        if tile_tensor == "output":
            n, f, ho, wo = tile_dim
            c = reduction_size["k"]
            hi, wi = ho, wo

            ret["input1"] = [n, c, hi, wi]
            ret["input2"] = [f, c, self._K, self._K]
            ret["output"] = [n, f, ho, wo]
        return ret

    def reg_usage(self, tile_dim, tile_tensor="output"):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        if tile_tensor == "output":
            n, f, ho, wo = tile_dim
            ret["input1"] = n * ho * wo
            ret["input2"] = f
            ret["output"] = n * f * ho * wo
        return ret, ret["input1"] + ret["input2"] + ret["output"]
    
    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        if tile_tensor == "output":
            n, f, ho, wo = tile_dim
            aligned_K = math.ceil(self.reduction_axis_len() / reduction_size["k"]) * reduction_size["k"]
            ret["input1"] = aligned_K * n * ho * wo * size_of(self.input_type)
            ret["input2"] = f * aligned_K * size_of(self.input_type)
            ret["output"] = n * f * ho * wo * size_of(self.input_type)
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if tile_tensor == "output":
            N, F, HO, WO = self.dims["output"]
            n, f, ho, wo = tile_dim
            grid_n = math.ceil(N / n)
            grid_f = math.ceil(F / f)
            grid_ho = math.ceil(HO / ho)
            grid_wo = math.ceil(WO / wo)
            return grid_n * grid_f * grid_ho * grid_wo
