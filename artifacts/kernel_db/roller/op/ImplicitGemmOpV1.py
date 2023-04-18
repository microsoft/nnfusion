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


class ImplicitGemmOpV1(OpBase):
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
        self.dims["input1"] = [C * K * K, N * self._HI * self._WI] # 2d data
        self.dims["input2"] = [F, C * K * K] # 2d kernel
        #self.dims["input1"] = [N, C, H + T + D, W + L + R] # 4d data
        #self.dims["input2"] = [F, C, K, K] # 4d kernel
        self.dims["output"] = [F, N * self._HO * self._WO]

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
            ox, oy = tile_dim
            aligned_K = math.ceil(self.reduction_axis_len() / reduction_size["k"] + 1) * reduction_size["k"]
            return ox * oy * aligned_K * 2

    def subtensor_dim(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        # given a tile size, returns the memory footprints of related subtensors
        ret = {}
        if tile_tensor == "output":
            ox, oy = tile_dim
            ox = min(ox, self._F)
            oy = min(oy, self._N * self._HO * self._WO)
            k = reduction_size["k"]
            if mem_level == 0:
                # conv shape
                ret["input1"] = [k, min(oy + self._K * self._S + 1, self.dims["input1"][1])]
                ret["input2"] = [ox, k]
                ret["output"] = [ox, oy]
            if mem_level == 1:
                # matmul shape
                ret["input1"] = [k, oy]
                ret["input2"] = [ox, k]
                ret["output"] = [ox, oy]
        return ret

    def reg_usage(self, tile_dim, tile_tensor="output"):
        # given a register tile size, returns the number of register used
        # for now tiling is only on output
        ret = {}
        if tile_tensor == "output":
            ox, oy = tile_dim
            ret["input1"] = oy
            ret["input2"] = ox
            ret["output"] = ox * oy
        return ret, ret["input1"] + ret["input2"] + ret["output"]
    
    def memory_workload(self, tile_dim, reduction_size, mem_level, tile_tensor="output"):
        # given a tile size and a memory level, returns the amount of bytes loaded/stored
        ret = {}
        if tile_tensor == "output":
            ox, oy = tile_dim
            if mem_level == 0:
                ret["input1"] = self._C * self._K * self._K * oy * 4
                ret["input2"] = ox * self._C * self._K * self._K * 4
                ret["output"] = ox * oy * 4
            elif mem_level == 1:
                aligned_K = ((self._K * self._K * self._C - 1) // reduction_size["k"] + 1) * reduction_size["k"]
                ret["input1"] = aligned_K * oy * 4
                ret["input2"] = ox * aligned_K * 4
                ret["output"] = ox * oy * 4
        return ret

    def get_grid_size(self, tile_dim, tile_tensor="output"):
        # Given a tile size, return the grid size of this op
        if tile_tensor == "output":
            M, N = self.dims["output"]
            m, n = tile_dim
            grid_m = int((M + (m - 1)) // m)
            grid_n = int((N + (n - 1)) // n)
            return grid_m * grid_n
