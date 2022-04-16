import tvm
from tvm import te
import os

FUSE_PAD = (os.getenv('FUSE_PAD') == "1")
CNHW = (os.getenv('CONV_LAYOUT')=="CNHW")
#python test_op.py --op conv_expr --shape 128 96 83 83 1 7 7 --rtile2_shape 1 1 1 1 1 1 1  --rtile1_shape 1 1 1 2 1 1 1 --rtile0_shape 2 1 4 96 7 7 1 --smem_tiling --reg_tiling --use_artificial_rtile
  
def conv_expr_S1D1P0(shape, dataType='float32', for_rtile=False, pad={}):
    S, D, P = 1, 1, 0
    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data",[N, C, H, W]), ("kernel", [F, C, KH, KW])], [("conv", [N, F, HO, WO])]

    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    c = te.reduce_axis((0, C), name='c')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')
    if CNHW:
        conv = te.compute((F, N, HO, WO), lambda n, f, ho, wo:\
                te.sum(data[c, n, ho * S + kh * D, wo * S + kw * D] *
                        kernel[f, c, kh, kw],
                        axis=[c, kh, kw])
                        , name='conv')
    else:
        conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
                te.sum(data[n, c, ho * S + kh * D, wo * S + kw * D] *
                        kernel[f, c, kh, kw],
                        axis=[c, kh, kw])
                        , name='conv')

    return [data, kernel], [conv]

def conv_expr_S1D1P1(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 1
    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data",[N, C, H, W]), ("kernel", [F, C, KH, KW])], [("conv", [N, F, HO, WO])]
   
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    c = te.reduce_axis((0, C), name='c')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    if CNHW:
        conv = te.compute((F, N, HO, WO), lambda n, f, ho, wo:\
                te.sum(data[c, n, ho * S + kh * D, wo * S + kw * D] *
                        kernel[f, c, kh, kw],
                        axis=[c, kh, kw])
                        , name='conv')
    else:
        conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
                te.sum(data[n, c, ho * S + kh * D, wo * S + kw * D] *
                        kernel[f, c, kh, kw],
                        axis=[c, kh, kw])
                        , name='conv')

    return [data, kernel], [conv]

def conv_expr_S2D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    if for_rtile:
        return [("data",[N, C, H, W]), ("kernel", [F, C, KH, KW])], [("conv", [N, F, HO, WO])]

    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    c = te.reduce_axis((0, C), name='c')
    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    if CNHW:
        conv = te.compute((F, N, HO, WO), lambda n, f, ho, wo:\
                te.sum(data[c, n, ho * S + kh * D, wo * S + kw * D] *
                        kernel[f, c, kh, kw],
                        axis=[c, kh, kw])
                        , name='conv')
    else:
        conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
                te.sum(data[n, c, ho * S + kh * D, wo * S + kw * D] *
                        kernel[f, c, kh, kw],
                        axis=[c, kh, kw])
                        , name='conv')

    return [data, kernel], [conv]

def fused_conv_expr_S1D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")


    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: conv[f, sa_fused0], tag="conv_unpad", name="conv_unpad")

    return [data, kernel, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_expr_S1D1P1(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 1
    OFFSET = P if FUSE_PAD else 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']

    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(
               te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW) if not FUSE_PAD else 
               te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW,
                    sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D > 0, 
                    sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D < H + 1, 
                    sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D > 0,
                    sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D < W + 1),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D - OFFSET,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D - OFFSET], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")


    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: conv[f, sa_fused0], tag="conv_unpad", name="conv_unpad")

    return [data, kernel, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_expr_S2D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]
    
    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape

    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")


    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: conv[f, sa_fused0], tag="conv_unpad", name="conv_unpad")

    return [data, kernel, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_bias_expr_S1D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")

    bias = te.placeholder([F], name="bias")

    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: conv[f, sa_fused0] + bias[f], tag="conv_unpad", name="conv_unpad")

    return [data, kernel, bias, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_relu_expr_S1D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")

    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: te.max(conv[f, sa_fused0], tvm.tir.const(0, conv.dtype)), tag="conv_unpad", name="conv_unpad")

    return [data, kernel, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_bias_relu_expr_S1D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")

    bias = te.placeholder([F], name="bias")

    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: te.max(conv[f, sa_fused0] + bias[f], tvm.tir.const(0, conv.dtype)), tag="conv_unpad", name="conv_unpad")

    return [data, kernel, bias, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_bias_expr_S1D1P1(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 1
    OFFSET = P if FUSE_PAD else 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(
               te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW) if not FUSE_PAD else 
               te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW,
                    sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D > 0, 
                    sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D < H + 1, 
                    sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D > 0,
                    sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D < W + 1),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D - OFFSET,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D - OFFSET], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")

    bias = te.placeholder([F], name="bias")

    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: conv[f, sa_fused0] + bias[f], tag="conv_unpad", name="conv_unpad")

    return [data, kernel, bias, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_relu_expr_S1D1P1(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 1
    OFFSET = P if FUSE_PAD else 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P
    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(
               te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW) if not FUSE_PAD else 
               te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW,
                    sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D > 0, 
                    sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D < H + 1, 
                    sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D > 0,
                    sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D < W + 1),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D - OFFSET,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D - OFFSET], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")

    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: te.max(conv[f, sa_fused0], tvm.tir.const(0, conv.dtype)), tag="conv_unpad", name="conv_unpad")

    return [data, kernel, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_bias_relu_expr_S1D1P1(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 1
    OFFSET = P if FUSE_PAD else 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(
               te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW) if not FUSE_PAD else 
               te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW,
                    sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D > 0, 
                    sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D < H + 1, 
                    sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D > 0,
                    sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D < W + 1),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D - OFFSET,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D - OFFSET], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")

    bias = te.placeholder([F], name="bias")

    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: te.max(conv[f, sa_fused0] + bias[f], tvm.tir.const(0, conv.dtype)), tag="conv_unpad", name="conv_unpad")

    return [data, kernel, bias, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_bias_expr_S2D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")

    bias = te.placeholder([F], name="bias")

    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: conv[f, sa_fused0] + bias[f], tag="conv_unpad", name="conv_unpad")

    return [data, kernel, bias, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_relu_expr_S2D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")

    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: te.max(conv[f, sa_fused0], tvm.tir.const(0, conv.dtype)), tag="conv_unpad", name="conv_unpad")

    return [data, kernel, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

def fused_conv_bias_relu_expr_S2D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    if for_rtile:
        F, SA_FUSED0, RA_FUSED0 = shape
        return [("data",[]), ("kernel", []), ("data_pad", [RA_FUSED0, SA_FUSED0]), ("kernel_pad", [F, RA_FUSED0])], \
            [("conv", [F, SA_FUSED0]), ("conv_unpad", [F, SA_FUSED0])]

    if len(shape) != 7:
        N = te.var('N')
        F = te.var('F')
        HO = te.var('HO')
        WO = te.var('WO')
        C = te.var('C')
        KH = te.var('KH')
        KW = te.var('KW')
    else:
        N, F, HO, WO, C, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    sa_fused0_pad = N * HO * WO
    if 'sa_fused0' in pad: sa_fused0_pad += pad['sa_fused0']
    ra_fused0_pad = C * KH * KW
    if 'ra_fused0' in pad: ra_fused0_pad += pad['ra_fused0']
    f_pad = F
    if 'f' in pad: f_pad += pad['f']
    
    data_shape = (C, N) if CNHW else (N,C)
    data_shape += ((H, W) if FUSE_PAD else (H + 2 * P, W + 2 * P))
    data = te.placeholder(data_shape, dtype=dataType, name="data")
    kernel = te.placeholder((F, C, KH, KW), dtype=dataType, name="kernel")

    ra_fused0 = te.reduce_axis((0, C * KH * KW), name="ra_fused0")

    data_pad = te.compute([ra_fused0_pad, sa_fused0_pad],
           lambda ra_fused0, sa_fused0: te.if_then_else(te.all(sa_fused0 < N * HO * WO, ra_fused0 < C * KH * KW),
           data[ra_fused0 // (KH * KW) if CNHW else sa_fused0 // (HO * WO),
                sa_fused0 // (HO * WO) if CNHW else ra_fused0 // (KH * KW),
                sa_fused0 % (HO * WO) // WO * S + ra_fused0 % (KH * KW) // (KW) * D,
                sa_fused0 % (HO * WO) % WO * S + ra_fused0 % (KH * KW) % (KW) * D], 0.0),
           tag="data_pad", name="data_pad")

    kernel_pad = te.compute([f_pad, ra_fused0_pad],
           lambda f, ra_fused0: te.if_then_else(te.all(f < F, ra_fused0 < C * KH * KW),
           kernel[f,
                ra_fused0 // (KH * KW),
                ra_fused0 % (KH * KW) // (KW),
                ra_fused0 % (KH * KW) % (KW)], 0.0),
           tag="kernel_pad", name="kernel_pad")

    bias = te.placeholder([F], name="bias")

    conv = te.compute([f_pad, sa_fused0_pad], lambda f, sa_fused0:\
                te.sum(data_pad[ra_fused0, sa_fused0] *
                        kernel_pad[f, ra_fused0],
                        axis=[ra_fused0])
                        , name='conv')
    conv_unpad = te.compute([F, N * HO * WO], lambda f, sa_fused0: te.max(conv[f, sa_fused0] + bias[f], tvm.tir.const(0, conv.dtype)), tag="conv_unpad", name="conv_unpad")

    return [data, kernel, bias, data_pad, kernel_pad], [conv, conv_unpad], {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}

