from xml import sax
import tvm
from tvm import te
from tvm.topi import nn
from utils import *

# python test_op.py --op depthwiseconv_expr --shape 128 96 83 83 7 7 --rtile2_shape 1 1 1 1 1 1  --rtile1_shape 1 1 1 2 1 1 --rtile0_shape 2 1 4 96 7 7 --smem_tiling --reg_tiling --use_artificial_rtile

def depthwiseconv_expr_S2D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [N, C, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [N, C, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [N, C, HO, WO]), ("depthwiseconv2d_unpad", [N, C, HO, WO])]
    data = te.placeholder((N, C, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((N, C, H + 2 * P, W + 2 * P),
                lambda n, c, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[n, c, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: te.sum(
            (data_pad[n, c, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: depthwiseconv2d[n, c, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_expr_S2D1P2(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 2
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [N, C, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [N, C, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [N, C, HO, WO]), ("depthwiseconv2d_unpad", [N, C, HO, WO])]
    data = te.placeholder((N, C, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((N, C, H + 2 * P, W + 2 * P),
                lambda n, c, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[n, c, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: te.sum(
            (data_pad[n, c, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: depthwiseconv2d[n, c, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_expr_S1D1P2(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 2
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [N, C, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [N, C, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [N, C, HO, WO]), ("depthwiseconv2d_unpad", [N, C, HO, WO])]
    data = te.placeholder((N, C, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((N, C, H + 2 * P, W + 2 * P),
                lambda n, c, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[n, c, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: te.sum(
            (data_pad[n, c, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: depthwiseconv2d[n, c, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_expr_S2D1P3(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 3
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [N, C, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [N, C, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [N, C, HO, WO]), ("depthwiseconv2d_unpad", [N, C, HO, WO])]
    data = te.placeholder((N, C, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((N, C, H + 2 * P, W + 2 * P),
                lambda n, c, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[n, c, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: te.sum(
            (data_pad[n, c, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: depthwiseconv2d[n, c, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_expr_S1D1P1(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 1
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [N, C, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [N, C, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [N, C, HO, WO]), ("depthwiseconv2d_unpad", [N, C, HO, WO])]
    data = te.placeholder((N, C, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((N, C, H + 2 * P, W + 2 * P),
                lambda n, c, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[n, c, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: te.sum(
            (data_pad[n, c, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: depthwiseconv2d[n, c, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_expr_S1D1P3(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 3
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [N, C, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [N, C, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [N, C, HO, WO]), ("depthwiseconv2d_unpad", [N, C, HO, WO])]
    data = te.placeholder((N, C, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((N, C, H + 2 * P, W + 2 * P),
                lambda n, c, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[n, c, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    depthwiseconv2d = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: te.sum(
            (data_pad[n, c, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (N, C, HO, WO),
        lambda n, c, ho, wo: depthwiseconv2d[n, c, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]



# CNHW exprs

def depthwiseconv_cnhw_expr_S2D1P0(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [C, N, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [C, N, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [C, N, HO, WO]), ("depthwiseconv2d_unpad", [C, N, HO, WO])]
    data = te.placeholder((C, N, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((C, N, H + 2 * P, W + 2 * P),
                lambda c, n, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[c, n, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: te.sum(
            (data_pad[c, n, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: depthwiseconv2d[c, n, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_cnhw_expr_S2D1P2(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 2
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [C, N, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [C, N, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [C, N, HO, WO]), ("depthwiseconv2d_unpad", [C, N, HO, WO])]
    data = te.placeholder((C, N, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((C, N, H + 2 * P, W + 2 * P),
                lambda c, n, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[c, n, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: te.sum(
            (data_pad[c, n, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: depthwiseconv2d[c, n, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_cnhw_expr_S1D1P2(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 2
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [C, N, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [C, N, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [C, N, HO, WO]), ("depthwiseconv2d_unpad", [C, N, HO, WO])]
    data = te.placeholder((C, N, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((C, N, H + 2 * P, W + 2 * P),
                lambda c, n, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[c, n, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: te.sum(
            (data_pad[c, n, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: depthwiseconv2d[c, n, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_cnhw_expr_S2D1P3(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 2, 1, 3
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [C, N, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [C, N, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [C, N, HO, WO]), ("depthwiseconv2d_unpad", [C, N, HO, WO])]
    data = te.placeholder((C, N, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((C, N, H + 2 * P, W + 2 * P),
                lambda c, n, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[c, n, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: te.sum(
            (data_pad[c, n, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: depthwiseconv2d[c, n, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_cnhw_expr_S1D1P1(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 1
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [C, N, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [C, N, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [C, N, HO, WO]), ("depthwiseconv2d_unpad", [C, N, HO, WO])]
    data = te.placeholder((C, N, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((C, N, H + 2 * P, W + 2 * P),
                lambda c, n, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[c, n, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    depthwiseconv2d = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: te.sum(
            (data_pad[c, n, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: depthwiseconv2d[c, n, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]

def depthwiseconv_cnhw_expr_S1D1P3(shape, dataType="float32", for_rtile=False, pad={}):
    S, D, P = 1, 1, 3
    N, C, HO, WO, KH, KW = shape
    H = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    W = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [C, N, H, W]), ("kernel", [C, KH, KW]), ("data_pad", [C, N, H + 2 * P, W + 2 * P]), ("kernel_pad", [C, KH, KW])],\
            [("depthwiseconv2d", [C, N, HO, WO]), ("depthwiseconv2d_unpad", [C, N, HO, WO])]
    data = te.placeholder((C, N, H, W), dtype=dataType, name="data")
    kernel = te.placeholder((C, KH, KW), dtype=dataType, name="kernel")

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    data_pad = te.compute((C, N, H + 2 * P, W + 2 * P),
                lambda c, n, ho, wo: te.if_then_else(te.all(P <= ho, ho < H + P, P <= wo, wo < W + P),
                data[c, n, ho, wo], 0.0),
                tag="data_pad", name="data_pad")

    kernel_pad = te.compute((C, KH, KW),
        lambda c, kh, kw: kernel[c, kh, kw], tag="kernel_pad", name="kernel_pad"
    )

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")

    depthwiseconv2d = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: te.sum(
            (data_pad[c, n, ho * S + kh * D, wo * S + kw * D] * kernel_pad[c, kh, kw]),
            axis=[kh, kw],
        ),
        name="depthwiseconv2d")

    depthwiseconv2d_unpad = te.compute(
        (C, N, HO, WO),
        lambda c, n, ho, wo: depthwiseconv2d[c, n, ho, wo], tag="depthwiseconv2d_unpad", name="depthwiseconv2d_unpad")

    return [data, kernel, data_pad, kernel_pad], [depthwiseconv2d, depthwiseconv2d_unpad]