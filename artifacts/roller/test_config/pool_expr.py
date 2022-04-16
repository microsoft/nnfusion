from xml import sax
import tvm
from tvm import te
from utils import *
import math

def avgpool2d_expr_S2P0(shape, dataType='float32', for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    B, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [B, HI, WI])], [("avgpool2d", [B, HO, WO])]
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, HI, WI), dtype=dataType, name="data")
    padded_data = te.compute((B, HI + 2 * P, WI + 2 * P),lambda b, ho, wo: 
                                te.if_then_else(
                                te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, ho, wo], 0.0), 
    name="padded_data")

    avgpool2d = te.compute((B, HO, WO), lambda b, ho, wo: 
                te.sum(padded_data[b, ho * S + kh * D, wo * S + kw * D] / (KH * KW),
            axis=[kh, kw],
        ),
        name="avgpool2d")

    return [data], [avgpool2d]

def avgpool2d_expr_S2P01(shape, dataType='float32', for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    B, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P - 1
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P - 1

    if for_rtile:
        return [("data", [B, HI, WI])], [("avgpool2d", [B, HO, WO])]
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, HI, WI), dtype=dataType, name="data")
    padded_data = te.compute((B, HI + 2 * P, WI + 2 * P),lambda b, ho, wo: 
                                te.if_then_else(
                                te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, ho, wo], 0.0), 
    name="padded_data")

    avgpool2d = te.compute((B, HO, WO), lambda b, ho, wo: 
                te.sum(padded_data[b, ho * S + kh * D, wo * S + kw * D] / (KH * KW),
            axis=[kh, kw],
        ),
        name="avgpool2d")

    return [data], [avgpool2d]

def avgpool2d_expr_S2P02(shape, dataType='float32', for_rtile=False, pad={}):
    S, D, P = 2, 1, 0
    B, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1

    if for_rtile:
        return [("data", [B, HI, WI])], [("avgpool2d", [B, HO, WO])]
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, HI, WI), dtype=dataType, name="data")
    padded_data = te.compute((B, HI + 2 * P, WI + 2 * P),lambda b, ho, wo: 
                                te.if_then_else(
                                te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, ho, wo], 0.0), 
    name="padded_data")

    avgpool2d = te.compute((B, HO, WO), lambda b, ho, wo: 
                te.sum(padded_data[b, ho * S + kh * D, wo * S + kw * D] / (KH * KW),
            axis=[kh, kw],
        ),
        name="avgpool2d")

    return [data], [avgpool2d]


def avgpool2d_expr_S1P0(shape, dataType='float32', for_rtile=False, pad={}):
    S, D, P = 1, 1, 0
    B, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [B, HI, WI])], [("avgpool2d", [B, HO, WO])]
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, HI, WI), dtype=dataType, name="data")
    padded_data = te.compute((B, HI + 2 * P, WI + 2 * P),lambda b, ho, wo: 
                                te.if_then_else(
                                te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, ho, wo], 0.0), 
    name="padded_data")

    avgpool2d = te.compute((B, HO, WO), lambda b, ho, wo: 
                te.sum(padded_data[b, ho * S + kh * D, wo * S + kw * D] / (KH * KW),
            axis=[kh, kw],
        ),
        name="avgpool2d")

    return [data], [avgpool2d]

def avgpool2d_expr_S2P1(shape, dataType='float32', for_rtile=False, pad={}):
    S, D, P = 2, 1, 1
    B, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [B, HI, WI])], [("avgpool2d", [B, HO, WO])]
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, HI, WI), dtype=dataType, name="data")
    padded_data = te.compute((B, HI + 2 * P, WI + 2 * P),lambda b, ho, wo: 
                                te.if_then_else(
                                te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, ho, wo], 0.0), 
    name="padded_data")

    avgpool2d = te.compute((B, HO, WO), lambda b, ho, wo: 
                te.sum(padded_data[b, ho * S + kh * D, wo * S + kw * D] / (KH * KW),
            axis=[kh, kw],
        ),
        name="avgpool2d")

    return [data], [avgpool2d]

def avgpool2d_expr_S1P1(shape, dataType='float32', for_rtile=False, pad={}):
    S, D, P = 1, 1, 1
    B, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [B, HI, WI])], [("avgpool2d", [B, HO, WO])]
    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, HI, WI), dtype=dataType, name="data")
    padded_data = te.compute((B, HI + 2 * P, WI + 2 * P),lambda b, ho, wo: 
                                te.if_then_else(
                                te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, ho, wo], 0.0), 
    name="padded_data")

    avgpool2d = te.compute((B, HO, WO), lambda b, ho, wo: 
                te.sum(padded_data[b, ho * S + kh * D, wo * S + kw * D] / (KH * KW),
            axis=[kh, kw],
        ),
        name="avgpool2d")

    return [data], [avgpool2d]

def maxpool2d_expr_S2P1(shape, dataType='float32', for_rtile=False, pad={}):
    S, D, P = 2, 1, 1
    B, HO, WO, KH, KW = shape
  
    HI = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    WI = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P

    if for_rtile:
        return [("data", [B, HI, WI])], [("avgpool2d", [B, HO, WO])]

    kh = te.reduce_axis((0, KH), name="kh")
    kw = te.reduce_axis((0, KW), name="kw")
    
    data = te.placeholder((B, HI + 1, WI + 1), dtype=dataType, name="data")
    padded_data = te.compute((B, HI + 2 * P, WI + 2 * P),lambda b, ho, wo: 
                                te.if_then_else(
                                te.all(ho < P + HI, wo < P + WI), data[b, ho, wo], -3.402823e37), 
                                #te.all(P <= ho, ho < P + HI, P <= wo, wo < P + WI), data[b, ho, wo], 0.0), 
    name="padded_data")

    maxpool2d = te.compute((B, HO, WO), lambda b, ho, wo: 
                te.max(padded_data[b, ho * S + kh * D, wo * S + kw * D],
            axis=[kh, kw],
        ),
        name="maxpool2d")

    return [data], [maxpool2d]
