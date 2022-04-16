from xml import sax
import tvm
from tvm import te
from utils import *


def add_expr(shape, dataType='float32', for_rtile=False, pad={}):
    if for_rtile:
        return [("A", shape), ("B", shape)], [("C", shape)]
    A = te.placeholder(shape, dtype=dataType, name="A")
    B = te.placeholder(shape, dtype=dataType, name="B")
    C = te.compute(shape, lambda *i: A(*i) + B(*i), name="C")
    return [A, B], [C]

def relu_expr(shape, dataType='float32', for_rtile=False, pad={}):
    if for_rtile:
        return [("A", shape)], [("C", shape)]
    A = te.placeholder(shape, dtype=dataType, name="A")
    C = te.compute(shape, lambda *i: te.max(A(*i), tvm.tir.const(0, A.dtype)), name="C")
    return [A], [C]

def relu_fusable_expr(shape, dataType='float32', for_rtile=False, pad={}):
    D1, D2, D3 = shape
    if for_rtile:
        return [("A", shape)], [("C", shape)]
    A = te.placeholder((D1, D2, D3), dtype=dataType, name="A")
    C = te.compute((D1, D2, D3), lambda d1, d2, d3: te.max(A[d1, d2, d3], tvm.tir.const(0, A.dtype)), name="C")
    return [A], [C]