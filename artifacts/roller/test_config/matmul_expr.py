from xml import sax
import tvm
from tvm import te
from utils import *
import os

CNHW = (os.getenv('CONV_LAYOUT')=="CNHW")

def matmul_expr(shape, dataType="float32", for_rtile=False, pad={}):
    M, N, K = shape
    if for_rtile:
        return [("A", [K, M] if CNHW else [M, K]), ("B", [K, N])], [("compute", [M, N])]
    A = te.placeholder((K, M) if CNHW else (M, K), dtype=dataType, name="A")
    B = te.placeholder((K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda y, x: te.sum((A[k, y] if CNHW else A[y, k]) * B[k, x], axis=k), name='compute')
    return [A, B], [C]

def batch_matmul_expr(shape, dataType="float32", for_rtile=False, pad={}):
    BC, M, N, K = shape
    if for_rtile:
        return [("A", [BC, M, K]), ("B", [BC, K, N])], [("compute", [BC, M, N])]
    A = te.placeholder((BC, M, K), dtype=dataType, name="A")
    B = te.placeholder((BC, K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((BC, M, N), lambda b, x, y: te.sum(A[b, x, k] * B[b, k, y], axis=k))
    return [A, B], [C]
