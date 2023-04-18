from xml import sax
import tvm
from tvm import te
# from roller.utils import *
import os

CNHW = (os.getenv('CONV_LAYOUT')=="CNHW")

def matmul_expr_00(shape, dataType="float32", for_rtile=False, pad={}):
    M, N, K = shape
    if for_rtile:
        return [("A", [M, K]), ("B", [K, N])], [("compute", [M, N])]
    A = te.placeholder((M, K), dtype=dataType, name="A")
    B = te.placeholder((K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda y, x: te.sum(A[y, k] * B[k, x], axis=k), name='compute')
    return [A, B], [C]

def matmul_expr_01(shape, dataType="float32", for_rtile=False, pad={}):
    M, N, K = shape
    if for_rtile:
        return [("A", [M, K]), ("B", [N, K])], [("compute", [M, N])]
    A = te.placeholder((M, K), dtype=dataType, name="A")
    B = te.placeholder((N, K), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda y, x: te.sum(A[y, k] * B[x, k], axis=k), name='compute')
    return [A, B], [C]

def matmul_expr_10(shape, dataType="float32", for_rtile=False, pad={}):
    M, N, K = shape
    if for_rtile:
        return [("A", [K, M]), ("B", [K, N])], [("compute", [M, N])]
    A = te.placeholder((K, M), dtype=dataType, name="A")
    B = te.placeholder((K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda y, x: te.sum(A[k, y] * B[k, x], axis=k), name='compute')
    return [A, B], [C]

def matmul_expr_11(shape, dataType="float32", for_rtile=False, pad={}):
    M, N, K = shape
    if for_rtile:
        return [("A", [K, M]), ("B", [N, K])], [("compute", [M, N])]
    A = te.placeholder((K, M), dtype=dataType, name="A")
    B = te.placeholder((N, K), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda y, x: te.sum(A[k, y] * B[x, k], axis=k), name='compute')
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


def batch_matmul_bcast_a_expr(shape, dataType="float32", for_rtile=False, pad={}):
    BC, M, N, K = shape
    if for_rtile:
        return [("A", [M, K]), ("B", [BC, K, N])], [("compute", [BC, M, N])]
    A = te.placeholder((M, K), dtype=dataType, name="A")
    B = te.placeholder((BC, K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((BC, M, N), lambda b, x, y: te.sum(A[x, k] * B[b, k, y], axis=k))
    return [A, B], [C]


def batch_matmul_4d_3d_expr(shape, dataType="float32", for_rtile=False, pad={}):
    B1, B2, M, N, K = shape
    if for_rtile:
        return [("A", [B1, B2, M, K]), ("B", [B2, K, N])], [("compute", [B1, B2, M, N])]
    A = te.placeholder((B1, B2, M, K), dtype=dataType, name="A")
    B = te.placeholder((B2, K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((B1, B2, M, N), lambda b1, b2, x, y: te.sum(A[b1, b2, x, k] * B[b2, k, y], axis=k))
    return [A, B], [C]


def batch_matmul_4d_4d_expr(shape, dataType="float32", for_rtile=False, pad={}):
    B1, B2, M, N, K = shape
    if for_rtile:
        return [("A", [B1, B2, M, K]), ("B", [B1, B2, K, N])], [("compute", [B1, B2, M, N])]
    A = te.placeholder((B1, B2, M, K), dtype=dataType, name="A")
    B = te.placeholder((B1, B2, K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((B1, B2, M, N), lambda b1, b2, x, y: te.sum(A[b1, b2, x, k] * B[b1, b2, k, y], axis=k))
    return [A, B], [C]