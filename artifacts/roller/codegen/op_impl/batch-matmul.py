from os import close
import tvm
from tvm import te
import numpy as np
import sys
from codegen import CodeGenerator
from tvm.contrib import nvcc
from tvm.topi.utils import get_const_tuple
from tvm.topi.testing import batch_matmul

@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", arch="compute_70")
    return ptx

def split_axis(factors, sch, op, axis):
    ret = []
    for i in range(0, len(factors)):
        ax0, ax1 = sch[op].split(axis, factor=int(np.prod(factors[i:])))
        ret.append(ax0)
        axis = ax1
    return ret + [axis]

def batch_matmul_v2(x, y):
    assert len(x.shape) == 4 and len(y.shape) == 4, "only support 4-dim batch_matmul"
    x_shape = get_const_tuple(x.shape)
    y_shape = get_const_tuple(y.shape)
    XB = x_shape[0]
    YB = y_shape[0]
    XC = x_shape[1]
    YC = y_shape[1]
    
    _, _, M, K = x.shape
    N = y.shape[2]
    k = te.reduce_axis((0, K), name="k")

    assert XB == YB or XB == 1 or YB == 1, "batch dimension doesn't match"
    assert x_shape[3] == y_shape[3], "shapes of x and y is inconsistant"
    assert XC == YC, "channel doesn't match"

    batch = max(XB, YB)
    channel = XC
    return te.compute(
        (batch, channel, M, N),
        lambda b, c, n, m: te.sum(x[b if XB != 1 else 0, c, n, k] * y[b if YB != 1 else 0, c, m, k], axis=k),
        tag="batch_matmul_v2",
    )

### params
XB = 128
YB = 1
C = 16
M = 512
K = 64
N = 512

tile_b = [1, 1]
tile_c = [1, 1]
tile_x = [16, 8]
tile_y = [16, 8]
tile_k = [32]

if len(sys.argv) == 7:
    XB = int(sys.argv[1])
    YB = int(sys.argv[2])
    C  = int(sys.argv[3])
    M  = int(sys.argv[4])
    K  = int(sys.argv[5])
    N  = int(sys.argv[6])

A = te.placeholder((XB, C, M, K), name="x")
B = te.placeholder((YB, C, N, K), name="y")
out = batch_matmul_v2(A, B)
s = te.create_schedule([out.op])

generator = CodeGenerator()
tile_dict = {'k': [32], 'b': [1, 1], 'c': [2, 1], 'n': [16, 4], 'm': [16, 4]}
generator.rewrite_schedule(s, tile_dict, True, True)

# AA = s.cache_read(A, "shared", [out])
# AL = s.cache_read(AA, "local", [out])
# BB = s.cache_read(B, "shared", [out])
# BL = s.cache_read(BB, "local", [out])
# CC = s.cache_write(out, "local")

# b, c, y, x = s[out].op.axis

# bb, tb, bi = split_axis(tile_b, s, out, b)
# bc, tc, ci = split_axis(tile_c, s, out, c)
# by, ty, yi = split_axis(tile_y, s, out, y)
# bx, tx, xi = split_axis(tile_x, s, out, x)

# s[out].reorder(bb, bc, by, bx, tb, tc, ty, tx, bi, ci, yi, xi)

# fused_bx = s[out].fuse(bb, bc, by, bx)
# fused_tx = s[out].fuse(tb, tc, ty, tx)

# s[out].bind(fused_bx, te.thread_axis("blockIdx.x"))
# s[out].bind(fused_tx, te.thread_axis("threadIdx.x"))

# s[CC].compute_at(s[out], fused_tx)
# bi, ci, yi, xi = s[CC].op.axis
# k = s[CC].op.reduce_axis[0]
# ko, ki = split_axis(tile_k, s, CC, k)
# s[CC].reorder(ko, ki, bi, ci, yi, xi)

# def optimize_read_cache(shared, local):
#     s[shared].compute_at(s[CC], ko)
#     s[local].compute_at(s[CC], ki)
#     b, c, y, x = s[shared].op.axis
#     fused = s[shared].fuse(b, c, y, x)
#     oo, ii = s[shared].split(fused, factor=tile_b[0]*tile_c[0]*tile_y[0]*tile_x[0])
#     s[shared].reorder(oo, ii)
#     s[shared].bind(ii, te.thread_axis("threadIdx.x"))

# optimize_read_cache(AA, AL)
# optimize_read_cache(BB, BL)

func = tvm.build(s, [A, B, out], "cuda")

with open('batch-matmul.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

def batch_matmul_numpy(x, y):
    XB, channel, M, _ = x.shape
    YB, channel, N, _ = y.shape
    batch = max(XB, YB)
    out = np.zeros((batch, channel, M, N)).astype(x.dtype)
    for i in range(batch):
        for j in range(channel):
            out[i][j] = np.dot(x[i if XB != 1 else 0][j], y[i if YB != 1 else 0][j].T)
    return out

ctx = tvm.gpu(0)
dtype = A.dtype
a_np = np.random.uniform(size=(XB, C, M, K)).astype(dtype)
b_np = np.random.uniform(size=(YB, C, N, K)).astype(dtype)
c_np = batch_matmul_numpy(a_np, b_np)

a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
c = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=dtype), ctx)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print("BatchMatMul: %f ms" % (evaluator(a, b, c).mean * 1e3))