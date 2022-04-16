from __future__ import absolute_import, print_function

import sys
import tvm
import tvm.testing
from tvm import te
import numpy as np
from codegen import CodeGenerator
from tvm import topi
from tvm.contrib import nvcc
import numpy as np

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

# Describe Sum of Rows
N = 1
C = 256
H = 56
W = 56
for i in range(len(sys.argv)):
    if sys.argv[i] == '-n':
        N = int(sys.argv[i+1])
    if sys.argv[i] == '-c':
        C = int(sys.argv[i+1])
    if sys.argv[i] == '-h':
        H = int(sys.argv[i+1])
    if sys.argv[i] == '-w':
        W = int(sys.argv[i+1])
print("N: {} C: {} H: {} W: {}".format(N, C, H, W))

X = te.placeholder((N, C, H, W), name='X')
Mean = te.placeholder((C,), name='Mean')
Var = te.placeholder((C,), name='Var')
Scale = te.placeholder((C,), name='Scale')
Offset = te.placeholder((C,), name='Offset')
Y = te.compute((N, C, H, W), lambda n, c, h, w: (X[n, c, h, w] - Mean[c]) / te.sqrt(Var[c] + 1e-5) * Scale[c] + Offset[c])

s = te.create_schedule(Y.op)

tile_dict = {"n":[1, 1, 1], "c":[1, 2, 2], "h":[1, 2, 2], "w":[1, 2, 2]}
generator = CodeGenerator()
generator.rewrite_schedule(s, tile_dict, False, False)

# n, c, h, w = Y.op.axis
# bn, vn, tn, ni = split_axis(tile_dict["n"], s, Y, n)
# bc, vc, tc, ci = split_axis(tile_dict["c"], s, Y, c)
# bh, vh, th, hi = split_axis(tile_dict["h"], s, Y, h)
# bw, vw, tw, wi = split_axis(tile_dict["w"], s, Y, w)

# s[Y].reorder(bn, bc, bh, bw, vn, vc, vh, vw, tn, tc, th, tw, ni, ci, hi, wi)
# fused_blck = s[Y].fuse(bn, bc, bh, bw)
# fused_thrd = s[Y].fuse(tn, tc, th, tw)
# fused_reg  = s[Y].fuse(ni, ci, hi, wi)

# s[Y].bind(fused_blck, te.thread_axis("blockIdx.x"))
# s[Y].bind(vn, te.thread_axis("vthread"))
# s[Y].bind(vc, te.thread_axis("vthread"))
# s[Y].bind(vh, te.thread_axis("vthread"))
# s[Y].bind(vw, te.thread_axis("vthread"))
# s[Y].bind(fused_thrd, te.thread_axis("threadIdx.x"))
# s[Y].unroll(fused_reg)

func = tvm.build(s, [X, Mean, Var, Scale, Offset, Y], "cuda")
with open('fused-batchnorm.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

# Verify the correctness of result kernel by comparing it to numpy.
ctx = tvm.gpu(0)
data = tvm.nd.array(np.random.normal(size=(N, C, H, W)).astype(X.dtype), ctx)
mean = tvm.nd.array(np.random.normal(size=(C,)).astype(Mean.dtype), ctx)
var = tvm.nd.array(np.absolute(np.random.normal(loc=1.0, size=(C,)).astype(Var.dtype)), ctx)
scale = tvm.nd.array(np.random.normal(size=(C,)).astype(Scale.dtype), ctx)
offset = tvm.nd.array(np.random.normal(size=(C,)).astype(Offset.dtype), ctx)
out = tvm.nd.array(np.zeros((N, C, H, W), dtype = Y.dtype), ctx)

func(data, mean, var, scale, offset, out)
evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
print("fused-batch-norm: %f ms" % (evaluator(data, mean, var, scale, offset, out).mean * 1e3))
res = (data.asnumpy() - mean.asnumpy().reshape(C, 1, 1)) / np.sqrt(var.asnumpy() + 1e-5).reshape(C, 1, 1) * scale.asnumpy().reshape(C, 1, 1) + offset.asnumpy().reshape(C, 1, 1)
tvm.testing.assert_allclose(out.asnumpy(), res, rtol=1e-4)