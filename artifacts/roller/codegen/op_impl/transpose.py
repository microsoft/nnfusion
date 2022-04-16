from __future__ import absolute_import, print_function

import sys
import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm.contrib import nvcc
from codegen import CodeGenerator

@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", arch="compute_70")
    return ptx

def split_axis(sche, op, axis, factors):
    ret = []
    for i in range(0, len(factors)):
        ax0, ax1 = sche[op].split(axis, factor=int(np.prod(factors[i:])))
        ret.append(ax0)
        axis = ax1
    return ret + [axis]

# Describe Sum of Rows
n = 1024
m = 1024
for i in range(len(sys.argv)):
    if sys.argv[i] == '-n':
        n = int(sys.argv[i+1])
    if sys.argv[i] == '-m':
        m = int(sys.argv[i+1])
print("N: {} M: {}".format(n, m))
A = te.placeholder((n, m), name="A")
C = te.compute((m, n), lambda x, y: A[y, x])

s = te.create_schedule(C.op)
generator = CodeGenerator()
tile_dict = {'x': [4, 8], 'y': [8, 4]}
generator.rewrite_schedule(s, tile_dict, False, False)

# # A_shared = s.cache_read(A, "shared", [C])
# y, x = s[C].op.axis
# by, vy, ty, ny = split_axis(s, C, y, [8, 4, 1])
# bx, vx, tx, nx = split_axis(s, C, x, [4, 8, 1])
# s[C].reorder(by, bx, ty, tx, ny, nx)

# s[C].bind(by, te.thread_axis("blockIdx.y"))
# s[C].bind(bx, te.thread_axis("blockIdx.x"))
# s[C].bind(vy, te.thread_axis("vthread"))
# s[C].bind(vx, te.thread_axis("vthread"))
# s[C].bind(ty, te.thread_axis("threadIdx.y"))
# s[C].bind(tx, te.thread_axis("threadIdx.x"))
# s[C].unroll(ny)
# s[C].unroll(nx)

# # fused_bx = s[C].fuse(by, bx)
# # fused_tx = s[C].fuse(ty, tx)
# # fused_nx = s[C].fuse(ny, nx)
# # s[C].bind(fused_bx, te.thread_axis("blockIdx.x"))
# # s[C].bind(fused_tx, te.thread_axis("threadIdx.x"))
# # s[C].unroll(fused_nx)

# # s[A_shared].compute_at(s[C], fused_tx)
# # ys, xs = s[A_shared].op.axis
# # bys, tys = s[A_shared].split(ys, factor=32)
# # bxs, txs = s[A_shared].split(xs, factor=32)
# # s[A_shared].reorder(bys, bxs, tys, txs)
# # fused_bxs = s[A_shared].fuse(bys, bxs)
# # fused_txs = s[A_shared].fuse(tys, txs)
# # s[A_shared].bind(fused_txs, te.thread_axis("threadIdx.x"))

func = tvm.build(s, [A, C], "cuda")
with open('transpose.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

# Verify the correctness of result kernel by comparing it to numpy.
ctx = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), ctx)
c = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
func(a, c)
evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
print("Transpose: %f ms" % (evaluator(a, c).mean * 1e3))
tvm.testing.assert_allclose(c.asnumpy(), np.transpose(a.asnumpy()), rtol=1e-1)


# generate kernel entry
# tvm_func_name = "tuned_fused_sum_op_float_i{0}_{1}_o{1}_{0}_kernel0".format(
#     n, m,
# )

# op_type = "Transpose"
# parameters = {
#     "input_shape": [n, m], 
#     "output_shape": [m, n]
# }

# gridDim = [v.value if not isinstance(v, int) else v for v in generator.blck_grid]
# blockDim = [v.value if not isinstance(v, int) else v for v in generator.thrd_grid]
# code = func.imported_modules[0].get_source()

# kernel = {
#     'tvm_func_name': tvm_func_name,
#     'op_type': op_type, 
#     'parameters': parameters, 
#     'code': code, 
#     'gridDim': gridDim,
#     'blockDim': blockDim
# }

# from kernel_db.convert_external import insert_kernel_entry
# insert_kernel_entry(kernel)