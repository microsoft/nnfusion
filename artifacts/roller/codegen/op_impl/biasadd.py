from os import close
import tvm
from tvm import te
import numpy as np
import sys
from codegen import CodeGenerator
from tvm.contrib import nvcc

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

### params
M = 512
N = 1024

if len(sys.argv) == 3:
    M = int(sys.argv[1])
    N = int(sys.argv[2])

print(M, N)

### Algorithm
A = te.placeholder((M, N), name="A")
B = te.placeholder((N,), name="B")
C = te.compute((M, N), lambda y, x: A[y, x] + B[x])
s = te.create_schedule(C.op)

generator = CodeGenerator()
tile_dict = {'y': [2, 2], 'x': [2, 2]}
generator.rewrite_schedule(s, tile_dict, False, False)
func = tvm.build(s, [A, B, C], "cuda")

# ## Caches
# A_shared = s.cache_read(A, "shared", [C])
# A_local = s.cache_read(A_shared, "local", [C])
# B_shared = s.cache_read(B, "shared", [C])
# B_local = s.cache_read(B_shared, "local", [C])
# C_local = s.cache_write(C, "local")

# # Tiling
# tile_y = [4, 16, 1]
# tile_x = [4, 12, 1]
# m, n = s[C].op.axis
# by, vy, ty, tm = split_axis(s, C, m, tile_y)
# bx, vx, tx, tn = split_axis(s, C, n, tile_x)

# s[C].reorder(by, bx, vy, vx, ty, tx, tm, tn)
# blck_fused = s[C].fuse(by, bx)
# # vthd_fused = s[C].fuse(vy, vx)
# thrd_fused = s[C].fuse(ty, tx)
# s[C].bind(blck_fused, te.thread_axis("blockIdx.x"))
# # s[C].bind(vthd_fused, te.thread_axis("vthread"))
# s[C].bind(thrd_fused, te.thread_axis("threadIdx.x"))

# # s[C].bind(bx, te.thread_axis("blockIdx.x"))
# # s[C].bind(by, te.thread_axis("blockIdx.y"))
# s[C].bind(vy, te.thread_axis("vthread"))
# s[C].bind(vx, te.thread_axis("vthread"))
# # vthd_fused = s[C].fuse(vy, vx)
# # s[C].bind(tx, te.thread_axis("threadIdx.x"))
# # s[C].bind(ty, te.thread_axis("threadIdx.y"))

# ### Shared
# # s[C_local].compute_at(s[C], tx)
# s[C_local].compute_at(s[C], thrd_fused)
# yi, xi = s[C_local].op.axis
# step_num, step_size = s[C_local].split(s[C_local].op.reduce_axis[0], Step_size)
# s[C_local].reorder(step_num, step_size, yi, xi)
# local_fused = s[C_local].fuse(yi, xi)
# s[C_local].unroll(local_fused)

# def optimize_read_cache(shared, local):
#     s[shared].compute_at(s[C_local], step_num)
#     s[local].compute_at(s[C_local], step_size)
#     y, x = s[shared].op.axis
#     fused = s[shared].fuse(y, x)
#     oo, ii = s[shared].split(fused, factor=tile_y[1]*tile_x[1])
#     s[shared].reorder(oo, ii)
#     s[shared].unroll(oo)
#     s[shared].bind(ii, te.thread_axis("threadIdx.x"))
#     # yo, yi = s[shared].split(y, tile_y[1])
#     # xo, xi = s[shared].split(x, tile_x[1])
#     # s[shared].reorder(yo, xo, yi, xi)
#     # s[shared].bind(yi, te.thread_axis("threadIdx.y"))
#     # s[shared].bind(xi, te.thread_axis("threadIdx.x"))

# optimize_read_cache(A_shared, A_local)
# optimize_read_cache(B_shared, B_local)

# func = tvm.build(s, [A, B, C], "cuda")
with open('biasadd.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

ctx = tvm.gpu(0)
A_h = np.random.uniform(size=(M, N)).astype(A.dtype)
B_h = np.random.uniform(size=(N,)).astype(B.dtype)
C_h = np.add(A_h, B_h)

A_d = tvm.nd.array(A_h, ctx)
B_d = tvm.nd.array(B_h, ctx)
C_d = tvm.nd.array(np.zeros((M, N), dtype=C.dtype), ctx)
func(A_d, B_d, C_d)
evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
print("Matmul: %f ms" % (evaluator(A_d, B_d, C_d).mean * 1e3))

np.testing.assert_allclose(C_h, C_d.asnumpy(), atol=1e-1)



# # generate kernel entry
# tvm_func_name = "tuned_fused_dot_op_float_i{0}_{1}_w{1}_{2}_o{0}_{2}_kernel0".format(
#     M, K, N
# )

# op_type = "Dot"
# parameters = {
#     "arg0_shape": [M, K], 
#     "arg1_shape": [K, N], 
#     "out_shape": [M, N],
#     "transpose_A": False,
#     "transpose_B": False
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

# import json
# with open("1.json", 'w+') as f:
#     json.dump(kernel, f)
# from kernel_db.convert_external import insert_kernel_entry
# insert_kernel_entry(kernel)