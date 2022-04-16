from __future__ import absolute_import, print_function

import sys
import tvm
import tvm.testing
from tvm import te
import numpy as np

# Describe Sum of Rows
n = 8192
m = 8192

for i in range(len(sys.argv)):
    if sys.argv[i] == '-n':
        n = int(sys.argv[i+1])
    if sys.argv[i] == '-m':
        m = int(sys.argv[i+1])
print("N: {} M: {}".format(n, m))

A = te.placeholder((n, m), name="A")
C = te.compute((n+2, m+2), lambda i, j: te.if_then_else(tvm.tir.all(i > 0, i <= n, j > 0, j <= m), A[i-1, j-1], 0.0), name="C")

# Reduction Factoring and Parallelization
s = te.create_schedule(C.op)
y, x = s[C].op.axis
by, ty = s[C].split(y, factor=32)
bx, tx = s[C].split(x, factor=32)
s[C].bind(by, te.thread_axis("blockIdx.y"))
s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(ty, te.thread_axis("threadIdx.y"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))

func = tvm.build(s, [A, C], "cuda")
with open('pad.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

# Verify the correctness of result kernel by comparing it to numpy.
ctx = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), ctx)
c = tvm.nd.array(np.zeros((n+2, m+2), dtype=C.dtype), ctx)
func(a, c)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print("Reduce Sum: %f ms" % (evaluator(a, c).mean * 1e3))
tvm.testing.assert_allclose(c.asnumpy(), np.pad(a.asnumpy(), (1, 1), 'constant', constant_values=(0.0, 0.0)), rtol=1e-4)


# generate kernel entry
tvm_func_name = "tuned_fused_pad_op_float_i{0}_{1}_o{2}_{3}_kernel0".format(
    n, m,
    n + 2, m + 2
)

op_type = "Pad"
parameters = {
    "input_shape": [n, m], 
    "output_shape": [n + 2, m + 2]
}

gridDim = [v.value if not isinstance(v, int) else v for v in generator.blck_grid]
blockDim = [v.value if not isinstance(v, int) else v for v in generator.thrd_grid]
code = func.imported_modules[0].get_source()

kernel = {
    'tvm_func_name': tvm_func_name,
    'op_type': op_type, 
    'parameters': parameters, 
    'code': code, 
    'gridDim': gridDim,
    'blockDim': blockDim
}

from kernel_db.convert_external import insert_kernel_entry
insert_kernel_entry(kernel)