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

k = n * m
A = te.placeholder((k,), name="A")
B = te.placeholder((n, m), dtype='int32', name="B")
C = te.compute((n, m), lambda i, j: A[B[i, j]], name="C")

# Reduction Factoring and Parallelization
s = te.create_schedule(C.op)
y, x = s[C].op.axis
by, ty = s[C].split(y, factor=32)
bx, tx = s[C].split(x, factor=32)
s[C].bind(by, te.thread_axis("blockIdx.y"))
s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(ty, te.thread_axis("threadIdx.y"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))

func = tvm.build(s, [A, B, C], "cuda")
with open('gather.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

# Verify the correctness of result kernel by comparing it to numpy.

ctx = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(k)).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.randint(0, k, size=(n, m), dtype='int32'), ctx)
c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
func(a, b, c)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print("Gather: %f ms" % (evaluator(a, b, c).mean * 1e3))
tvm.testing.assert_allclose(c.asnumpy(), np.take(a.asnumpy(), b.asnumpy()), rtol=1e-4)


# generate kernel entry
tvm_func_name = "tuned_fused_pad_op_float_i{0}_{1}_o{2}_{3}_kernel0".format(
    n, m,
    n, m,
)

op_type = "Pad"
parameters = {
    "input_shape": [n, m], 
    "output_shape": [n, m]
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