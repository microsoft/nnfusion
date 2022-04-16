from __future__ import absolute_import, print_function

import sys
import tvm
import tvm.testing
from tvm import te
import numpy as np
from codegen import CodeGenerator

# Describe Sum of Rows
n = 4096
m = 8192
for i in range(len(sys.argv)):
    if sys.argv[i] == '-n':
        n = int(sys.argv[i+1])
    if sys.argv[i] == '-m':
        m = int(sys.argv[i+1])
print("N: {} M: {}".format(n, m))

A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")

s = te.create_schedule(B.op)
tile_dict = {"i": [32, 1], "k": [32]}
bind_dict = {"space": ["blockIdx.x", "threadIdx.y", None], "reduce": [None, "threadIdx.x"]}

generator = CodeGenerator()
new_s, new_args = generator.rewrite_schedule(s, tile_dict, bind_dict, False, False)
func = tvm.build(new_s, new_args, "cuda")
# no, ni = s[B].split(B.op.axis[0], factor=32)
# ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

# s[B].bind(no, te.thread_axis("blockIdx.x"))
# s[B].bind(ni, te.thread_axis("threadIdx.y"))

# tx = te.thread_axis("threadIdx.x")
# s[B].bind(ki, tx)
# s[B].set_store_predicate(tx.var.equal(0))

# func = tvm.build(s, [A, B], "cuda")
with open('reduce-sum.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

# Verify the correctness of result kernel by comparing it to numpy.
ctx = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
func(a, b)
evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
print("Reduce Sum: %f ms" % (evaluator(a, b).mean * 1e3))
tvm.testing.assert_allclose(b.asnumpy(), np.sum(a.asnumpy(), axis=1), rtol=1e-4)



# generate kernel entry
tvm_func_name = "tuned_fused_sum_op_float_i{0}_{1}_o0_{1}_kernel0".format(
    n, m,
)

op_type = "Sum"
parameters = {
    "input_shape": [n, m], 
    "output_shape": [0, m]
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