from __future__ import absolute_import, print_function

import sys
import tvm
import tvm.testing
from tvm import te
import numpy as np
from codegen import CodeGenerator

# Describe Sum of Rows
n = 8192
m = 8192
if len(sys.argv) == 3:
    n = int(sys.argv[1])
    m = int(sys.argv[2])

print("N: {} M: {}".format(n, m))
A = te.placeholder((n, m), name="A")
B = te.placeholder((n, m), name="B")
C = te.compute((n, m), lambda i, j: A[i, j] + B[i, j], name="C")
s = te.create_schedule(C.op)

generator = CodeGenerator()
tile_dict = {"i":[32, 1], "j":[32,1]}
bind_dict = {"space": ["blockIdx.x", "threadIdx.x", None], "reduce": [None, None]}
new_s, new_args = generator.rewrite_schedule(s, tile_dict, bind_dict, False, False)
func = tvm.build(new_s, new_args, "cuda")

# Reduction Factoring and Parallelization
# y, x = C.op.axis
# by, ty = s[C].split(y, factor=32)
# bx, tx = s[C].split(x, factor=32)
# s[C].bind(by, te.thread_axis("blockIdx.y"))
# s[C].bind(bx, te.thread_axis("blockIdx.x"))
# s[C].bind(ty, te.thread_axis("threadIdx.y"))
# s[C].bind(tx, te.thread_axis("threadIdx.x"))

# func = tvm.build(s, [A, B, C], "cuda")
with open('element-wise.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

# Verify the correctness of result kernel by comparing it to numpy.
ctx = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=(n, m)).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
func(a, b, c)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print("Element-wise: %f ms" % (evaluator(a, b, c).mean * 1e3))
tvm.testing.assert_allclose(c.asnumpy(), np.add(a.asnumpy(), b.asnumpy()), rtol=1e-4)



# generate kernel entry
tvm_func_name = "tuned_fused_Add_op_float_i{0}_{1}_o{0}_{1}_kernel0".format(
    n, m,
)

op_type = "Add"
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