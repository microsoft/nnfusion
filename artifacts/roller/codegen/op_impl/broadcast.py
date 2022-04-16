from __future__ import absolute_import, print_function

import sys
import tvm
import tvm.testing
from tvm import te
import numpy as np

# Describe Sum of Rows
n = 4096
m = 8192
for i in range(len(sys.argv)):
    if sys.argv[i] == '-n':
        n = int(sys.argv[i+1])
    if sys.argv[i] == '-m':
        m = int(sys.argv[i+1])
print("N: {} M：{}".format(n, m))

A = te.placeholder((n,), name="A")
C = te.compute((n, m), lambda i, j: A[i], name="C")

# Reduction Factoring and Parallelization
s = te.create_schedule(C.op)
y, x = C.op.axis
bx, tx = s[C].split(x, factor=32)
by, ty = s[C].split(y, factor=32)
s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(by, te.thread_axis("blockIdx.y"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))
s[C].bind(ty, te.thread_axis("threadIdx.y"))

func = tvm.build(s, [A, C], "cuda")
with open('broadcast.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

# Verify the correctness of result kernel by comparing it to numpy.
ctx = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(n)).astype(A.dtype), ctx)
c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
func(a, c)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print("Broadcast: %f ms" % (evaluator(a, c).mean * 1e3))
tvm.testing.assert_allclose(c.asnumpy(), np.transpose(np.tile(a.asnumpy(), (m, 1))), rtol=1e-4)