import numpy as np
import tvm
from tvm import te

def reduction(n):
    r = te.reduce_axis((0, n), name='r')
    X = te.placeholder((n, ), name='X')
    Y = te.compute(
        [1], lambda i : te.sum(X[r], axis=[r]), name='Y')
    return X, Y

def default_sch(n):
    X, Y = reduction(n)
    sch = te.create_schedule(Y.op)
    r = sch[Y].op.reduce_axis[0]
    # o = sch[Y].op.axis[0]
    ro, ri = sch[Y].split(r, factor=1024)
    sch[Y].reorder(ro, ri)

    sch[Y].bind(ro, te.thread_axis("blockIdx.x"))
    sch[Y].bind(ri, te.thread_axis("threadIdx.x"))
    print(tvm.lower(sch, [X, Y], simple_mode=True))
    return sch, (X, Y)

target = tvm.target.cuda(arch="sm_61")
n = 8192
sch, (X, Y) = default_sch(n)
mod = tvm.build(sch, [X, Y], target=target)
kernel_code = mod.imported_modules[0].get_source()
print(kernel_code)
