import logging
import sys
import numpy as np 

import tvm
from tvm import te, topi, testing
import tvm.testing
import time


def tvm_reduction_tune_op(shape, axis, keep_dim):
    data = te.placeholder(shape, name="data")
    out = topi.sum(data, axis, keep_dim)
    s = topi.cuda.reduction.schedule_reduce(out)
    return s, [data, out]

def tune_reduction(shape, axis, keep_dim=False, n_trial=1000):
    op = tvm_reduction_tune_op

    with tvm.target.Target("cuda"):
        s, arg_bufs = op(shape, axis, keep_dim)
        func = tvm.build(s, arg_bufs)

    out_shape = []
    for i in range(len(shape)):
        if isinstance(axis, int) and i == axis:
            continue
        if isinstance(axis, tuple) and i in axis:
            continue
        out_shape.append(shape[i])
    print(out_shape)

    dev = tvm.cuda()
    a_np = np.random.uniform(size=shape).astype(np.float32)
    c_np = np.random.uniform(size=out_shape).astype(np.float32)
    a_tvm = tvm.nd.array(a_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)

    func(a_tvm, c_tvm)
    evaluator = func.time_evaluator(func.entry_name, dev, number=1000)
    print("Time cost of this operator: %.10f" % evaluator(a_tvm, c_tvm).mean)

    print("Lowered TIR:")
    print(tvm.lower(s, arg_bufs, simple_mode=True))
    print(func.imported_modules[0].get_source())  # print kernel code

def main():
    shape = tuple([int(s) for s in sys.argv[1:-1]])
    axis = int(sys.argv[-1])
    if axis + 1 < len(shape):
        axis = tuple([x for x in range(axis, len(shape))])
    path = ""
    # path = sys.argv[4] if len(sys.argv) == 5 else ""
    print(shape, "axis:", axis)
    tune_reduction(shape, axis)

start_time = time.time()
main()
print("compilation time: %s" % (time.time() - start_time))