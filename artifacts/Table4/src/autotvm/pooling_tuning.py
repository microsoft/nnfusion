import logging
import sys
import numpy as np 

import time

import tvm
from tvm import te, topi, testing
import tvm.testing
from tvm.topi.nn.utils import get_pad_tuple


def pool_layer(pool_type, N, CI, H, W, KH, KW, strides, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    pool = topi.nn.pool2d(data, (KH, KW), (strides, strides), (1, 1), get_pad_tuple(padding, (KH, KW)), pool_type=pool_type)
    s = topi.cuda.pooling.schedule_pool(pool, "nchw")
    return s, [data, pool]

def tune_pool(pool_type, N, CI, H, W, KH, KW, strides, padding, n_trial=1000):
    # with tvm.target.Target("cuda"):
    with tvm.target.Target("rocm"):
        s, arg_bufs = pool_layer(pool_type, N, CI, H, W, KH, KW, strides, padding)
        func = tvm.build(s, arg_bufs)

    # dev = tvm.gpu()
    dev = tvm.rocm()
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    b_np = np.random.uniform(size=tuple([v.value for v in arg_bufs[1].shape])).astype(np.float32)

    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    

    func(a_tvm, b_tvm)
    evaluator = func.time_evaluator(func.entry_name, dev, number=400)
    print("Time cost of this operator: %.10f" % evaluator(a_tvm, b_tvm).mean)

    print("Lowered TIR:")
    print(tvm.lower(s, arg_bufs, simple_mode=True))
    print(func.imported_modules[0].get_source())  # print kernel code

def main():
    N, CI, H, W, K, strides = tuple([int(s) for s in sys.argv[2:8]])
    pool_type = sys.argv[1]
    padding = sys.argv[8]
    print(pool_type, N, CI, H, W, K, strides, padding)
    tune_pool(pool_type, N, CI, H, W, K, K, strides, padding)

t1 = time.time()
main()
t2 = time.time()
print("time:", t2 - t1)