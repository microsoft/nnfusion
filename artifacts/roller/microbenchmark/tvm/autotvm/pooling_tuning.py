import logging
import sys
import numpy as np 

import tvm
from tvm import te, topi, testing
import tvm.testing
from tvm.topi.nn.utils import get_pad_tuple
import time
import os
from utils.parse_launch_config import parse_launch_config


def get_log_filename(pool_type, path, *shape):
    return os.path.join("{0}_pooling_{1}.log".format(pool_type, "_".join([str(s) for s in shape])))

def pool_layer(pool_type, N, CI, H, W, KH, KW, strides, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    pool = topi.nn.pool2d(data, (KH, KW), (strides, strides), (1, 1), get_pad_tuple(padding, (KH, KW)), pool_type=pool_type)
    s = topi.cuda.pooling.schedule_pool(pool, "nchw")
    return s, [data, pool]
    
def tune_pool(pool_type, N, CI, H, W, KH, KW, strides, padding, path, n_trial=1000):
    # log_filename = get_log_filename(pool_type, path, N, CI, H, W, KH, KW, strides, padding)
    with tvm.target.Target("cuda"):
        s, arg_bufs = pool_layer(pool_type, N, CI, H, W, KH, KW, strides, padding)
        func = tvm.build(s, arg_bufs)

    dev = tvm.cuda()
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    b_np = np.random.uniform(size=tuple([v.value for v in arg_bufs[1].shape])).astype(np.float32)

    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    
    # log_filename = get_log_filename(pool_type, path, *shape)

    func(a_tvm, b_tvm)
    evaluator = func.time_evaluator(func.entry_name, dev, number=400)
    print("Time cost of this operator: %.10f" % evaluator(a_tvm, b_tvm).mean)
    tir = str(tvm.lower(s, arg_bufs, simple_mode=True))
    source_code = func.imported_modules[0].get_source()
    kernel_filename = log_filename[:-4] + ".cc"
    grid, block = parse_launch_config(tir)
    launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
    param = "//"+"_".join([str(pool_type), str(N), str(CI), str(H), str(W), str(KH), str(strides), str(padding)]) + "\n"
    for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"

    with open(kernel_filename, "w") as f:
        f.write(launch_config_as_comment + param + for_nnfusion + source_code)
    
def main():
    N, CI, H, W, K, strides = tuple([int(s) for s in sys.argv[2:8]])
    pool_type = sys.argv[1]
    padding = sys.argv[8]
    path = sys.argv[9] if len(sys.argv) == 10 else ""
    print(pool_type, N, CI, H, W, K, strides, padding)
    tune_pool(pool_type, N, CI, H, W, K, K, strides, padding, path)

start_time = time.time()
main()
print("compilation time: %s" % (time.time() - start_time))