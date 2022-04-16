import tvm
from tvm import te, topi, testing, autotvm
import logging
from tvm.contrib.pickle_memoize import memoize
import json
import os
import sys

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(N, CI, H, W, KH, KW, strides, padding, path):
    return os.path.join(path, "depthwise_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.log".format(N, CI, H, W, KH, KW, strides, padding))

def search_depthwise_conv2d_nchw_configs(N, CI, H, W, KH, KW, strides, padding, path, n_trial=1000):
    input_shape, filter_shape= (N, CI, H, W), (CI, 1, KH, KW)
    log_filename = get_log_filename(N, CI, H, W, KH, KW, strides, padding, path)
    
    data = te.placeholder(input_shape, name='data', dtype="float32")
    kernel = te.placeholder(filter_shape, name='kernel', dtype="float32")

    # task = autotvm.task.create("depthwise_conv2d_nchw.cuda", args=(data, kernel, strides, padding, 1, "float32"), target='cuda')
    task = autotvm.task.create("depthwise_conv2d_nchw.cuda", args=(data, kernel, strides, padding, 1, "float32"), target='rocm')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=1, min_repeat_ms=100, timeout=200)
    )

    tuner = autotvm.tuner.XGBTuner(task)
    # n_trial = min(n_trial, len(task.config_space))
    if not path:
        tuner.tune(n_trial=n_trial, measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(log_filename)])
    
    # inspect the best config
    dispatch_context = autotvm.apply_history_best(log_filename)
    best_config = dispatch_context.query(task.target, task.workload)
    print(log_filename)
    print("\nBest config:")
    print(best_config)
    with dispatch_context:
        # with tvm.target.create('cuda'):
        with tvm.target.create('rocm'):
            s, arg_bufs = task.instantiate(best_config)
            print("Lowered TIR:")
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            # func = tvm.build(s, arg_bufs, 'cuda', name='matmul')
            func = tvm.build(s, arg_bufs, 'rocm', name='matmul')
            print(func.imported_modules[0].get_source())  # print kernel code


def main():
    N, CI, H, W, KH, KW, strides = [int(s) for s in sys.argv[1:8]]
    padding = sys.argv[8]
    path = sys.argv[9] if len(sys.argv) == 10 else ""
    print(N, CI, H, W, KH, KW, strides, padding, path)
    search_depthwise_conv2d_nchw_configs(N, CI, H, W, KH, KW, strides, padding, path)

main()
