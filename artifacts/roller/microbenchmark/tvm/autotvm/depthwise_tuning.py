import tvm
from tvm import te, topi, testing, autotvm
import logging
from tvm.contrib.pickle_memoize import memoize
import json
import os
import sys

sys.path.append("..")
from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get
from utils.depthwise_cuda import execute

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(N, CI, H, W, KH, KW, strides, padding, path):
    return os.path.join(path, "depthwise_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.log".format(N, CI, H, W, KH, KW, strides, padding))

def search_depthwise_conv2d_nchw_configs(N, CI, H, W, KH, KW, strides, padding, path, n_trial=1000):
    input_shape, filter_shape= (N, CI, H, W), (CI, 1, KH, KW)
    log_filename = get_log_filename(N, CI, H, W, KH, KW, strides, padding, path)
    
    data = te.placeholder(input_shape, name='data', dtype="float32")
    kernel = te.placeholder(filter_shape, name='kernel', dtype="float32")

    task = autotvm.task.create("depthwise_conv2d_nchw.cuda", args=(data, kernel, strides, padding, 1, "float32"), target='cuda')
    # print(task.config_space)
    print(task.flop)
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
        with tvm.target.Target('cuda'):
            s, arg_bufs = task.instantiate(best_config)
            tir = str(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs, 'cuda', name='depthwise')
            source_code = func.imported_modules[0].get_source()
            kernel_filename = log_filename[:-4] + ".cc"
            grid, block = parse_launch_config(tir)
            launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
            param = "//"+"_".join([str(N), str(CI), str(H), str(W), str(KH), str(strides), str(padding)]) + "\n"
            for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"
            with open(kernel_filename, "w") as f:
                f.write(launch_config_as_comment + param + for_nnfusion + source_code)
            
            print("best runtime:", get(log_filename)[0] * 1000)
            # execute(kernel_filename)

def main():
    N, CI, H, W, KH, KW, strides, dilation = [int(s) for s in sys.argv[1:9]]
    padding = sys.argv[9]
    path = sys.argv[10] if len(sys.argv) == 11 else ""
    print(N, CI, H, W, KH, KW, strides, padding, dilation, path)

    search_depthwise_conv2d_nchw_configs(N, CI, H, W, KH, KW, strides, padding, path)

main()
