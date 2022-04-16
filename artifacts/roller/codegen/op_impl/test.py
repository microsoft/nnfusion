import logging
import sys

import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing

from tvm import autotvm

# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding):
    return "conv2d_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.log".format(N, CI, H, W, CO, KH, KW, strides, padding)

def tune_conv2d_nchw(N, CI, H, W, CO, KH, KW, strides, padding, n_trial = 2000):
    log_filename = get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    task = autotvm.task.create(
        "conv2d_nchw.cuda", args=(data, kernel, strides, padding, 1, "float32"), target="cuda"
    )
    print(task.config_space)

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner = autotvm.tuner.XGBTuner(task)
    # n_trial = min(n_trial, len(task.config_space))
    tuner.tune(
        n_trial=n_trial,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_filename)],
    )

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(log_filename)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

def main():
    N, CI, H, W, CO, KH, KW, strides, dilation, padding = [int(s) for s in sys.argv[1:]]
    print(N, CI, H, W, CO, KH, KW, strides, padding)
    tune_conv2d_nchw(N, CI, H, W, CO, KH, KW, strides, padding)

main()
