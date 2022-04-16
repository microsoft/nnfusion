import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
import logging
import sys
sys.path.append("..")
from utils.parse_launch_config import parse_launch_config

# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding, path):
    return os.path.join(path, "ansor_conv2d_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.log".format(N, CI, H, W, CO, KH, KW, strides, padding))

@auto_scheduler.register_workload
def conv2d_layer(N, CI, H, W, CO, KH, KW, strides, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

def tune_conv2d_nchw(N, CI, H, W, CO, KH, KW, strides, padding, path, n_trial=1000):
    log_filename = get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding, path)
    # target = tvm.target.Target("cuda")
    target = tvm.target.Target("rocm")

    task = auto_scheduler.SearchTask(
        func=conv2d_layer, args=(N, CI, H, W, CO, KH, KW, strides, padding), target=target
    )

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trial,  # change this to 1000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_filename)],
        verbose=2,
    )

    # Run auto-tuning (search)
    if not path:
        task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_filename)
    print(log_filename)
    print("Lowered TIR:")
    tir = str(tvm.lower(sch, args, simple_mode=True))
    # source_code = task.print_best(log_filename, print_mode="cuda")
    source_code = task.print_best(log_filename, print_mode="schedule")
    
    kernel_filename = log_filename[:-4] + ".cc"
    grid, block = parse_launch_config(tir)
    print(grid, block)
    launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
    param = "//"+"_".join([str(N), str(CI), str(H), str(W), str(CO), str(KH), str(strides), str(padding)]) + "\n"
    with open(kernel_filename, "w") as f:
        f.write(launch_config_as_comment + param + source_code)
    
    print(grid, block)
    print(source_code)

def main():
    N, CI, H, W, CO, KH, KW, strides, dilation, padding = [int(s) for s in sys.argv[1:11]]
    path = sys.argv[11] if len(sys.argv) == 12 else ""
    print(N, CI, H, W, CO, KH, KW, strides, padding, path)
    tune_conv2d_nchw(N, CI, H, W, CO, KH, KW, strides, padding, path)

main()