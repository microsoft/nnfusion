import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
import logging
import sys
sys.path.append("..")
from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get
from utils.depthwise_cuda import execute

# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(N, CI, H, W, KH, KW, strides, padding, path):
    return os.path.join(path, "ansor_depthwise_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.log".format(N, CI, H, W, KH, KW, strides, padding))

@auto_scheduler.register_workload
def depthwise_conv2d_layer(N, CI, H, W, KH, KW, strides, padding):
    input_shape, filter_shape= (N, CI, H, W), (CI, 1, KH, KW)
    data = te.placeholder(input_shape, name='data', dtype="float32")
    kernel = te.placeholder(filter_shape, name='kernel', dtype="float32")
    depthwise_conv = topi.nn.depthwise_conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype="float32")
    return [data, kernel, depthwise_conv]

def search_depthwise_conv2d_nchw_configs(N, CI, H, W, KH, KW, strides, padding, path, n_trial=1000):
    log_filename = get_log_filename(N, CI, H, W, KH, KW, strides, padding, path)
    target = tvm.target.Target("cuda")

    task = auto_scheduler.SearchTask(
        func=depthwise_conv2d_layer, args=(N, CI, H, W, KH, KW, strides, padding), target=target
    )
    print(task.compute_dag.flop_ct)

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
    tir = str(tvm.lower(sch, args, simple_mode=True))
    source_code = task.print_best(log_filename, print_mode="cuda")
    
    kernel_filename = log_filename[:-4] + ".cc"
    grid, block = parse_launch_config(tir)
    launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
    param = "//"+"_".join([str(N), str(CI), str(H), str(W), str(KH), str(strides), str(padding)]) + "\n"
    for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"
    with open(kernel_filename, "w") as f:
        f.write(launch_config_as_comment + param + for_nnfusion + source_code)
    
    print("best runtime:", get(log_filename)[0] * 1000)
    execute(kernel_filename)
    
def main():
    N, CI, H, W, KH, KW, strides, dilation = [int(s) for s in sys.argv[1:9]]
    padding = sys.argv[9]
    path = sys.argv[10] if len(sys.argv) == 11 else ""

    # log_filename = get_log_filename(N, CI, H, W, KH, KW, strides, padding, path)
    # from tvm.topi.nn.utils import get_pad_tuple
    # padding = get_pad_tuple(padding, (KH, KW))[0]

    # input_shape, filter_shape= (N, CI, H, W), (CI, 1, KH, KW)
    # data = te.placeholder(input_shape, name='data', dtype="float32")
    # kernel = te.placeholder(filter_shape, name='kernel', dtype="float32")
    # depthwise_conv = topi.nn.depthwise_conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype="float32")

    # res = "python3 ${{PARSE}} --op_type DepthwiseConv2dNative --source_file {0}.cc --input0_shape {1} --input1_shape {2} --output0_shape {3} --stride {4} --padding {5} --dilation {6} --json_file={0}.json".format(
    #     log_filename[:-4], 
    #     " ".join(map(lambda x: str(x), [x for x in data.shape])), 
    #     " ".join(map(lambda x: str(x), [x for x in kernel.shape])), 
    #     " ".join(map(lambda x: str(x), [x for x in depthwise_conv.shape])),
    #     " ".join(map(lambda x: str(x), [strides, strides])),
    #     " ".join(map(lambda x: str(x), [padding, padding])),
    #     " ".join(map(lambda x: str(x), [dilation, dilation])),
    #     )
    # res = "python3 ${{INJ}} {0}.json".format(log_filename[:-4])
    # print(res)

    print(N, CI, H, W, KH, KW, strides, padding, dilation, path)
    search_depthwise_conv2d_nchw_configs(N, CI, H, W, KH, KW, strides, padding, path)

main()
