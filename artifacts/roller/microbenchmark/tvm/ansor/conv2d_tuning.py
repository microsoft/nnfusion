import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
import logging
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import utils
from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get
from utils.conv_cuda import execute


op1, op2 = None, None
# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding, path):
    filename = os.path.join(path, "ansor_conv2d_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}".format(N, CI, H, W, CO, KH, KW, strides, padding))
    if op1 is not None:
        filename = filename + "_" + op1
    if op2 is not None:
        filename = filename + "_" + op2
    return filename + ".log"

@auto_scheduler.register_workload
def conv2d_layer(N, CI, H, W, CO, KH, KW, strides, padding):
    data = te.placeholder((N, CI, H, W), name="input0")
    kernel = te.placeholder((CO, CI, KH, KW), name="input1")
    C = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype="float32")

    add = None
    if op1 == "relu":
        C = topi.nn.relu(C)
    if op1 == "add":
        add = te.placeholder(C.shape, name='input2')
        C = topi.add(C, add)
    
    if op2 == "relu":
        C = topi.nn.relu(C)
    if op2 == "add":
        add = te.placeholder(C.shape, name='input2')
        C = topi.add(C, add)
    
    if add is None:
        return [data, kernel, C]
    else:
        return [data, kernel, add, C]

def tune_conv2d_nchw(N, CI, H, W, CO, KH, KW, strides, padding, path, n_trial=1000):
    log_filename = get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding, path)
    target = tvm.target.Target("cuda")

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
    tir = str(tvm.lower(sch, args, simple_mode=True))
    source_code = task.print_best(log_filename, print_mode="cuda")
    
    kernel_filename = log_filename[:-4] + ".cc"
    grid, block = parse_launch_config(tir)
    print(grid, block)
    launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
    param = "//"+"_".join([str(N), str(CI), str(H), str(W), str(CO), str(KH), str(strides), str(padding)]) + "\n"
    for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"

    with open(kernel_filename, "w") as f:
        f.write(launch_config_as_comment + param + for_nnfusion + source_code)
    
    print("best runtime:", get(log_filename)[0] * 1000)
    # execute(kernel_filename)

def main():
    N, CI, H, W, CO, KH, KW, strides, dilation = [int(s) for s in sys.argv[1:10]]
    padding = sys.argv[10]
    
    global op1, op2
    if len(sys.argv) == 12:
        op1, op2 = sys.argv[11], None
    elif len(sys.argv) == 13:
        op1, op2 = sys.argv[11], sys.argv[12]
    else:
        op1, op2 = None, None

    # if op1 == "relu" and op2 is None:
    #     op = "Fused_Convolution_Relu"
    # elif op1 == "add" and op2 is None:
    #     op = "Fused_Convolution_Add"
    # elif op1 == "add" and op2 == "relu":
    #     op = "Fused_Convolution_Add_Relu"
    # else:
    #     op = "Convolution"

    # path = sys.argv[11] if len(sys.argv) == 12 else ""
    # path = "e2e/ansor"
    path = ""
    # log_filename = get_log_filename(N, CI, H, W, CO, KH, KW, strides, padding, path)
    # filename = log_filename[:-4]
    
    # if op1 is not None:
    #     filename = filename + "_" + op1
    # if op2 is not None:
    #     filename = filename + "_" + op2
    
    # print(op, op1, op2, filename)
    # from tvm.topi.nn.utils import get_pad_tuple
    # padding = get_pad_tuple(padding, (KH, KW))[0]
    
    # data = te.placeholder((N, CI, H, W), name="data")
    # kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    # conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype="float32")

    # res = "python3 ${{PARSE}} --op_type {7} --source_file {0}.cc --input0_shape {1} --input1_shape {2} --output0_shape {3} --stride {4} --padding {5} --dilation {6} --json_file={0}.json".format(
    #     filename, 
    #     " ".join(map(lambda x: str(x), [x for x in data.shape])), 
    #     " ".join(map(lambda x: str(x), [x for x in kernel.shape])), 
    #     " ".join(map(lambda x: str(x), [x for x in conv.shape])),
    #     " ".join(map(lambda x: str(x), [strides, strides])),
    #     " ".join(map(lambda x: str(x), [padding, padding])),
    #     " ".join(map(lambda x: str(x), [dilation, dilation])),
    #     op
    #     )
    # res = "python3 ${{INJ}} {0}.json".format(filename)
    # print(res)

    print(N, CI, H, W, CO, KH, KW, strides, padding, path)
    tune_conv2d_nchw(N, CI, H, W, CO, KH, KW, strides, padding, path)

main()
