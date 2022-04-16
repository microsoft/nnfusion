import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
import logging
import sys
sys.path.append("..")
from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get

# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(t, path, *shape):
    return os.path.join(path, "ansor_{0}_{1}.log".format(t, "_".join([str(s) for s in shape])))

@auto_scheduler.register_workload
def biasadd_layer(*shape):
    data1 = te.placeholder(shape, name="data1")
    data2 = te.placeholder(shape[1:], name="data2")
    add = te.compute(shape, lambda x, y: data1[x, y] + data2[y])
    return [data1, data2, add]

@auto_scheduler.register_workload
def add_layer(*shape):
    data1 = te.placeholder(shape, name="data1")
    data2 = te.placeholder(shape, name="data2")
    add = topi.add(data1, data2)
    return [data1, data2, add]

@auto_scheduler.register_workload
def mul_layer(*shape):
    data1 = te.placeholder(shape, name="data1")
    data2 = te.placeholder(shape, name="data2")
    mul = topi.multiply(data1, data2)
    return [data1, data2, mul]

@auto_scheduler.register_workload
def tanh_layer(*shape):
    data = te.placeholder(shape, name="data")
    tanh = topi.tanh(data)
    return [data, tanh]

@auto_scheduler.register_workload
def sigmoid_layer(*shape):
    data = te.placeholder(shape, name="data")
    sigmoid = topi.sigmoid(data)
    return [data, sigmoid]

@auto_scheduler.register_workload
def relu_layer(*shape):
    data = te.placeholder(shape, name="data")
    relu = topi.nn.relu(data)
    return [data, relu]

@auto_scheduler.register_workload
def transpose_layer(*shape):
    data = te.placeholder(shape, name="data")
    transpose = topi.transpose(data)
    return [data, transpose]

def tune_injective(t, shape, path, n_trial=1000):
    log_filename = get_log_filename(t, path, *shape)
    target = tvm.target.Target("cuda")
    layer = None
    if t == "add":
        layer = add_layer
    elif t == "biasadd":
        layer = biasadd_layer
    elif t == "transpose":
        layer = transpose_layer
    elif t == "mul":
        layer = mul_layer
    elif t == "tanh":
        layer = tanh_layer
    elif t == "sigmoid":
        layer = sigmoid_layer
    elif t == "relu":
        layer = relu_layer
    else:
        raise ValueError("unrecognized type: " + t)
    
    task = auto_scheduler.SearchTask(
        func=layer, args=shape, target=target
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
    launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
    param = "//"+"_".join([t] + [str(s) for s in shape]) + "\n"
    for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"
    with open(kernel_filename, "w") as f:
        f.write(launch_config_as_comment + param + for_nnfusion + source_code)

    print("best runtime:", get(log_filename)[0] * 1000)

def main():
    shape = tuple([int(s) for s in sys.argv[2:]])
    path = "ansor/injective"
    # path = sys.argv[4] if len(sys.argv) == 5 else ""
    t = sys.argv[1]
    print(t, shape, path)
    tune_injective(t, shape, path)

main()