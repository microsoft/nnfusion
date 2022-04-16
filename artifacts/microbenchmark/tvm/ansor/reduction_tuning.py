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
from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get, get1

# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(path, shape, axis):
    return os.path.join(path, "ansor_{0}_{1}_axis_{2}.log".format("reduction", "_".join([str(s) for s in shape]), axis))

@auto_scheduler.register_workload
def reduction_layer(shape, axis, keep_dim):
    data = te.placeholder(shape, name="data")
    reduction = topi.sum(data, axis, keep_dim)
    return [data, reduction]

def tune_reduction(shape, axis, keep_dim=False, path="", n_trial=1000):
    log_filename = get_log_filename(path, shape, axis)
    target = tvm.target.Target("cuda")
    layer = reduction_layer

    task = auto_scheduler.SearchTask(
        func=layer, args=(shape, axis, keep_dim), target=target
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
    param = "//"+"_".join([str(s) for s in shape] + ["axis"] + [str(axis)]) + "\n"
    for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"

    with open(kernel_filename, "w") as f:
        f.write(launch_config_as_comment + param + for_nnfusion + source_code)

    print("best runtime: ", get(log_filename)[0] * 1000)
    print("compilation time: ", get1(log_filename))

def main():
    shape = []
    path = ""
    for s in sys.argv[1:]:
        try:
            d = int(s)
            shape.append(d)
        except ValueError:
            path = s
    
    axis = int(shape[-1])
    shape = tuple(shape[:-1])
    if axis + 1 < len(shape):
        axis = tuple([x for x in range(axis, len(shape))])
    
    print(shape, "axis:", axis)
    tune_reduction(shape, axis, path=path)

main()
