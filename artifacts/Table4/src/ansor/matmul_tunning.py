import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
import logging
import sys
sys.path.append("..")
from utils.parse_launch_config import parse_launch_config

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(M, K, N, path):
    return os.path.join(path, "ansor_matmul_{0}_{1}_{2}.log".format(M, K, N))

@auto_scheduler.register_workload
def matmul_layer(batch, in_dim, out_dim):
    data = te.placeholder((batch, in_dim), name='A', dtype="float32")
    weight = te.placeholder((in_dim, out_dim), name='B', dtype="float32")
    k = te.reduce_axis((0, in_dim), name="k")
    matmul = te.compute((batch, out_dim), lambda x, y: te.sum(data[x, k] * weight[k, y], axis=k))
    return [data, weight, matmul]

def search_matmul_config(batch, in_dim, out_dim, path, n_trial=1000):
    log_filename = get_log_filename(batch, in_dim, out_dim, path)
    # target = tvm.target.Target("cuda")
    target = tvm.target.Target("rocm")

    task = auto_scheduler.SearchTask(
        func=matmul_layer, args=(batch, in_dim, out_dim), target=target
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
    source_code = task.print_best(log_filename, print_mode="schedule")
    # source_code = task.print_best(log_filename, print_mode="rocm")
    
    kernel_filename = log_filename[:-4] + ".cc"
    grid, block = parse_launch_config(tir)
    launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
    param = "//"+"_".join([str(batch), str(in_dim), str(out_dim)]) + "\n"
    with open(kernel_filename, "w") as f:
        f.write(launch_config_as_comment + param + source_code)
    
    print(grid, block)
    print(source_code)


def main():
    batch, in_dim, out_dim = [int(s) for s in sys.argv[1:4]]
    path = sys.argv[4] if len(sys.argv) == 5 else ""
    print(batch, in_dim, out_dim, path)
    search_matmul_config(batch, in_dim, out_dim, path)

main()