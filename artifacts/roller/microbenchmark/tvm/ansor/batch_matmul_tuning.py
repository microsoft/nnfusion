import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
import logging
import sys
sys.path.append("..")
from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(batch, C, M, K, N, path):
    return os.path.join(path, "ansor_batch_matmul_{0}_{1}_{2}_{3}_{4}.log".format(batch, C, M, K, N))

@auto_scheduler.register_workload
def batch_matmul_layer(batch, C, M, K, N):
    A = te.placeholder((batch * C, M, K), name='A')
    B = te.placeholder((batch * C, K, N), name='B')
    C = topi.nn.batch_matmul(A, B, transpose_a=False, transpose_b=False)
    return [A, B, C]

def search_batch_matmul_config(batch, C, M, K, N, path, n_trial=1000):
    log_filename = get_log_filename(batch, C, M, K, N, path)
    target = tvm.target.Target("cuda")

    task = auto_scheduler.SearchTask(
        func=batch_matmul_layer, args=(batch, C, M, K, N), target=target
    )

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300, timeout=20)
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
    param = "//"+"_".join([str(batch), str(C), str(M), str(K), str(N)]) + "\n"
    for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"
    with open(kernel_filename, "w") as f:
        f.write(launch_config_as_comment + param + for_nnfusion + source_code)
    print("best runtime:", get(log_filename)[0] * 1000)

def main():
    batch, C, M, K, N = [int(s) for s in sys.argv[1:6]]
    path = sys.argv[6] if len(sys.argv) == 7 else ""

    # A = te.placeholder((batch * C, M, K), name='A')
    # B = te.placeholder((batch * C, K, N), name='B')
    # out = topi.nn.batch_matmul(A, B, transpose_a=False, transpose_b=False)

    # log_filename = get_log_filename(batch, C, M, K, N, path)
    # res = "python3 ${{PARSE}} --op_type BatchMatMul --source_file {0}.cc --input0_shape {1} --input1_shape {2} --output0_shape {3} --transpose_A False --transpose_B False --json_file={0}.json".format(
    #     log_filename[:-4], 
    #     " ".join([str(batch), str(C), str(M), str(K)]), 
    #     " ".join([str(batch), str(C), str(K), str(N)]), 
    #     " ".join([str(batch), str(C), str(M), str(N)]))
    # res = "python3 ${{INJ}} {0}.json".format(log_filename[:-4])
    # print(res)
    
    print(batch, C, M, K, N)
    search_batch_matmul_config(batch, C, M, K, N, path)

main()