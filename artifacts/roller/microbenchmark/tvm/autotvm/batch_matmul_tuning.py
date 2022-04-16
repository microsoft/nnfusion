import tvm
import logging
import sys
from tvm import autotvm
from tvm import te, topi, testing
import json
import os
from tvm.topi.utils import traverse_inline, get_const_tuple

sys.path.append("..")
from utils.parse_launch_config import parse_launch_config
from utils.get_best_config import get

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(batch, C, M, K, N, path):
    return os.path.join(path, "batch_matmul_{0}_{1}_{2}_{3}_{4}.log".format(batch, C, M, K, N))


def search_batch_matmul_config(batch, C, M, K, N, path, n_trial=1000):
    data = te.placeholder((batch * C, M, K), name='A')
    weight = te.placeholder((batch * C, K, N), name='B')

    task = autotvm.task.create("batch_matmul.cuda", args=(data, weight, None, None, False, False), target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=1, min_repeat_ms=100, timeout=200)
    )

    log_filename = get_log_filename(batch, C, M, K, N, path)

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
            print("Lowered TIR:")
            tir = str(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs, 'cuda', name='conv')
            source_code = func.imported_modules[0].get_source()
            kernel_filename = log_filename[:-4] + ".cc"
            grid, block = parse_launch_config(tir)
            launch_config_as_comment = "//"+"_".join(map(lambda x: str(x), grid + block)) + "\n"
            param = "//"+"_".join([str(batch), str(C), str(M), str(K), str(N)]) + "\n"
            for_nnfusion = "//dim3 grid(" + ", ".join(map(lambda x: str(x), grid)) + ");\n" + "//dim3 block(" + ", ".join(map(lambda x: str(x), block)) + ");\n"
            with open(kernel_filename, "w") as f:
                f.write(launch_config_as_comment + param + for_nnfusion + source_code)
            
            print("best runtime:", get(log_filename)[0] * 1000)
            # execute(kernel_filename)



def main():
    batch, C, M, K, N = [int(s) for s in sys.argv[1:6]]
    path = sys.argv[6] if len(sys.argv) == 7 else ""
    print(batch, C, M, K, N)
    search_batch_matmul_config(batch, C, M, K, N, path)

main()