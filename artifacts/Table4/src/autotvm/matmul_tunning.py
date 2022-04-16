import tvm
import logging
import sys
from tvm import autotvm, relay
from tvm import te, topi, testing
import json
import os

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def get_log_filename(M, K, N, path):
    return os.path.join(path, "matmul_{0}_{1}_{2}.log".format(M, K, N))

def search_matmul_config(batch, in_dim, out_dim, path, tc=False, n_trial=1000):
    print("n_trial:", n_trial)
    data = te.placeholder((batch, in_dim), name='A', dtype="float32")
    # weight = te.placeholder((out_dim, in_dim), name='B', dtype="float32")
    weight = te.placeholder((in_dim, out_dim), name='B', dtype="float32")

    # schedule = "dense_small_batch.gpu" if batch == 1 else "dense_large_batch.gpu"
    schedule = "dense.rocm"
    if tc:
        schedule = "dense_tensorcore.cuda"
    print(schedule)
    # task = autotvm.task.create(schedule, args=(
    #     data, weight), target='cuda')
    task = autotvm.task.create(schedule, args=(
        data, weight), target='rocm')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=1, min_repeat_ms=100, timeout=200)
    )

    log_filename = get_log_filename(batch, in_dim, out_dim, path)

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
        # with tvm.target.create('cuda'):
        with tvm.target.create('rocm'):
            s, arg_bufs = task.instantiate(best_config)
            print("Lowered TIR:")
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            # func = tvm.build(s, arg_bufs, 'cuda', name='matmul')
            func = tvm.build(s, arg_bufs, 'rocm', name='matmul')
            print(func.imported_modules[0].get_source())  # print kernel code


def main():
    batch, in_dim, out_dim, tc = [int(s) for s in sys.argv[1:5]]
    path = sys.argv[5] if len(sys.argv) == 6 else ""
    print(batch, in_dim, out_dim, tc, path)
    search_matmul_config(batch, in_dim, out_dim, path, tc)

main()