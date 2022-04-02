import onnx
import numpy as np
import os.path as osp
import time
import tvm
from tvm import te
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

prefix = "temp"
target = tvm.target.cuda(arch="sm_61")

@auto_scheduler.register_workload
def matmul(n, m, l):
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((l, m), name='B')
    C = te.compute((n, m),
                    lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                    name='C')
    return [A, B, C]

def get_data(n, m, k, constructor=None):
    """Return random a, b and empty c with the same shape.
    """
    np.random.seed(0)
    a = np.random.normal(size=(n, k)).astype(np.float32)
    b = np.random.normal(size=(k, m)).astype(np.float32)
    c = np.empty((n, m), dtype='float32')
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

from d2ltvm import bench_workload
def bench_matmul_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    times = []
    for size in sizes:
        n, m, k = size
        s, (A, B, C) = func(n, m, k)
        mod = tvm.build(s, [A, B, C], target)
        ctx = tvm.device(target, 3)
        a, b, c = get_data(n, m, k, lambda x: tvm.nd.array(x, device=ctx))
        times.append(bench_workload(workload))
    return np.array(times)

n, m, l = 4096, 128, 128
task = tvm.auto_scheduler.SearchTask(func=matmul, args=(n, m, l), target=target)
log_file = osp.join(prefix, "ansor_tune.log")

print("========== Task (workload key: %s) ==========" % (task.workload_key))
print(task.compute_dag)

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10, device=3)

    tuner = auto_scheduler.TaskScheduler([task])
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

# run_tuning()
sch, args = task.apply_best(log_file)

import memopt
passes = [
    # (0, memopt.debug_pass),
    (0, memopt.modify_input_pass),
    (0, memopt.modify_output_pass),
    (4, memopt.get_kernel_info_pass),
]
with tvm.transform.PassContext(config={"tir.add_lower_pass": passes}), \
    memopt.Scope(sch) as scope:
    scope.shared_mem_outputs = ["C"]
    scope.shared_mem_inputs = ["A"]
    mod = tvm.build(sch, args, target=target)


kernel_code = mod.imported_modules[0].get_source()
print(kernel_code)
def tile(n, m, k):
    return sch, args
tms = bench_matmul_tvm(tile, [(n, m, l)], 'cuda')
print("Result", tms)

# # Compile with the history best
# print("Compile...")
# with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
#     mod = tvm.build(sch, [X, K, Y], target=target)
