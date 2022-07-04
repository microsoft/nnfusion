import torch
from ops import *
import tvm
from tvm import auto_scheduler
import ctypes
import memopt
from memopt.tvm_build import tvm_build
import os
import hashlib

def translate_to_tvm(expr, input_dict):
    from lang.generic import einstein_v2, OUTPUT_TEMP, INPUT_TEMP
    OUTPUT_TEMP.clear()
    INPUT_TEMP.clear()
    einstein_v2(expr, input_dict)
    return INPUT_TEMP + OUTPUT_TEMP

__expr, __input_dict = None, None
@auto_scheduler.register_workload
def workload(key: int):
    return translate_to_tvm(__expr, __input_dict)

def test(expr, input_dict, name="ansor.log"):
    global __expr, __input_dict
    __expr = expr
    __input_dict = input_dict
    key = int(hashlib.md5(bytes("- einstein_v2('{}', {})".format(expr, str(input_dict)), encoding="utf-8")).hexdigest(), 16)
    task = tvm.auto_scheduler.SearchTask(func=workload, args=[key], target="cuda")
    log_file = os.path.join("temp", name)

    print("========== Task (workload key: %s) ==========" % (task.workload_key))
    print(task.compute_dag)

    def run_tuning():
        print("Begin tuning...")
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10, device=3)

        tuner = auto_scheduler.TaskScheduler([task])
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=512,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)

    # run_tuning()
    sch, args = task.apply_best(log_file)
    with memopt.Scope(sch) as scope:
        kernel_code = tvm_build(sch, args, "cuda", [], [], name="MyMatMul", global_kernel=True)
        cp = memopt.utils.CompileResult(None, kernel_code, scope.block_size, scope.grid_size, "MyMatMul", args)
        cp.append_host_call()
        cp.compile_and_load()
        print(cp.profile())

# test(*matmul_nt(4096, 128, 128))
# test(*conv_nchw(64, 64, 64, 64, 63, 3, 3))
# test(*dwconv_nhwc_v2(64, 64, 64, 64, 3, 3))
test(*matmul_nn(65, 1024, 3072))
