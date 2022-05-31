
import torch
import tvm
from tvm import auto_scheduler
import memopt
import os

def translate_to_tvm(expr, input_dict):
    from lang.generic import einstein_v2, OUTPUT_TEMP, INPUT_TEMP
    OUTPUT_TEMP.clear()
    INPUT_TEMP.clear()
    einstein_v2(expr, input_dict)
    return INPUT_TEMP + OUTPUT_TEMP

__expr, __input_dict = None, None
@auto_scheduler.register_workload
def workload():
    return translate_to_tvm(__expr, __input_dict)

def test(expr, input_dict, name="ansor.log"):
    global __expr, __input_dict
    __expr = expr
    __input_dict = input_dict
    task = tvm.auto_scheduler.SearchTask(func=workload, args=(), target="cuda")
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
        kernel_code = memopt.build_op(sch, args, "cuda", [], [], name="MyMatMul", global_kernel=True)
        cp = memopt.utils.CompileResult(None, kernel_code, scope.block_size, scope.grid_size, "MyMatMul", args)
        cp.append_host_call()
        cp.compile_and_load()
        print(cp.profile())


expr1 = """
mediate0[N0, N1, N2, N3] = input0[N3] where N0 in 1, N1 in 2160, N2 in 3840;
mediate1[N, HO, WO, F] +=! input1[N, -0 + KH + HO * 1, -0 + KW + WO * 1, C] * input2[KH, KW, C, F] where HO in 2160, WO in 3840;
mediate2[N0, N1, N2, N3] = mediate1[N0, N1, N2, N3] + mediate0[N0, N1, N2, N3];
output0[N0, N1, N2, N3] = mediate2[N0, N1, N2, N3].call(`max`, [const(0).cast(mediate2[N0, N1, N2, N3].dtype())]);
"""
input_dict1={ "input0" : { "dtype" : "float32", "shape" : [64]} ,  "input1" : { "dtype" : "float32", "shape" : [1, 2160, 3840, 21]} ,  "input2" : { "dtype" : "float32", "shape" : [1, 1, 21, 64]} }

# test(expr1 ,input_dict1, "conv_2160_3840_64_21.log")

expr2 = """
mediate0[N0, N1, N2, N3] = input0[N3] where N0 in 1, N1 in 2160, N2 in 3840;
mediate1[N, HO, WO, F] +=! input1[N, -0 + KH + HO * 1, -0 + KW + WO * 1, C] * input2[KH, KW, C, F] where HO in 2160, WO in 3840;
mediate2[N0, N1, N2, N3] = mediate1[N0, N1, N2, N3] + mediate0[N0, N1, N2, N3];
output0[N0, N1, N2, N3] = const(1).cast(mediate2[N0, N1, N2, N3].dtype()) / (const(1).cast(mediate2[N0, N1, N2, N3].dtype()) + (-mediate2[N0, N1, N2, N3]).call(`exp`));
"""

input_dict2={ "input0" : { "dtype" : "float32", "shape" : [1]} ,  "input1" : { "dtype" : "float32", "shape" : [1, 2160, 3840, 64]} ,  "input2" : { "dtype" : "float32", "shape" : [1, 1, 64, 1]} }
# test(expr2 ,input_dict2, "conv_2160_3840_64_21.log")

expr3 = " mediate0[N0, N1, N2, N3] = input0[N3] where N0 in 1, N1 in 1080, N2 in 1920;   mediate1[N, HO, WO, C] +=! input1[N, -1 + KH + HO * 1, -1 + KW + WO * 1, C].when([-1 + KH + HO * 1 >= 0, -1 + KH + HO * 1 < 1080, -1 + KW + WO * 1 >= 0, -1 + KW + WO * 1 < 1920], const(0.0).cast(input1[N, -1 + KH + HO * 1, -1 + KW + WO * 1, C].dtype())) * input2[KH, KW, C, 0] where HO in 1080, WO in 1920, KH in 3, KW in 3;  output0[N0, N1, N2, N3] = mediate1[N0, N1, N2, N3] + mediate0[N0, N1, N2, N3]; "
input_dict3={ "input0" : { "dtype" : "float32", "shape" : [16]} ,  "input1" : { "dtype" : "float32", "shape" : [1, 1080, 1920, 16]} ,  "input2" : { "dtype" : "float32", "shape" : [3, 3, 16, 1]} }
# test(expr3 ,input_dict3, "depthwiseconv_1080_1920_16_16.log")

expr4 = " mediate0[N0, N1, N2, N3] = input0[N3] where N0 in 1, N1 in 1080, N2 in 1920;   mediate1[N, HO, WO, F] +=! input1[N, -0 + KH + HO * 1, -0 + KW + WO * 1, C] * input2[KH, KW, C, F] where HO in 1080, WO in 1920;  mediate2[N0, N1, N2, N3] = mediate1[N0, N1, N2, N3] + mediate0[N0, N1, N2, N3]; output0[N0, N1, N2, N3] = mediate2[N0, N1, N2, N3].call(`max`, [const(0).cast(mediate2[N0, N1, N2, N3].dtype())]);"
input_dict4 = { "input0" : { "dtype" : "float32", "shape" : [16]} ,  "input1" : { "dtype" : "float32", "shape" : [1, 1080, 1920, 16]} ,  "input2" : { "dtype" : "float32", "shape" : [1, 1, 16, 16]} }
test(expr4 ,input_dict4, "conv_1080_1920_16_16.log")
