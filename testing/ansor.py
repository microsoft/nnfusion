import onnx
import os.path as osp
import time
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
import argparse

def run_ansor(prefix, device, skip_tuning):
    target = tvm.target.cuda(arch="sm_70")
    onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
    mod, params = relay.frontend.from_onnx(onnx_model)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    log_file = osp.join(prefix, "ansor_tune.log")

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    num_trials = len(tasks) * 800
    if osp.exists(log_file):
        with open(log_file, "r") as f:
            cur_records = len(f.readlines())
        num_trials -= cur_records
    if num_trials > 0 and not skip_tuning:
        print("Begin tuning...")
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10, device=device)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=num_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        tuner.tune(tune_option)

    # Compile with the history best
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)

    # Create graph executor
    dev = tvm.device(str(target), device)
    module = graph_executor.GraphModule(lib["default"](dev))

    # Evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, min_repeat_ms=500, end_to_end=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="temp")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--skip', action="store_true")
    args = parser.parse_args()
    start_time = time.time()
    run_ansor(args.prefix, args.device, args.skip)
