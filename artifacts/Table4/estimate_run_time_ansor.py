import subprocess
import os.path as osp
import time
import onnx
import tvm
import tvm.relay.testing
from tvm import auto_scheduler, relay
import multiprocessing

sub_dirs = [
    ["mobilenet", (1, ), ("fp32", )],
    ["bert", (1, ), ("fp32", )],
]

def get_sub_dirs(prefix):
    results = []
    model_strings = []
    for model, bs, tp in sub_dirs:
        osp.join(prefix, model)
        for b in bs:
            for t in tp:
                suffix = str(b) + ("_fp16" if t == "fp16" else "")
                results.append(osp.join(prefix, model, suffix))
                model_strings.append(f"Model: {model} BS: {b}, dtype: {t}")
    return results, model_strings

def run_ansor_for_once(prefix):
    target = tvm.target.cuda(arch="sm_70")
    onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
    mod, params = relay.frontend.from_onnx(onnx_model)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    log_file = ".temp_log"

    num_trials = len(tasks) * 800

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0
    )
    start_time = time.time()
    with open("output", "w") as f:
        print(start_time, num_trials, file=f)
    tuner.tune(tune_option)

if __name__ == "__main__":
    prefix = "/sharepoint/e2e/"
    for sub_dir, model_string in zip(*get_sub_dirs(prefix)):
        p = multiprocessing.Process(target=run_ansor_for_once, args=[sub_dir])
        p.start()
        while True:
            if osp.exists("total_latency.tsv"):
                with open("total_latency.tsv") as f:
                    if len(f.read()) > 1:
                        end_time = time.time()
                        p.kill()
                        p.join()
                        break
            time.sleep(1)
        with open("output") as f:
            start_time, num_trails = f.read().split()
            start_time = float(start_time)
            num_trails = int(num_trails)
            time_cost = (end_time - start_time) * num_trails / 64
            print(f"Time used: {time_cost}s")
        subprocess.run(["rm", ".temp_log", "total_latency.tsv", "output"], check=True)
