import onnx
import numpy as np
import os.path as osp
import time
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

prefix = "/home/v-yiningshi/learn_tvm/testing/temp/bert"
target = tvm.target.cuda(arch="sm_70")
# import tensorflow.compat.v1 as tf
# pt_model = open(osp.join(prefix, "classifier.pb"), "rb")
# graph_def = tf.GraphDef()
# graph_def.ParseFromString(pt_model.read())
# mod, params = relay.frontend.from_tensorflow(graph_def, "NHWC")
# feed_dict = dict(np.load(osp.join(prefix, "inputs.npz"), allow_pickle=True))
# shape_dict = {key: value.shape for key, value in feed_dict.items()}
onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
mod, params = relay.frontend.from_onnx(onnx_model)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
log_file = osp.join(prefix, "ansor_tune.log")

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10, device=3)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=len(tasks) * 512,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

# run_tuning()

# Compile with the history best
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Create graph executor
dev = tvm.device(str(target), 3)
module = graph_executor.GraphModule(lib["default"](dev))

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, min_repeat_ms=500, end_to_end=False))
