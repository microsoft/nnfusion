import argparse
import os.path as osp
import numpy as np
import onnx
import tvm
import tvm.relay.testing
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor
import ctypes
cuda = ctypes.CDLL("libcudart.so")

def run_ansor(prefix, device, skip_tuning):
    target = tvm.target.cuda(arch="sm_70")
    onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
    mod, params = relay.frontend.from_onnx(onnx_model)
    log_file = osp.join(prefix, "ansor_tune.log")
    # Compile with the history best
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)

    # Create graph executor
    dev = tvm.device(str(target), device)
    module = graph_executor.GraphModule(lib["default"](dev))
    module.benchmark(dev, min_repeat_ms=500, end_to_end=False)
    cuda.cudaProfilerStart()
    print(module.benchmark(dev, repeat=1, number=2, end_to_end=False))
    cuda.cudaProfilerStop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="temp")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--skip', action="store_true")
    args = parser.parse_args()
    run_ansor(args.prefix, args.device, args.skip)
