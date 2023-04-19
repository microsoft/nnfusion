import argparse
import ctypes
import time

import numpy as np
import torch
import torch_blade
from model.pytorch import *

cuda = ctypes.CDLL("libcudart.so")

def run_blade(model, inputs):
    cu_inputs = []
    for item in inputs:
        cu_inputs.append(item.cuda() if isinstance(item, torch.Tensor) else item)

    torch_config = torch_blade.config.Config()
    torch_config.enable_mlir_amp = False # disable mix-precision
    model = torch.jit.trace(model.cuda().eval(), cu_inputs, strict=False)
    # model = torch.jit.trace(model, inputs, strict=False).cuda().eval()

    torch._C._jit_pass_inline(model._c.forward.graph)
    torch._C._jit_pass_remove_dropout(model._c)

    with torch.no_grad(), torch_config:
        # BladeDISC torch_blade optimize will return an optimized TorchScript
        model = torch_blade.optimize(model, allow_tracing=True, model_inputs=tuple(cu_inputs))

    def get_runtime():
        tic = time.time()
        _ = model(*cu_inputs)
        cuda.cudaDeviceSynchronize()
        return (time.time() - tic) * 1000
    with torch.no_grad():
        print("Warming up ...")
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime() # warmup
        [get_runtime() for i in range(50)] # extra warmup
        times = [get_runtime() for i in range(100)]
        print(f"avg: {np.mean(times)} ms")
        print(f"min: {np.min(times)} ms")
        print(f"max: {np.max(times)} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()
    assert (args.model in globals()), "Model {} not found.".format(args.model)

    if args.model == "bert":
        args.model = "bert_v0" # use the huggingface version seem to be better.

    torch.random.manual_seed(0)
    model, inputs = globals()[args.model](args.bs)

    if args.fp16:
        model = model.half()
        inputs = [x.half() if torch.is_floating_point(x) else x for x in inputs]
    run_blade(model, inputs)
