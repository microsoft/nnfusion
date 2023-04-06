import argparse
import ctypes
import time

import numpy as np
import torch
from model.pytorch import *

torch.backends.cudnn.benchmark = True
cuda = ctypes.CDLL("libcudart.so")

def run_nimble(model, inputs):
    cu_inputs = []
    for item in inputs:
        cu_inputs.append(item.cuda() if isinstance(item, torch.Tensor) else item)
    with torch.no_grad():
        nimble_model = torch.cuda.Nimble(model)
        nimble_model.prepare(cu_inputs, training=False, use_multi_stream=True)

    def get_runtime():
        tic = time.time()
        _ = nimble_model(*cu_inputs)
        cuda.cudaDeviceSynchronize()
        return (time.time() - tic) * 1000
    with torch.no_grad():
        _ = [get_runtime() for i in range(50)] # warmup
        times = [get_runtime() for i in range(100)]
    print("mean: {}ms min: {}ms max: {}ms".format(np.mean(times), np.min(times), np.max(times)))
    # cuda.cudaProfilerStart()
    # get_runtime()
    # cuda.cudaProfilerStop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()
    assert (args.model in globals()), "Model {} not found.".format(args.model)

    torch.random.manual_seed(0)
    model, inputs = globals()[args.model](args.bs)
    model = model.cuda()
    if args.fp16:
        model = model.half()
        inputs = [x.half() if torch.is_floating_point(x) else x for x in inputs]
    run_nimble(model, inputs)
