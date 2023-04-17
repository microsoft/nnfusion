import argparse
import ctypes
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

cuda = ctypes.CDLL("libcudart.so")

def get_workspace(onnx_model_path):
    print("Preparing workspace ...")
    np.random.seed(0)
    onnx_model = onnx.load(onnx_model_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    inputs, outputs = [], []
    for value in ort_session.get_inputs():
        if value.type == 'tensor(int64)':
            tensor = torch.ones(value.shape, dtype=torch.int64, device=0)
        elif value.type == 'tensor(float16)':
            tensor = torch.randn(value.shape, dtype=torch.float16, device=0)
        elif value.type == 'tensor(float)':
            tensor = torch.randn(value.shape, dtype=torch.float32, device=0)
        else:
            raise NotImplementedError(value.type)
        inputs.append(tensor)
    for value in ort_session.get_outputs():
        if value.type == 'tensor(int64)':
            tensor = torch.empty(value.shape, dtype=torch.int64, device=0)
        elif value.type == 'tensor(float16)':
            tensor = torch.empty(value.shape, dtype=torch.float16, device=0)
        elif value.type == 'tensor(float)':
            tensor = torch.empty(value.shape, dtype=torch.float32, device=0)
        else:
            raise NotImplementedError(value.type)
        outputs.append(tensor)

    return inputs, outputs

def profile_welder(prefix, inputs, outputs):
    lib_path = os.path.join(prefix, "build/libnnfusion_naive_rt.so")
    if not os.path.exists(lib_path):
        return None
    lib = ctypes.CDLL(lib_path)
    cur_dir = os.path.abspath(".")
    os.chdir(prefix)
    lib.cuda_init()
    args = [ctypes.c_void_p(tensor.data_ptr()) for tensor in inputs + outputs]
    def get_runtime():
        tic = time.monotonic_ns()
        lib.kernel_entry(*args)
        cuda.cudaDeviceSynchronize()
        return (time.monotonic_ns() - tic) / 1e6

    print("Warming up ...")
    st = time.time()
    while time.time() - st < 1.0:
        get_runtime() # warmup

    print("Running 100 iterations ...")
    times = [get_runtime() for _ in range(100)]
    print(f"avg: {np.mean(times)} ms")
    print(f"min: {np.min(times)} ms")
    print(f"max: {np.max(times)} ms")
    os.chdir(cur_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    args = parser.parse_args()

    prefix = args.prefix
    inputs, outputs = get_workspace(os.path.join(prefix, "model.onnx"))

    profile_welder(os.path.join(prefix, "nnfusion_rt/cuda_codegen/"), inputs, outputs)
