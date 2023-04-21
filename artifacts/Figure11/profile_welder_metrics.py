import argparse
import ctypes
import os
import time
import json
import torch

cuda = ctypes.CDLL("libcudart.so")

def profile_welder(prefix):
    cur_dir = os.path.abspath(".")
    os.chdir(prefix)
    lib_path = "build/libnnfusion_naive_rt.so"
    if not os.path.exists(lib_path):
        return None
    lib = ctypes.CDLL(lib_path)
    lib.cuda_init()

    inputs, outputs = [], []
    with open("para_info.json") as f:
        para_info = json.load(f)

    for _, desc in para_info["input"].items():
        shape = desc["shape"]
        dtype = desc["id"]
        if "int64" in dtype:
            tensor = torch.ones(shape, dtype=torch.int64, device=0)
        elif "float16" in dtype:
            tensor = torch.randn(shape, dtype=torch.float16, device=0)
        elif "float32" in dtype:
            tensor = torch.randn(shape, dtype=torch.float32, device=0)
        else:
            raise NotImplementedError(dtype)
        inputs.append(tensor)
    for _, desc in para_info["output"].items():
        shape = desc["shape"]
        dtype = desc["id"]
        if "int64" in dtype:
            tensor = torch.empty(shape, dtype=torch.int64, device=0)
        elif "float16" in dtype:
            tensor = torch.empty(shape, dtype=torch.float16, device=0)
        elif "float32" in dtype:
            tensor = torch.empty(shape, dtype=torch.float32, device=0)
        else:
            raise NotImplementedError(dtype)
        outputs.append(tensor)

    args = [ctypes.c_void_p(tensor.data_ptr()) for tensor in inputs + outputs]
    def get_runtime():
        tic = time.monotonic_ns()
        lib.kernel_entry(*args)
        cuda.cudaDeviceSynchronize()
        return (time.monotonic_ns() - tic) / 1e6

    st = time.time()
    while time.time() - st < 1.0:
        get_runtime() # warmup

    cuda.cudaProfilerStart()
    get_runtime()
    cuda.cudaProfilerStop()
    os.chdir(cur_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    args = parser.parse_args()

    profile_welder(os.path.join(args.prefix, "nnfusion_rt/cuda_codegen/"))
