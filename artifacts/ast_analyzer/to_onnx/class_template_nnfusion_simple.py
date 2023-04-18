from ast_analyzer.grad.cfg import backward, forward
import torch
import os
import sys
from time import time
from torch.utils.cpp_extension import load


def load_model(model_path):
    assert(model_path.endswith('.onnx'))
    workdir = os.path.abspath(model_path[:-5])
    rt_dir = os.path.join(workdir, "nnfusion_rt/cuda_codegen")
    print("[rt_dir]", rt_dir, "name", workdir.split("/")[-1])
    old_dir = os.getcwd()
    os.chdir(rt_dir)
    loaded = load(
        name=workdir.split("/")[-1][:-8],
        sources=["nnfusion_rt.cu"],
        extra_cflags=["-Wall", "-Wextra", "-march=native", "-g", "-O2"],
        extra_cuda_cflags=["-gencode arch=compute_70,code=sm_70", "-O2",  "--expt-relaxed-constexpr -D__HALF_COMPARE_EX__"],
    )
    loaded.cuda_init()
    os.chdir(old_dir)
    return loaded.kernel_entry, loaded.cuda_free

forward_call, free_func = load_model("tmp/^^MODELNAME-forward.onnx")
    
def run(^^INPUTS):
    # print("[run pybind simple]")
    return forward_call(^^INPUTS)
    # for i, x in enumerate([^^INPUTS]):
    #     if isinstance(x, int):
    #         print("input", i, x)
    #     else:
    #         with open(f"bin/^^MODELNAME_input_ref_{i}.bin", "wb") as f:
    #             x.cpu().detach().numpy().tofile(f)
    #         x.cuda()
            
    # o = forward_call(^^INPUTS)

    # for i, x in enumerate(o):
    #     if isinstance(x, int):
    #         print("output", i, x)
    #     else:
    #         with open(f"bin/^^MODELNAME_output_ref_{i}.bin", "wb") as f:
    #             x.cpu().detach().numpy().tofile(f)
    #         x.cuda()
    # return o
