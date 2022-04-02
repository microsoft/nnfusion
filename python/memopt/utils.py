import tvm
from .modify_input_pass import modify_input_pass
from .modify_output_pass import modify_output_pass
from .scope import Scope, get_scope
import numpy as np

import ctypes
import os
import tempfile

_tvm_default_name = "default_function_kernel0"

def build_op(sch, args, target, sm_outputs=[], sm_inputs=[], name=_tvm_default_name, global_kernel=True):
    passes = [
        (0, modify_output_pass),
        (0, modify_input_pass),
    ]
    assert(isinstance(sm_outputs, (tuple, list)))
    assert(isinstance(sm_inputs, (tuple, list)))
    scope = get_scope()
    with tvm.transform.PassContext(config={"tir.add_lower_pass": passes}):
        scope.shared_mem_outputs = sm_outputs
        scope.shared_mem_inputs = sm_inputs
        mod = tvm.build(sch, args, target=target)

        src = mod.imported_modules[0].get_source()
        index = src.index(_tvm_default_name)
        if global_kernel:
            prefix = "__global__ void __launch_bounds__(%d) " % np.prod(scope.block_size)
        else:
            prefix = "__device__ void "
        src = prefix + name + src[index+len(_tvm_default_name):]
    return src

def ctypesCloseLibrary(lib):
    dlclose_func = ctypes.CDLL(None).dlclose
    dlclose_func.argtypes = [ctypes.c_void_p]
    dlclose_func.restype = ctypes.c_int

    dlclose_func(lib._handle)

def append_host_call(kernel_code, block, grid, num_params, name=_tvm_default_name, measure_time=True):
    args = ["args" + str(i) for i in range(num_params)]
    call_args = ", ".join(args)
    args = ["float* args" + str(i) for i in range(num_params)]
    def_args = ", ".join(args)
    block_str = "dim3({}, {}, {})".format(block[0], block[1], block[2])
    grid_str = "dim3({}, {}, {})".format(grid[0], grid[1], grid[2])
    if measure_time:
        template = """
extern "C" float function({}) {{
    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    {}<<<{}, {}>>>({});
    cudaEventRecord(stop, 0);
    if (cudaEventSynchronize(stop) != cudaSuccess)
        return -1;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}}
"""
    else:
        template = """
extern "C" void function({}) {{
    default_function_kernel0<<<{}, {}>>>({});
}}
"""
    header = "#include <cuda_runtime.h>\n"
    host_call = template.format(def_args, name, block_str, grid_str, call_args)
    return header + kernel_code + "\n" + host_call

def compile_and_load(kernel_code):
    src = tempfile.NamedTemporaryFile(mode='w', suffix=".cu")
    lib_name = src.name.replace(".cu", ".so")
    src.write(kernel_code)
    src.flush()
    ret = os.system("nvcc --compiler-options '-fPIC' --shared {} -lcuda -gencode=arch=compute_61,code=compute_61 -o {}".format(src.name, lib_name))
    assert(ret == 0)
    lib = ctypes.CDLL(lib_name)
    ret = os.system("rm {}".format(lib_name))
    assert(ret == 0)
    return lib

