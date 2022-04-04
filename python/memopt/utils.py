from .graph import OutputNode, PlaceHolderNode
import tvm
from .modify_input_pass import modify_input_pass
from .modify_output_pass import modify_output_pass
from .debug_pass import get_kernel_info_pass
from .scope import Scope, get_scope
from .schedule_rewrite import CodeGenerator
import numpy as np
import regex as re

import ctypes
import os
import tempfile

_tvm_default_name = "default_function_kernel0"
_type_map = {"float32" : "float"}

def build_op(sch, args, target, sm_outputs=[], sm_inputs=[], name=_tvm_default_name, global_kernel=True):
    passes = [
        (0, modify_output_pass),
        (0, modify_input_pass),
        (4, get_kernel_info_pass),
    ]
    assert(isinstance(sm_outputs, (tuple, list)))
    assert(isinstance(sm_inputs, (tuple, list)))
    scope = get_scope()
    # from .debug import debug
    # debug({**globals(), **locals()})
    func_args = ", ".join(["{}* __restrict__ {}".format(_type_map[var.dtype], var.name) for var in args])
    with tvm.transform.PassContext(config={"tir.add_lower_pass": passes}):
        scope.shared_mem_outputs = sm_outputs
        scope.shared_mem_inputs = sm_inputs
        mod = tvm.build(sch, args, target=target)

        src = mod.imported_modules[0].get_source()
        index = src.index("{")
        if global_kernel:
            prefix = "__global__ void __launch_bounds__(%d) " % np.prod(scope.block_size)
        else:
            prefix = "__device__ void "
            func_args += ", char* shared"
        src = prefix + name + "({}) ".format(func_args) + src[index:]
        # removing shared memory allocation
        for var in scope.shared_mem_inputs:
            s_var = var+"_shared"
            src = re.sub(r"__shared__ (\w+) {}\[\d+\];".format(s_var), r"\1* {} = {};".format(s_var, var), src, 1)
        if not global_kernel:
            for var, offset in scope.interal_shared_memory_offset.items():
                src = re.sub(r"__shared__ (\w+) {}\[\d+\];".format(var), r"\1* {} = (\1*)(shared+{});".format(var, offset), src, 1)
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
    host_call = template.format(def_args, name, grid_str, block_str, call_args)
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

def compose_global_kernel(topo_order, configs, target, name):
    # check inputs and outputs
    subgraph_inputs_idx_map = {}
    subgraph_outputs_idx_map = {}
    for op in topo_order:
        if isinstance(op, PlaceHolderNode):
            idx = len(subgraph_inputs_idx_map)
            subgraph_inputs_idx_map[op] = idx
        elif isinstance(op, OutputNode):
            idx = len(subgraph_outputs_idx_map)
            subgraph_outputs_idx_map[op] = idx
    # -------------------------------------------------
    cgen = CodeGenerator()
    idx = 0
    total_interal_shared_memory = 0
    for op in topo_order:
        if isinstance(op, PlaceHolderNode):
            continue
        config = configs[op]
        sch = tvm.te.create_schedule(op.args[-1].op)
        sch = cgen.rewrite_schedule(sch, config, True, True, target_stage="C")
        with Scope(sch) as scope:
            shared_inputs = []
            shared_outputs = []
            for i, input in enumerate(op.inputs):
                if not isinstance(input.src_node, PlaceHolderNode):
                    shared_inputs.append(op.args[i].name)
            for output in op.outputs:
                if not isinstance(input.dst_node, OutputNode):
                    shared_outputs.append(op.args[-1].name)
                    break
            kernel_code = build_op(sch, op.args, target, shared_outputs, shared_inputs, name=name+"_kernel_"+str(idx), global_kernel=False)
            idx += 1
            total_interal_shared_memory = max(total_interal_shared_memory, scope.total_interal_shared_memory)
    return None
