from ast import Not
from .graph import OutputNode, PlaceHolderNode
import tvm
from .modify_input_pass import modify_input_pass
from .modify_output_pass import modify_output_pass
from .debug_pass import debug_pass, get_kernel_info_pass
from .scope import Scope, get_scope
from .schedule_rewrite import CodeGenerator
from .bestfit import BestFit
import numpy as np
import regex as re
import io

import ctypes
import os
import tempfile

_tvm_default_name = "default_function_kernel0"
_type_map = {"float32" : "float"}

def get_valid_name(var):
    if var.name.find(".") >= 0:
        name = var.name[:var.name.index(".")]
    else:
        name = var.name
    return name if var.value_index == 0 else name + str(var.value_index)

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
    func_args = ", ".join(["{}* __restrict__ {}".format(_type_map[var.dtype], get_valid_name(var)) for var in args])
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
    if (cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if (cudaEventSynchronize(stop) != cudaSuccess) return -1;
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

def can_free(node, out_id, done_ops):
    for edge in node.outputs:
        if edge.src_id == out_id and edge.dst_node not in done_ops:
            return False
    return True

def compose_global_kernel(topo_order, configs, target, name):
    # check inputs and outputs
    kernel_args_name_map = {}
    num_inputs, num_outputs = 0, 0
    for op in topo_order:
        if isinstance(op, (PlaceHolderNode, OutputNode)):
            continue
        else:
            for edge in op.inputs:
                if isinstance(edge.src_node, PlaceHolderNode):
                    kernel_args_name_map[op.args[edge.dst_id]] = "input"+str(num_inputs)
                    num_inputs += 1
            for edge in op.outputs:
                if isinstance(edge.dst_node, OutputNode):
                    kernel_args_name_map[op.args[edge.src_id+len(op.inputs)]] = "output"+str(num_outputs)
                    num_outputs += 1

    # -------------------------------------------------
    cgen = CodeGenerator()
    allocator = BestFit()
    block_map = {}
    device_func_uid = 0
    done_op = set()
    statements = []
    block_size, grid_size = None, None
    code = io.StringIO()
    for op in topo_order:
        done_op.add(op)
        if isinstance(op, (PlaceHolderNode, OutputNode)):
            continue
        config = configs[op]
        sch = tvm.te.create_schedule(op.args[-1].op)
        shared_inputs = []
        shared_outputs = []
        shared_inputs_idx = []
        for input in op.inputs:
            if not isinstance(input.src_node, PlaceHolderNode):
                shared_inputs.append(op.args[input.dst_id].name)
                shared_inputs_idx.append(input.dst_id)
        for output in op.outputs:
            if not isinstance(output.dst_node, OutputNode):
                shared_outputs.append(len(op.inputs)+output.src_id)
            shared_outputs = list(set(shared_outputs)) # unique
        sch = cgen.rewrite_schedule(sch, config, True, True, target_stage=op.args[-1].name, tile_blacklist=shared_inputs)
        with Scope(sch) as scope:
            func_name = name+"_kernel_"+str(device_func_uid)
            kernel_code = build_op(sch, op.args, target, shared_outputs, shared_inputs, name=func_name, global_kernel=False)
            if block_size is None:
                block_size = scope.block_size
                grid_size = scope.grid_size
            else:
                assert(block_size == scope.block_size)
                assert(grid_size == scope.grid_size)
            code.write(kernel_code)
            # from .debug import debug
            # debug({**globals(), **locals()})
            block_map[op] = {}
            for idx in shared_outputs:
                num_bytes = scope.exteral_shared_memroy_size[idx]
                block = allocator.malloc(num_bytes)
                block_map[op][idx-len(op.inputs)] = block
            internal_shared_mem = allocator.malloc(scope.total_interal_shared_memory)
            for idx, var_name in zip(shared_inputs_idx, shared_inputs):
                num_bytes = scope.exteral_shared_memroy_size[var_name]
                src_node = op.inputs[idx].src_node
                src_id = op.inputs[idx].src_id
                if can_free(src_node, src_id, done_op):
                    allocator.free(block_map[src_node][src_id])

            allocator.free(internal_shared_mem)
            print(allocator.limit)
            arg_list = []
            for idx in range(len(op.inputs)):
                if idx in shared_inputs_idx:
                    src_node = op.inputs[idx].src_node
                    src_id = op.inputs[idx].src_id
                    dtype = _type_map[src_node.args[src_id+len(src_node.inputs)].dtype]
                    arg_list.append("({}*)(shared+{})".format(dtype, block_map[src_node][src_id].start))
                else:
                    arg_list.append(kernel_args_name_map[op.args[idx]])
            for idx in range(len(op.inputs), len(op.args)):
                if idx in shared_outputs:
                    dtype = _type_map[op.args[idx].dtype]
                    arg_list.append("({}*)(shared+{})".format(dtype, block_map[op][idx-len(op.inputs)].start))
                else:
                    arg_list.append(kernel_args_name_map[op.args[idx]])
            arg_list.append("shared+{}".format(internal_shared_mem.start))
            call_str = func_name + "(" + ", ".join(arg_list) + ");"
            statements.append(call_str)
            device_func_uid += 1

    statements.insert(0, "__shared__ char shared[{}];".format(allocator.limit))
    for stmt in statements:
        print(stmt)
    kernel_args_dtype_map = {v : _type_map[k.dtype] for k, v in kernel_args_name_map.items()}
    kernel_args_name = ["{}* {}".format(kernel_args_dtype_map[arg], arg)
        for arg in sorted(kernel_args_name_map.values())]
    prefix = "__global__ void __launch_bounds__({}) {}({})".format(
        np.prod(scope.block_size), name,
        ", ".join(kernel_args_name)
    )
    print(prefix)
    code.write(prefix)
    code.write(" {\n")
    for stmt in statements:
        code.write("  "+stmt+"\n")
    code.write("}\n")

    args = sorted(kernel_args_name_map, key = lambda k: kernel_args_name_map[k])
    return code.getvalue(), block_size, grid_size, args
