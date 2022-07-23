from .graph import OutputNode, PlaceHolderNode, find_topo_sort
from .scope import Scope
from .schedule_rewrite import CodeGenerator
from .bestfit import BestFit
from .reference import get_ref_tensor
from .tvm_build import tvm_build, _type_map

from concurrent.futures import ThreadPoolExecutor
import tvm
import numpy as np
import io
import ctypes
import os
import subprocess
import tempfile

class CompileResult:
    def __init__(self, config, code, block_size, grid_size, name, args) -> None:
        self.config = config
        self.code = code
        self.block_size = block_size
        self.grid_size = grid_size
        self.args = args
        self.name = name
        self.host_code = None
        self.lib = None
        self.latency = None
        self.origin = self

    def set_io_desc(self, input_desc, output_desc):
        self.input_desc = input_desc
        self.output_desc = output_desc

    def append_host_call(self):
        num_params = len(self.args)
        args = ["args" + str(i) for i in range(num_params)]
        call_args = ", ".join(args)
        args = ["{}* args{}".format(_type_map[self.args[i].dtype], i) for i in range(num_params)]
        def_args = ", ".join(args)
        block_str = "dim3({}, {}, {})".format(self.block_size[0], self.block_size[1], self.block_size[2])
        grid_str = "dim3({}, {}, {})".format(self.grid_size[0], self.grid_size[1], self.grid_size[2])
        call_str = "{}<<<{}, {}>>>({})".format(self.name, grid_str, block_str, call_args)
        host_funcs = \
"""
extern "C" void call({}) {{
    {};
}}
""".format(def_args, call_str)

        host_funcs += \
"""
extern "C" float profile({}) {{
    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    {};
    if (cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if (cudaEventSynchronize(stop) != cudaSuccess) return -1;
    cudaEventElapsedTime(&ms, start, stop);
    int repeats = int(ceil(100.0 / ms));
    cudaEventRecord(start, 0);
    for (int _ = 0; _ < repeats; _++)
        {};
    if (cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if (cudaEventSynchronize(stop) != cudaSuccess) return -1;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeats;
}}
""".format(def_args, call_str, call_str)
        header = \
"""
#include <cuda_runtime.h>
#include <math.h>
"""
        self.host_code = header + self.code + "\n" + host_funcs
        return self.host_code

    def compile_and_load(self):
        assert self.host_code
        src = tempfile.NamedTemporaryFile(mode='w', suffix=".cu")
        lib_name = src.name.replace(".cu", ".so")
        src.write(self.host_code)
        src.flush()
        compute_version = "".join(tvm.contrib.nvcc.get_target_compute_version().split("."))
        ret = subprocess.run(
            ["nvcc", "--compiler-options", "'-fPIC'", "--shared", src.name, "-lcuda",
            "-gencode=arch=compute_{},code=compute_{}".format(compute_version, compute_version),
            "-o", lib_name])
        if ret.returncode != 0:
            return None
        # ret = os.system("nvcc --compiler-options '-fPIC' --shared {} -lcuda -gencode=arch=compute_61,code=compute_61 -o {}".format(src.name, lib_name))
        self.lib = ctypes.CDLL(lib_name)
        self.lib.profile.restype = ctypes.c_float
        subprocess.run(["rm", lib_name], check=True)
        return self.lib

    def profile(self, device="cuda:0"):
        assert self.lib
        import torch
        torch.cuda.set_device(device)
        torch_arrs = []
        for arg in self.args:
            shape = list(map(int, arg.shape))
            arr = get_ref_tensor(shape, device, arg.dtype)
            torch_arrs.append(arr)
        latency = self.lib.profile(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_arrs])
        if latency < 0:
            return 10000
        self.latency = latency
        return latency

    def get_example_outputs(self, device="cuda:0", seed=0):
        import torch
        torch.cuda.set_device(device)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch_arrs = []
        for arg in self.args:
            shape = list(map(int, arg.shape))
            arr = get_ref_tensor(shape, device, arg.dtype)
            torch_arrs.append(arr)
        self.lib.call(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_arrs])
        torch.cuda.synchronize(device)
        outputs = []
        for i, arg in enumerate(self.args):
            if arg.name.startswith("output"):
                outputs.append(torch_arrs[i].cpu().numpy())
        return outputs

    def close_lib(self):
        if self.lib is None:
            return
        dlclose_func = ctypes.CDLL(None).dlclose
        dlclose_func.argtypes = [ctypes.c_void_p]
        dlclose_func.restype = ctypes.c_int
        dlclose_func(self.lib._handle)
        self.lib = None

    def __del__(self):
        self.close_lib()

def compile_and_load_parallel(cpresults):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        libs = executor.map(CompileResult.compile_and_load, cpresults)
    return list(libs)

def can_free(node, out_id, done_ops):
    for edge in node.outputs:
        if edge.src_id == out_id and edge.dst_node not in done_ops:
            return False
    return True

def compose_global_kernel(output_nodes, configs, target, name) -> CompileResult:
    # check inputs and outputs
    topo_order = find_topo_sort(output_nodes)
    tensor_name_map = {}
    num_inputs, num_outputs = 0, 0
    for op in topo_order:
        if isinstance(op, PlaceHolderNode):
            tensor_name_map[op] = "input" + str(num_inputs)
            num_inputs += 1
        elif isinstance(op, OutputNode):
            tensor_name_map[op] = "output" + str(num_outputs)
            num_outputs += 1
    kernel_args_name_map = {}
    for op in topo_order:
        if isinstance(op, (PlaceHolderNode, OutputNode)):
            continue
        else:
            for edge in op.inputs:
                if isinstance(edge.src_node, PlaceHolderNode):
                    kernel_args_name_map[op.args[edge.dst_id]] = tensor_name_map[edge.src_node]
            for edge in op.outputs:
                if isinstance(edge.dst_node, OutputNode):
                    kernel_args_name_map[op.args[edge.src_id+len(op.inputs)]] = tensor_name_map[edge.dst_node]

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
        sch = op.create_schedule()
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
            shared_outputs = list(set(shared_outputs)) #

        sch = cgen.rewrite_schedule(sch, config, shared_inputs=shared_inputs)

        with Scope(sch) as scope:
            func_name = "_".join([name, str(device_func_uid), op.name])
            kernel_code = tvm_build(sch, op.args, target, shared_outputs, shared_inputs, name=func_name, global_kernel=False,
                block_reorder=config["block_reorder"] if "block_reorder" in config else None)
            if block_size is None:
                block_size = scope.block_size
                grid_size = scope.grid_size
            else:
                assert(block_size == scope.block_size)
                assert(grid_size == scope.grid_size)
            code.write(kernel_code)
            block_map[op] = {}
            internal_shared_mem = allocator.malloc(scope.total_interal_shared_memory)
            for idx, var_name in zip(shared_inputs_idx, shared_inputs):
                src_node = op.inputs[idx].src_node
                src_id = op.inputs[idx].src_id
                if can_free(src_node, src_id, done_op):
                    allocator.free(block_map[src_node][src_id])
            allocator.free(internal_shared_mem)
            for idx in shared_outputs:
                num_bytes = scope.exteral_shared_memroy_size[idx]
                block = allocator.malloc(num_bytes)
                block_map[op][idx-len(op.inputs)] = block

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

    if allocator.limit > 0:
        statements.insert(0, "__shared__ char shared[{}];".format(allocator.limit))
    else:
        statements.insert(0, "char* shared = NULL;".format(allocator.limit))
    kernel_args_dtype_map = {v : _type_map[k.dtype] for k, v in kernel_args_name_map.items()}
    kernel_args_name = []
    for i in range(num_inputs):
        kernel_args_name.append("input"+str(i))
    for i in range(num_outputs):
        kernel_args_name.append("output"+str(i))
    kernel_args = ["{}* {}".format(kernel_args_dtype_map[arg], arg) for arg in kernel_args_name]
    prefix = "__global__ void __launch_bounds__({}) {}({})".format(
        np.prod(scope.block_size), name,
        ", ".join(kernel_args)
    )
    code.write(prefix)
    code.write(" {\n")
    for stmt in statements:
        code.write("  "+stmt+"\n")
    code.write("}\n")

    # fused kernel args
    args = []
    for arg_name in kernel_args_name:
        for k, v in kernel_args_name_map.items():
            if v == arg_name:
                args.append(k)
                break
    return CompileResult(configs, code.getvalue(), block_size, grid_size, name, args)

