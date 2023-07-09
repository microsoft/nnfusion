from typing import List

import numpy as np
import regex as re
import tvm

from .schedule.scheduler_base import SchedulerBase

TVM_DEFAULT_NAME = "default_function_kernel0"
_type_map = {"float32": "float", "float16": "half", "float64": "double", "int64": "int64_t", "int32": "int", "bool": "int8_t", "int8": "int8_t", "int16": "int16_t"}
_type_bytes = {"float": 4, "double": 8, "half": 2, "int16" : 2, "int": 4, "int64_t": 8, "bool": 1, "int8_t": 1, "signed char": 1}

def get_valid_name(var):
    if var.name.find(".") >= 0:
        name = var.name[:var.name.index(".")]
    else:
        name = var.name
    return name if var.value_index == 0 else name + "_" + str(var.value_index)

def get_block_flatten_code(block_size):
    if block_size[1] == 1 and block_size[2] == 1:
        return ""
    elif block_size[2] == 1:
        return "  int __flatten_tid = threadIdx.x;\n  const dim3 threadIdx(__flatten_tid % {}, __flatten_tid / {}, 0);\n"\
            .format(block_size[0], block_size[0])
    else: # not possible in our schedule
        raise NotImplementedError()

_c_op_map = {tvm.tir.FloorMod : '%', tvm.tir.FloorDiv : "/", tvm.tir.Add : "+", tvm.tir.Sub : "-", tvm.tir.Mul: "*"}

def _lower_C_simple(expr : tvm.tir.PrimExpr) -> str:
    if isinstance(expr, tvm.tir.expr.BinaryOpExpr):
        left = _lower_C_simple(expr.a)
        right = _lower_C_simple(expr.b)
        if type(expr) in _c_op_map:
            return "({} {} {})".format(left, _c_op_map[type(expr)], right)
        else:
            raise NotImplementedError(expr)
    elif isinstance(expr, tvm.tir.expr.Var):
        assert expr.name == "block_idx"
        return "__bid"
    elif isinstance(expr, tvm.tir.expr.ConstExpr):
        return str(expr.value)
    else:
        raise NotImplementedError(expr)

def get_block_reorder_code(block_reoder_expr: tvm.tir.PrimExpr) -> str:
    return "  int __bid = blockIdx.x;\n  const dim3 blockIdx({}, 0, 0);\n"\
        .format(_lower_C_simple(block_reoder_expr))

def tvm_build(sch: SchedulerBase, target: tvm.target.Target, name: str = TVM_DEFAULT_NAME,
              global_kernel=True, flatten_block=True, reuse_disabled_inputs=[]) -> str:
    func_args = ", ".join(["{}* __restrict__ {}".format(_type_map[var.dtype], get_valid_name(var)) for var in sch.args])

    def is_independent_alloc(tensor_name):
        if tensor_name.endswith(".wmma.accumulator.shared") and len(sch.shared_outputs) > 0:
            return True
        return tensor_name in  [x.name + ".shared" for x in sch.shared_inputs]

    def is_reuse_disabled(tensor_name):
        return tensor_name in [x.name + ".shared" for x in reuse_disabled_inputs]

    tvm._ffi.register_func("memopt.is_independent_alloc", is_independent_alloc, override=True)
    tvm._ffi.register_func("memopt.is_reuse_disabled", is_reuse_disabled, override=True)

    old_entry = tvm.get_global_func("tvm_callback_cuda_compile")
    tvm.register_func("tvm_callback_cuda_compile", override=True)(lambda x:"")
    src = sch.build(target)
    tvm.register_func("tvm_callback_cuda_compile", override=True)(old_entry)
    tvm._ffi.register_func("memopt.is_independent_alloc", lambda x:False, override=True)
    tvm._ffi.register_func("memopt.is_reuse_disabled", lambda x:False, override=True)

    exteral_shared_memroy_size = {}
    total_internal_shared_memory = 0
    for idx in sch.shared_outputs:
        tile_shape = sch.config.block
        dtype_bytes = (tvm.DataType(sch.args[idx].dtype).bits + 7) // 8
        if idx in sch.config.output_strides:
            strides = sch.config.output_strides[idx].compute_strides_from_shape(tile_shape)
            exteral_shared_memroy_size[idx] = tile_shape[0] * strides[0] * dtype_bytes
        else:
            exteral_shared_memroy_size[idx] = int(np.prod(tile_shape)) * dtype_bytes

    index = src.rindex(TVM_DEFAULT_NAME)
    index = src.index("{", index)
    if flatten_block:
        flat_block_code = get_block_flatten_code(sch.block_size)
        sch.block_size = [int(np.prod(sch.block_size)), 1, 1]
        src = src[:index+2] + flat_block_code + src[index+2:]
    if sch.config.block_order is not None:
        block_reorder_code = get_block_reorder_code(sch.config.block_order)
        src = src[:index+2] + block_reorder_code + src[index+2:]
    if global_kernel:
        prefix = "__global__ void __launch_bounds__(%d) " % np.prod(sch.block_size)
    else:
        prefix = "__device__ void "
        func_args += ", char* shared"
    src = prefix + name + "({}) ".format(func_args) + src[index:]
    # removing shared memory allocation
    # check wmma accumulator shared
    if len(sch.shared_outputs) > 0:
        reuse_output_name = get_valid_name(sch.args[sch.shared_outputs[0]])
        src = re.sub(r"__shared__ (\w+) (\w+wmma_accumulator_shared)\[\d+\];", r"\1* \2 = {};".format(reuse_output_name), src, 1)
    for tensor in sch.shared_inputs:
        shared_var_name = tensor.name + "_shared"
        matched = re.findall(r"__shared__ ((?:signed |unsigned )?\w+) {}\[(\d+)\];".format(shared_var_name), src)
        assert len(matched) == 1
        dtype, size = matched[0]
        exteral_shared_memroy_size[tensor] = int(size) * _type_bytes[dtype]
        src = re.sub(r"__shared__ ((?:signed |unsigned )?\w+) {}\[\d+\];".format(shared_var_name), r"\1* {} = {};".format(shared_var_name, tensor.name), src, 1)
    if not global_kernel:
        pattern = r"__shared__ ((?:signed |unsigned )?\w+) (\w+)\[(\d+)\];"
        offset = 0
        for dtype, var, size in re.findall(pattern, src):
            src = re.sub(r"__shared__ ((?:signed |unsigned )?\w+) {}\[\d+\];".format(var), r"\1* {} = (\1*)(shared+{});".format(var, offset), src, 1)
            buffer_len = int(size) * _type_bytes[dtype]
            buffer_len = (buffer_len + 31) // 32 * 32
            offset += buffer_len
        total_internal_shared_memory = offset
    if global_kernel:
        pattern = r"__shared__ ((?:signed |unsigned )?\w+) (\w+)\[(\d+)\];"
        for dtype, var, size in re.findall(pattern, src):
            buffer_len = int(size) * _type_bytes[dtype]
            buffer_len = (buffer_len + 31) // 32 * 32
            src = re.sub(r"__shared__ ((?:signed |unsigned )?\w+) {}\[\d+\];".format(var), r"__shared__ \1 {}[{}];".format(var, buffer_len // _type_bytes[dtype]), src, 1)
    return src, exteral_shared_memroy_size, total_internal_shared_memory
