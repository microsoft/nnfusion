from typing import List

import numpy as np
import regex as re
import tvm

from .IRpass import *

TVM_DEFAULT_NAME = "default_function_kernel0"
_type_map = {"float32": "float", "float16": "half", "float64": "double", "int64": "int64_t", "int32": "int", "bool": "bool", "int8": "int8_t"}
_type_bytes = {"float": 4, "double": 8, "half": 2, "int": 4, "int64_t": 8, "bool": 1, "int8_t": 1, "signed char": 1}

def get_valid_name(var):
    if var.name.find(".") >= 0:
        name = var.name[:var.name.index(".")]
    else:
        name = var.name
    return name if var.value_index == 0 else name + str(var.value_index)

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

def tvm_build(sch: tvm.te.Schedule, args: List[tvm.te.Tensor], target: tvm.target.Target,
              sm_outputs: List[int] = [], sm_inputs: List[tvm.te.Tensor] = [], name: str = TVM_DEFAULT_NAME,
              global_kernel=True, block_reorder=None, strides={}, flatten_block=True, reuse_disabled_inputs=[]) -> str:
    scope = get_scope()
    passes = [
        (0, modify_output_pass),
        (0, modify_input_pass),
    ]
    assert(isinstance(sm_outputs, (tuple, list)))
    assert(isinstance(sm_inputs, (tuple, list)))
    func_args = ", ".join(["{}* __restrict__ {}".format(_type_map[var.dtype], get_valid_name(var)) for var in args])
    with tvm.transform.PassContext(config={"tir.add_lower_pass": passes}):
        scope.shared_mem_outputs = sm_outputs
        scope.shared_mem_inputs = sm_inputs
        scope.reuse_disabled_inputs = reuse_disabled_inputs
        scope.strides = strides
        old_entry = tvm.get_global_func("tvm_callback_cuda_compile")
        tvm.register_func("tvm_callback_cuda_compile", override=True)(lambda x:"")
        mod = tvm.build(sch, args, target=target)
        tvm.register_func("tvm_callback_cuda_compile", override=True)(old_entry)

        src = mod.imported_modules[0].get_source()
        index = src.rindex(TVM_DEFAULT_NAME)
        index = src.index("{", index)
        if flatten_block:
            flat_block_code = get_block_flatten_code(scope.block_size)
            scope.block_size = [int(np.prod(scope.block_size)), 1, 1]
            src = src[:index+2] + flat_block_code + src[index+2:]
        if block_reorder is not None:
            block_reorder_code = get_block_reorder_code(block_reorder)
            src = src[:index+2] + block_reorder_code + src[index+2:]
        if global_kernel:
            prefix = "__global__ void __launch_bounds__(%d) " % np.prod(scope.block_size)
        else:
            prefix = "__device__ void "
            func_args += ", char* shared"
        src = prefix + name + "({}) ".format(func_args) + src[index:]
        # removing shared memory allocation
        # check wmma accumulator shared
        if len(sm_outputs) > 0:
            reuse_output_name = get_valid_name(args[sm_outputs[0]])
            src = re.sub(r"__shared__ (\w+) (\w+wmma_accumulator_shared)\[\d+\];", r"\1* \2 = {};".format(reuse_output_name), src, 1)
        for tensor in scope.shared_mem_inputs:
            shared_var_name = tensor.name + "_shared"
            matched = re.findall(r"__shared__ ((?:signed |unsigned )?\w+) {}\[(\d+)\];".format(shared_var_name), src)
            assert len(matched) == 1
            dtype, size = matched[0]
            scope.exteral_shared_memroy_size[tensor] = int(size) * _type_bytes[dtype]
            src = re.sub(r"__shared__ ((?:signed |unsigned )?\w+) {}\[\d+\];".format(shared_var_name), r"\1* {} = {};".format(shared_var_name, tensor.name), src, 1)
        if not global_kernel:
            pattern = r"__shared__ ((?:signed |unsigned )?\w+) (\w+)\[(\d+)\];"
            offset = 0
            for dtype, var, size in re.findall(pattern, src):
                src = re.sub(r"__shared__ ((?:signed |unsigned )?\w+) {}\[\d+\];".format(var), r"\1* {} = (\1*)(shared+{});".format(var, offset), src, 1)
                buffer_len = int(size) * _type_bytes[dtype]
                buffer_len = (buffer_len + 31) // 32 * 32
                offset += buffer_len
            scope.total_internal_shared_memory = offset
    return src
