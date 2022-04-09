import tvm
from .scope import get_scope
import numpy as np

@tvm.tir.transform.prim_func_pass(opt_level=0)
def modify_output_pass(f, mod, ctx):
    def process(op):
        nonlocal buffer_map
        lhs_name = op.buffer.name
        lhs_shape = op.buffer.shape
        indices = op.indices
        if lhs_name in get_scope().shared_mem_outputs:
            new_indices = [tvm.tir.stmt_functor.substitute(expr, blockIdx_var_map) for expr in indices]
            indices_bound = [ana.const_int_bound(expr) for expr in indices]
            shape = [bound.max_value + 1 for bound in indices_bound]
            assert lhs_name not in buffer_map
            assert all([bound.min_value == 0 for bound in indices_bound])
            assert all([bound.max_value < 1e9 for bound in indices_bound])
            num_bytes = np.prod(shape) * (int(tvm.DataType(op.buffer.dtype).bits) // 8)
            get_scope().exteral_shared_memroy_size[lhs_name] = num_bytes
            buffer = tvm.tir.decl_buffer(shape, op.buffer.dtype, lhs_name, op.buffer.data, op.buffer.strides,
                op.buffer.elem_offset, op.buffer.scope, op.buffer.data_alignment, op.buffer.offset_factor)
            buffer_map[lhs_name] = buffer
            op = tvm.tir.BufferStore(buffer, op.value, new_indices, op.span)
            return op
        return op

    def process2(op):
        name = op.buffer.name
        if name in get_scope().shared_mem_outputs:
            buffer = buffer_map[name]
            bounds = [tvm.ir.Range(0, x) for x in buffer.shape]
            return tvm.tir.BufferRealize(buffer, bounds, op.condition, op.body, op.span)
        return op

    ana = get_scope().analyzer
    blockIdx_var_map = {}
    for iter_var in get_scope().bounds:
        if iter_var.var.name.startswith("blockIdx"):
            blockIdx_var_map[iter_var.var] = tvm.tir.const(0)

    buffer_map = {}
    new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferStore"])
    new_body = tvm.tir.stmt_functor.ir_transform(new_body, None, process2, ["tir.BufferRealize"])

    # reshape outputs if use shared_memory
    new_buffer_map = {}
    for k, v in f.buffer_map.items():
        if k.name in buffer_map:
            v = buffer_map[k.name]
        new_buffer_map[k] = v
    f = tvm.tir.function.PrimFunc(params=f.params, body=new_body,
        ret_type=f.ret_type, buffer_map=new_buffer_map, attrs=f.attrs)
    return f
