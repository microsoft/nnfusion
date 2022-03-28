import tvm
from .scope import get_scope

@tvm.tir.transform.prim_func_pass(opt_level=0)
def modify_output_pass(f, mod, ctx):
    def process(op):
        nonlocal shared_output_shape
        lhs_name = op.buffer.name
        lhs_shape = op.buffer.shape
        indices = op.indices
        if lhs_name in get_scope().shared_mem_outputs:
            new_indices = [tvm.tir.stmt_functor.substitute(expr, blockIdx_var_map) for expr in indices]
            indices_bound = [ana.const_int_bound(expr) for expr in indices]
            shape = [bound.max_value + 1 for bound in indices_bound]
            assert lhs_name not in shared_output_shape
            assert all([bound.min_value == 0 for bound in indices_bound])
            assert all([bound.max_value < 1e9 for bound in indices_bound])
            shared_output_shape[lhs_name] = shape
            op = tvm.tir.BufferStore(op.buffer, op.value, new_indices, op.span)
            return op
        return op

    ana = get_scope().analyzer
    bounds = get_scope().bounds
    blockIdx_var_map = {}
    shared_output_shape = {}
    for iter_var in bounds:
        if iter_var.var.name.startswith("blockIdx"):
            blockIdx_var_map[iter_var.var] = tvm.tir.const(0)

    new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferStore"])

    # reshape outputs if use shared_memory
    new_buffer_map = {}
    for k, v in f.buffer_map.items():
        if k.name in shared_output_shape:
            v = tvm.tir.decl_buffer(shared_output_shape[k.name], v.dtype, v.name, v.data, v.strides,
                v.elem_offset, v.scope, v.data_alignment, v.offset_factor)
        new_buffer_map[k] = v
    f = tvm.tir.function.PrimFunc(params=f.params, body=new_body,
        ret_type=f.ret_type, buffer_map=new_buffer_map, attrs=f.attrs)
    return f
