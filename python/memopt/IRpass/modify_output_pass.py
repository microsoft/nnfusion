import tvm
from ..scope import get_scope
import numpy as np

@tvm.tir.transform.prim_func_pass(opt_level=0)
def modify_output_pass(f, mod, ctx):
    target_buffer = {}
    for idx in get_scope().shared_mem_outputs:
        target_buffer[f.buffer_map[f.params[idx]]] = idx

    def process(op):
        nonlocal buffer_map
        indices = op.indices
        if op.buffer in target_buffer:
            new_indices = [tvm.tir.stmt_functor.substitute(expr, blockIdx_var_map) for expr in indices]
            indices_bound = [ana.const_int_bound(expr) for expr in indices]
            shape = [bound.max_value + 1 for bound in indices_bound]
            assert op.buffer not in buffer_map
            assert all([bound.min_value == 0 for bound in indices_bound])
            assert all([bound.max_value < 1e9 for bound in indices_bound])
            if op.buffer.name in get_scope().strides:
                strides = get_scope().strides[op.buffer.name]
                num_bytes = shape[0] * strides[0] * (int(tvm.DataType(op.buffer.dtype).bits) // 8)
            else:
                strides = op.buffer.strides
                num_bytes = np.prod(shape) * (int(tvm.DataType(op.buffer.dtype).bits) // 8)
            get_scope().exteral_shared_memroy_size[target_buffer[op.buffer]] = num_bytes
            buffer = tvm.tir.decl_buffer(shape, op.buffer.dtype, op.buffer.name, op.buffer.data, strides,
                op.buffer.elem_offset, op.buffer.scope, op.buffer.data_alignment, op.buffer.offset_factor)
            buffer_map[op.buffer] = buffer
            op = tvm.tir.BufferStore(buffer, op.value, new_indices, op.span)
            return op
        return op

    def process2(op):
        if op.buffer in target_buffer:
            buffer = buffer_map[op.buffer]
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

    # --------------------- add barrier before and after smem write ----------------
    stack = []
    write_stmt = None
    def add_to_stack(op):
        stack.append(op)

    def process3(op):
        nonlocal write_stmt, stack
        stack.pop(-1)
        if (isinstance(op, tvm.tir.BufferStore) and op.buffer in buffer_map.values()) or (write_stmt and write_stmt == op):
            if not any([isinstance(x, (tvm.tir.stmt.For, tvm.tir.stmt.IfThenElse)) for x in stack]):
                write_stmt = None
                barrier = tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"]))
                return tvm.tir.stmt_seq(barrier, op, barrier)
            else:
                write_stmt = stack[-1]
                return None
        return None
    new_body = tvm.tir.stmt_functor.ir_transform(new_body, add_to_stack, process3)
    assert (len(stack) == 0)
    assert (write_stmt is None)

    # ------------------- reshape outputs if use shared_memory ------------------------
    new_buffer_map = {}
    for k, v in f.buffer_map.items():
        if v in buffer_map:
            v = buffer_map[v]
        new_buffer_map[k] = v
    f = tvm.tir.function.PrimFunc(params=f.params, body=new_body,
        ret_type=f.ret_type, buffer_map=new_buffer_map, attrs=f.attrs)
    return f
