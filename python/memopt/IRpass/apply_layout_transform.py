import tvm

from .scope import get_scope


def process(op : tvm.tir.BufferStore):
    if not op.buffer.name in get_scope().apply_buffer_layout:
        return op
    layout = get_scope().apply_buffer_layout[op.buffer.name]

    assert(len(op.indices) == 1) # already flattened

    new_indices = [layout(op.indices[0])]

    return tvm.tir.BufferStore(op.buffer, op.value, new_indices, op.span)

@tvm.tir.transform.prim_func_pass(opt_level=0)
def apply_layout_transform_pass(f, mod, ctx):
    new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferStore"])
    return f.with_body(new_body)
