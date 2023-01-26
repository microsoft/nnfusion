import tvm


def process(op : tvm.tir.BufferLoad):
    if not op.buffer.name.endswith("cutlass.warp.mma"):
        return op

    substitute_vars = {}
    def fvisit(var):
        if isinstance(var, tvm.tir.expr.Var):
            if str(var).startswith("threadIdx.x"):
                substitute_vars[var] = tvm.tir.const(0)

    for expr in op.indices:
        tvm.tir.stmt_functor.post_order_visit(expr, fvisit=fvisit)
    new_indices = [tvm.tir.stmt_functor.substitute(expr, substitute_vars) for expr in op.indices]
    return tvm.tir.BufferLoad(op.buffer, new_indices, op.span)

@tvm.tir.transform.prim_func_pass(opt_level=0)
def fragment_access_rewrite_pass(f, mod, ctx):
    new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferLoad"])
    return f.with_body(new_body)
