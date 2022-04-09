import tvm
from .scope import Scope, get_scope

@tvm.tir.transform.prim_func_pass(opt_level=0)
def modify_input_pass(f, mod, ctx):
    def process(op):
        lhs_name = op.buffer.name
        if lhs_name.endswith(".shared") and lhs_name[:-len(".shared")] in get_scope().shared_mem_inputs:
            # print("Removing shared mem load :",)
            # print(op)
            return tvm.tir.stmt.SeqStmt([])
        return op

    def process2(op):
        if op.buffer.name not in get_scope().shared_mem_inputs:
            return op
        new_indices = [tvm.tir.stmt_functor.substitute(expr, blockIdx_var_map) for expr in op.indices]
        indices_bound = [get_scope().analyzer.const_int_bound(expr) for expr in new_indices]
        # from .debug import debug
        # debug({**globals(), **locals()})
        return tvm.tir.BufferLoad(op.buffer, new_indices, op.span)

    blockIdx_var_map = {}
    bounds = get_scope().bounds
    for iter_var in bounds:
        if iter_var.var.name.startswith("blockIdx"):
            blockIdx_var_map[iter_var.var] = tvm.tir.const(0)
    new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process2, ["tir.BufferLoad"])
    new_body = tvm.tir.stmt_functor.ir_transform(new_body, None, process, ["tir.BufferStore"])
    return f.with_body(new_body)
