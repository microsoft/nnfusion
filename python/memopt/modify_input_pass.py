import tvm
from .scope import get_scope

@tvm.tir.transform.prim_func_pass(opt_level=0)
def modify_input_pass(f, mod, ctx):
    def process(op):
        lhs_name = op.buffer.name
        if lhs_name.endswith(".shared") and lhs_name[:-len(".shared")] in get_scope().shared_mem_inputs:
            # print("Removing shared mem load :",)
            # print(op)
            return tvm.tir.stmt.SeqStmt([])
        return op

    new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferStore"])
    return f.with_body(new_body)
