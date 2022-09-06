import tvm
from tvm import te
from ..scope import get_scope
import numpy as np

@tvm.tir.transform.prim_func_pass(opt_level=0)
def debug_pass(f, mod, ctx):
    def printer(op):
        print(op, type(op))
        print("-----------------------------------------")

    tvm.tir.stmt_functor.post_order_visit(f.body, printer)
    return f

@tvm.tir.transform.prim_func_pass(opt_level=0)
def check_memory_access_pass(f, mod, ctx):
    def checker(op):
        if op.buffer_var.name.endswith(".shared"):
            if "blockIdx" in op.index.astext():
                raise Exception("Invalid shared memory access")
        return op

    tvm.tir.stmt_functor.ir_transform(f.body, None, checker, ["tir.Load"])
    return f
