import tvm
from tvm import te
from .scope import get_scope
import numpy as np

@tvm.tir.transform.prim_func_pass(opt_level=0)
def debug_pass(f, mod, ctx):
    def printer(op):
        print(op, type(op))
        print("-----------------------------------------")

    tvm.tir.stmt_functor.post_order_visit(f.body, printer)
    return f

@tvm.tir.transform.prim_func_pass(opt_level=0)
def get_kernel_info_pass(f, mod, ctx):
    def process(op):
        if isinstance(op, tvm.tir.stmt.Allocate):
            name = op.buffer_var.name
            if not name.endswith("shared"):
                return
            num_elements = np.prod(op.extents)
            num_bytes = num_elements * (int(tvm.DataType(op.dtype).bits) // 8)
            normalized_name = name.replace(".", "_")
            get_scope().interal_shared_memory[normalized_name] = op.buffer_var
            print("Allocate", name, num_bytes)
    tvm.tir.stmt_functor.post_order_visit(f.body, process)
    return f
