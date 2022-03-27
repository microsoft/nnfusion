import tvm
from tvm import te
from .scope import get_scope
import numpy as np

@tvm.tir.transform.prim_func_pass(opt_level=0)
def debug_pass(f, mod, ctx):
    def printer(op):
        print(op, type(op))
        print("-----------------------------------------")
    # import code
    # code.interact(local=locals())
    tvm.tir.stmt_functor.post_order_visit(f.body, printer)
    return f

@tvm.tir.transform.prim_func_pass(opt_level=0)
def get_kernel_info_pass(f, mod, ctx):

    grid_block_size = {
        "threadIdx.x" : 1, "threadIdx.y" : 1, "threadIdx.z" : 1,
        "blockIdx.x" : 1, "blockIdx.y" : 1, "blockIdx.z" : 1,
    }
    def process(op):
        nonlocal grid_block_size
        if isinstance(op, tvm.tir.stmt.Allocate):
            name = op.buffer_var.name
            if not name.endswith("shared"):
                return
            num_elements = np.prod(op.extents)
            num_bytes = num_elements * (int(tvm.DataType(op.dtype).bits) // 8)
            print("Allocate", name, num_bytes)
        elif isinstance(op, tvm.tir.stmt.AttrStmt):
            name = op.node.var.name
            if op.attr_key == 'thread_extent' and name in grid_block_size:
                grid_block_size[name] = max(int(op.value), grid_block_size[name])
    tvm.tir.stmt_functor.post_order_visit(f.body, process)
    print(grid_block_size)
    return f
