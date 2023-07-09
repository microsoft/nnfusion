import tvm


@tvm.tir.transform.prim_func_pass(opt_level=0)
def debug_pass(f, mod, ctx):
    def printer(op):
        print(op, type(op))
        print("-----------------------------------------")

    tvm.tir.stmt_functor.post_order_visit(f.body, printer)
    return f
