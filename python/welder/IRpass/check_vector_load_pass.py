import tvm
from tvm import arith, tir

from .pass_base import PassBase


class CheckVectorLoadPass(PassBase):
    def __init__(self) -> None:
        super().__init__()

    def run(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
        body = tvm.tir.stmt_functor.ir_transform(f.body, None, self.check_loop, ["tir.For"])
        return f.with_body(body)

    def check_loop(self, op: tir.For):
        if op.kind != tir.ForKind.VECTORIZED: return
        if "check_vector_load" not in op.annotations or not op.annotations["check_vector_load"]:
            return
        assert (op.min == 0)
        var = op.loop_var
        lanes = op.extent
        analyzer = arith.Analyzer()
        analyzer.bind(var, 0)
        can_vector_load = True
        if lanes == 8:
            possible_vec_set = {2, 4}
        elif lanes == 4:
            possible_vec_set = {2}
        else:
            possible_vec_set = {}

        def functor(op):
            if isinstance(op, tir.BufferLoad):
                me = analyzer.modular_set(op.indices[0])
                nonlocal can_vector_load
                if me.coeff % lanes != 0 or me.base % lanes != 0:
                    can_vector_load = False
                for val in possible_vec_set.copy():
                    if me.coeff % val != 0 or me.base % val != 0:
                        possible_vec_set.remove(val)

        tvm.tir.stmt_functor.post_order_visit(op.body, functor)

        if can_vector_load:
            return op

        # if original vectorize is not possible, try vectorize with a smaller size
        if len(possible_vec_set) == 0:
            return tir.For(var, 0, lanes, tir.ForKind.SERIAL, op.body)
        else:
            vec_size = max(possible_vec_set)
            outer_var = tir.Var(var.name + "outer", var.dtype)
            inner_var = tir.Var(var.name + "inner", var.dtype)
            return tir.For(outer_var, 0, lanes // vec_size, tir.ForKind.SERIAL,
                tir.For(inner_var, 0, vec_size, tir.ForKind.VECTORIZED,
                tir.stmt_functor.substitute(op.body, {var : inner_var + vec_size * outer_var}))
            )

    def insert_place(self) -> int:
        return 1
