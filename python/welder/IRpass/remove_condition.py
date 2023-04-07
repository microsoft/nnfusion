from .pass_base import PassBase
from tvm import tir
import tvm

class RemoveConditionInVectorizePass(PassBase):
    def __init__(self) -> None:
        super().__init__()

    def run(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
        body = tir.stmt_functor.ir_transform(f.body, None, self.check_loop, ["tir.For"])
        return f.with_body(body)

    def check_loop(self, op: tir.For):
        if op.kind != tir.ForKind.VECTORIZED: return
        if "remove_vector_condition" not in op.annotations or not op.annotations["remove_vector_condition"]:
            return
        var = op.loop_var

        def functor(op: tir.Call):
            if op.op.name == "tir.if_then_else":
                cond = op.args[0]
                cond0 = tir.stmt_functor.substitute(cond, {var: tir.const(0)})
                return tir.if_then_else(cond0, op.args[1], op.args[2])

        return tvm.tir.stmt_functor.ir_transform(op, None, functor, ["tir.Call"])

    def insert_place(self) -> int:
        return 1
