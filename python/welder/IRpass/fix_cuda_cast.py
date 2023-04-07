import tvm
from tvm import tir

from .pass_base import PassBase


class FixCudaCastPass(PassBase):
    def __init__(self) -> None:
        super().__init__()

    def run(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
        def process(op):
            if op.value.dtype == "int64" and op.dtype == "float16":
                return tir.Cast(op.dtype, tir.Cast("float32", op.value, op.span), op.span)
            return op

        new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.Cast"])
        return f.with_body(new_body)

    def insert_place(self) -> int:
        return 0
