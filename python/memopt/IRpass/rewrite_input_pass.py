from typing import List

import tvm
from tvm import te, tir

from .pass_base import PassBase


class RewriteInputPass(PassBase):
    def __init__(self, shared_inputs: List[te.Tensor]) -> None:
        self.shared_inputs = shared_inputs
        super().__init__()

    def run(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
        shared_input_names = [x.name for x in self.shared_inputs]
        def process(op):
            lhs_name = op.buffer.name
            if lhs_name.endswith(".shared") and lhs_name[:-len(".shared")] in shared_input_names:
                return tvm.tir.stmt.SeqStmt([])
            return op

        new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferStore"])
        return f.with_body(new_body)

    def insert_place(self) -> int:
        return 0
