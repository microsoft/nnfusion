from typing import List

import tvm
from tvm import te, tir

from .pass_base import PassBase


class RewriteInputPass(PassBase):
    def __init__(self, shared_inputs: List[te.Tensor], from_legacy_te: bool) -> None:
        self.shared_inputs = shared_inputs
        self.from_legacy_te = from_legacy_te
        super().__init__()

    def remove_blockIdx(self, expr: tir.Stmt) -> tir.Stmt:
        substitute_vars = {}
        def fvisit(var):
            if isinstance(var, tvm.tir.expr.Var):
                if str(var).startswith("blockIdx"):
                    substitute_vars[var] = tvm.tir.const(0)
        tvm.tir.stmt_functor.post_order_visit(expr, fvisit=fvisit)
        new_expr = tvm.tir.stmt_functor.substitute(expr, substitute_vars)
        return new_expr

    def run(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
        shared_input_names = [x.name for x in self.shared_inputs]
        def process(op):
            lhs_name = op.buffer.name
            if lhs_name.endswith(".shared") and lhs_name[:-len(".shared")] in shared_input_names:
                return tvm.tir.stmt.SeqStmt([])
            return op
        def process2(op):
            if op.buffer.name not in shared_input_names:
                return op
            new_indices = [self.remove_blockIdx(expr) for expr in op.indices]
            return tvm.tir.BufferLoad(op.buffer, new_indices, op.span)

        new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process2, ["tir.BufferLoad"])
        new_body = tvm.tir.stmt_functor.ir_transform(new_body, None, process, ["tir.BufferStore"])
        return f.with_body(new_body)

    def insert_place(self) -> int:
        return 0 if self.from_legacy_te else 1
