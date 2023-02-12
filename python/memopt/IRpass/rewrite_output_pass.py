from typing import List

import numpy as np
import tvm
from tvm import tir

from .pass_base import PassBase


class RewriteOutputPass(PassBase):
    def __init__(self, shared_output: List[int], strides, tile_shape) -> None:
        self.shared_output = shared_output
        self.strides = strides
        self.tile_shape = tile_shape
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
        target_buffer = {}
        for idx in self.shared_output:
            target_buffer[f.buffer_map[f.params[idx]]] = idx

        def process(op):
            nonlocal buffer_map
            indices = op.indices
            if op.buffer in target_buffer:
                new_indices = [self.remove_blockIdx(expr) for expr in indices]
                shape = self.tile_shape
                assert op.buffer not in buffer_map
                if target_buffer[op.buffer] in self.strides:
                    strides = self.strides[target_buffer[op.buffer]].compute_strides_from_shape(shape)
                else:
                    strides = op.buffer.strides
                buffer = tvm.tir.decl_buffer(shape, op.buffer.dtype, op.buffer.name, op.buffer.data, strides,
                    op.buffer.elem_offset, op.buffer.scope, op.buffer.data_alignment, op.buffer.offset_factor)
                buffer_map[op.buffer] = buffer
                op = tvm.tir.BufferStore(buffer, op.value, new_indices, op.span)
                return op
            return op

        def process2(op):
            if op.buffer in target_buffer:
                buffer = buffer_map[op.buffer]
                bounds = [tvm.ir.Range(0, x) for x in buffer.shape]
                return tvm.tir.BufferRealize(buffer, bounds, op.condition, op.body, op.span)
            return op

        buffer_map = {}
        new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferStore"])
        new_body = tvm.tir.stmt_functor.ir_transform(new_body, None, process2, ["tir.BufferRealize"])

        # --------------------- add barrier before and after smem write ----------------
        stack = []
        write_stmt = None
        def add_to_stack(op):
            stack.append(op)

        def process3(op):
            nonlocal write_stmt, stack
            stack.pop(-1)
            if (isinstance(op, tvm.tir.BufferStore) and op.buffer in buffer_map.values()) \
                or (isinstance(op, tvm.tir.Call) and op.op.name == "tir.tvm_store_matrix_sync") \
                or (write_stmt and write_stmt == op):
                if not any([isinstance(x, (tvm.tir.For, tvm.tir.IfThenElse)) for x in stack]) \
                    and not isinstance(stack[-1], tvm.tir.Evaluate): # sometimes calls to tir.tvm_store_matrix_sync are wrapped by evaluate node
                    write_stmt = None
                    barrier = tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"]))
                    # TODO: need to remove unnecessary/redundant barrier?
                    return tvm.tir.stmt_seq(barrier, op, barrier)
                else:
                    write_stmt = stack[-1]
                    return None
            return None
        new_body = tvm.tir.stmt_functor.ir_transform(new_body, add_to_stack, process3)
        assert (len(stack) == 0)
        assert (write_stmt is None)

        # ------------------- reshape outputs if use shared_memory ------------------------
        new_buffer_map = {}
        for k, v in f.buffer_map.items():
            if v in buffer_map:
                v = buffer_map[v]
            new_buffer_map[k] = v
        f = tvm.tir.function.PrimFunc(params=f.params, body=new_body,
            ret_type=f.ret_type, buffer_map=new_buffer_map, attrs=f.attrs)
        return f

    def insert_place(self) -> int:
        return 0
