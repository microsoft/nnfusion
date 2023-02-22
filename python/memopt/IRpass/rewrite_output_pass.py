from typing import List

import numpy as np
import tvm
from tvm import tir

from ..config import Stride
from .pass_base import PassBase


class RewriteOutputPass(PassBase):
    def __init__(self, shared_output: List[int], strides, tile_shape, from_legacy_te: bool) -> None:
        self.shared_output = shared_output
        self.strides = strides
        self.tile_shape = tile_shape
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

    def add_memory_barrier(self, body, buffer_map):
        # --------------------- add barrier before and after smem write ----------------
        stack = []
        write_stmt = None
        def add_to_stack(op):
            stack.append(op)

        def process(op):
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
        body = tvm.tir.stmt_functor.ir_transform(body, add_to_stack, process)
        assert (len(stack) == 0)
        assert (write_stmt is None)
        return body

    def run(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
        if self.from_legacy_te:
            return self.run_ndim(f, mod, ctx)
        else:
            return self.run_flattened(f, mod, ctx)

    def remap_ndim(self, expr, shape0, strides):
        index = []
        for dim in reversed(shape0):
            index.append(expr % dim)
            expr = expr // dim
        result = 0
        for indice, step in zip(index, reversed(strides)):
            result += indice * step
        result = tvm.arith.Analyzer().simplify(result)
        return result

    def run_flattened(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
        target_buffer = {}
        for idx in self.shared_output:
            target_buffer[f.buffer_map[f.params[idx]]] = idx
        tile_shape, full_shape = self.tile_shape

        def process(op):
            nonlocal buffer_map
            indices = op.indices
            if op.buffer in target_buffer:
                assert len(op.indices) == 1
                assert op.buffer not in buffer_map
                if target_buffer[op.buffer] in self.strides:
                    strides = self.strides[target_buffer[op.buffer]].compute_strides_from_shape(tile_shape)
                else:
                    strides = Stride().compute_elements_from_shape(tile_shape)
                new_indices = [self.remap_ndim(self.remove_blockIdx(indices[0]), full_shape, strides)]
                
                buffer = tvm.tir.decl_buffer([strides[0] * tile_shape[0]], op.buffer.dtype, op.buffer.name, op.buffer.data, op.buffer.strides,
                    op.buffer.elem_offset, op.buffer.scope, op.buffer.data_alignment, op.buffer.offset_factor)
                buffer_map[op.buffer] = buffer
                op = tvm.tir.BufferStore(buffer, op.value, new_indices, op.span)
                return op
            return op

        buffer_map = {}
        body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferStore"])
        body = self.add_memory_barrier(body, buffer_map)
        
        # ------------------- reshape outputs if use shared_memory ------------------------
        new_buffer_map = {}
        for k, v in f.buffer_map.items():
            if v in buffer_map:
                v = buffer_map[v]
            new_buffer_map[k] = v

        return tvm.tir.function.PrimFunc(params=f.params, body=body,
            ret_type=f.ret_type, buffer_map=new_buffer_map, attrs=f.attrs)

    def run_ndim(self, f: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.tir.PrimFunc:
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
        body = tvm.tir.stmt_functor.ir_transform(f.body, None, process, ["tir.BufferStore"])
        body = tvm.tir.stmt_functor.ir_transform(body, None, process2, ["tir.BufferRealize"])
        body = self.add_memory_barrier(body, buffer_map)

        # ------------------- reshape outputs if use shared_memory ------------------------
        new_buffer_map = {}
        for k, v in f.buffer_map.items():
            if v in buffer_map:
                v = buffer_map[v]
            new_buffer_map[k] = v
        f = tvm.tir.function.PrimFunc(params=f.params, body=body,
            ret_type=f.ret_type, buffer_map=new_buffer_map, attrs=f.attrs)
        return f

    def insert_place(self) -> int:
        return 0 if self.from_legacy_te else 1
