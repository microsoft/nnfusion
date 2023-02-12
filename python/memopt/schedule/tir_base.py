from typing import List

import tvm
from tvm import te, tir

from ..config import Stride
from .scheduler_base import SchedulerBase


class TIRSchedulerBase(SchedulerBase):
    def create_schedule(self) -> tir.Schedule:
        workload = te.create_prim_func(self.args)
        ir_module = tvm.IRModule({"main": workload})
        return tir.Schedule(ir_module)

    def debug_schedule(self):
        print(self.sche.mod["main"].script())

    def schedule_compute_inline(self) -> None:
        for op in self.ops:
            if op in (self.reduce_op, self.output_op):
                continue
            block = self.sche.get_block(op.name)
            self.sche.compute_inline(block)
        if self.reduce_op != None and self.reduce_op != self.output_op:
            block = self.sche.get_block(self.output_op.name)
            self.sche.reverse_compute_inline(block)

    def cooperative_fetch(self, SS: tir.Block, dim_offset: int, strides: Stride = Stride(), inner_step: int = 1, vectorize_inner=True):
        assert self.block_size[2] == 1
        axes = self.sche.get_loops(SS)[dim_offset:]
        if strides.is_valid():
            self.sche.storage_align(SS, 0, strides.ax, strides.stride - 1, strides.stride)
        fused = self.sche.fuse(*axes)
        oo, ty, tx, tv = self.sche.split(fused, factors=[None, self.block_size[1], self.block_size[0], inner_step])
        if vectorize_inner:
            self.sche.vectorize(tv)
        else:
            self.sche.unroll(tv)
        self.sche.unroll(oo)
        self.sche.bind(tx, "threadIdx.x")
        self.sche.bind(ty, "threadIdx.y")

    def plan_cache(self):
        pass

    def build(self, target) -> str:
        with tvm.transform.PassContext(config={"tir.add_lower_pass": self.passes}):
            mod = tvm.build(self.sche.mod["main"], self.args, target=target)
        return mod.imported_modules[0].get_source()
