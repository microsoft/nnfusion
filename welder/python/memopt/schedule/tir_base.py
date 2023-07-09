from typing import List

import numpy as np
import tvm
from tvm import te, tir

from ..config import Stride
from ..IRpass import *
from .scheduler_base import SchedulerBase


class TIRSchedulerBase(SchedulerBase):
    def create_schedule(self) -> tir.Schedule:
        workload = te.create_prim_func(self.args)
        ir_module = tvm.IRModule({"main": workload})
        return tir.Schedule(ir_module)

    def debug_schedule(self):
        print(self.sche.mod["main"].script())

    def schedule_compute_inline(self) -> None:
        for op in reversed(self.ops):
            if op not in (self.reduce_op, self.output_op):
                block = self.sche.get_block(op.name)
                self.sche.compute_inline(block)
        if self.reduce_op != None and self.reduce_op != self.output_op:
            block = self.sche.get_block(self.output_op.name)
            self.sche.reverse_compute_inline(block)

    def cooperative_fetch(self, SS: tir.Block, dim_offset: int, strides: Stride=Stride(), vector_load: int=1, use_pragma_unroll: bool=False):
        loops = self.sche.get_loops(SS)
        if len(loops) == dim_offset:
            # handle fetching only one element
            loops.append(self.sche.add_unit_loop(SS))
        assert len(loops) > dim_offset
        axes = loops[dim_offset:]
        if strides.is_valid():
            self.sche.storage_align(SS, 0, strides.ax, strides.stride - 1, strides.stride)
        ax = self.sche.fuse(*axes)
        if vector_load > 1:
            ax, tv = self.sche.split(ax, factors=[None, vector_load])
            self.sche.vectorize(tv)
            self.sche.annotate(tv, "check_vector_load", True)
            self.sche.annotate(tv, "remove_vector_condition", True)
        if self.block_size[0] > 1:
            ax, tx = self.sche.split(ax, factors=[None, self.block_size[0]])
            self.sche.bind(tx, "threadIdx.x")
        if self.block_size[1] > 1:
            ax, ty = self.sche.split(ax, factors=[None, self.block_size[1]])
            self.sche.bind(ty, "threadIdx.y")
        if self.block_size[2] > 1:
            ax, tz = self.sche.split(ax, factors=[None, self.block_size[2]])
            self.sche.bind(tz, "threadIdx.z")
        self.sche.unroll(ax)
        if use_pragma_unroll:
            self.sche.annotate(ax, "pragma_unroll_explicit", False)

    def build(self, target) -> str:
        with tvm.transform.PassContext(config={"tir.add_lower_pass": self.passes}):
            mod = tvm.build(self.sche.mod["main"], self.args, target=target)
        return mod.imported_modules[0].get_source()

    def make_passes(self) -> None:
        self.passes.append(RewriteOutputPass(self.shared_outputs, self.config.output_strides,
                                             (self.config.block, self.output_op.output(0).shape), False).get_pass())
        self.passes.append(RewriteInputPass(self.shared_inputs, False).get_pass())
        self.passes.append(FixCudaCastPass().get_pass())
        self.passes.append(CheckVectorLoadPass().get_pass())
        self.passes.append(RemoveConditionInVectorizePass().get_pass())

    def detect_op_inputs(self, consumer_ops):
        op_input_map = {op : set() for op in consumer_ops}
        for op in reversed(self.ops):
            op_inputs = [t.op for t in op.input_tensors]
            if op in consumer_ops:
                op_input_map[op].update(op_inputs)
            else:
                for c_op in consumer_ops:
                    if op in op_input_map[c_op]:
                        op_input_map[c_op].update(op_inputs)
        return op_input_map

    def requires_cache(self, tensor, op):
        assert tensor in op.input_tensors
        cache = isinstance(tensor.op, te.PlaceholderOp) \
                    and len(op.output(0).shape) > len(tensor.shape) \
                    and np.prod(op.output(0).shape) > np.prod(tensor.shape) # is broadcast
        return cache

    def make_cache_plan(self):
        cache_plan = {}
        # do not cache reduce op's input again
        already_cached = self.reduce_op.input_tensors if self.reduce_op else []
        for op in self.none_reduce_ops:
            for tensor in op.input_tensors:
                if tensor in already_cached: continue
                if self.requires_cache(tensor, op):
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)
        return cache_plan
