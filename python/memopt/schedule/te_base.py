import numpy as np
from tvm import te

from ..config import Stride
from .scheduler_base import SchedulerBase


class TESchedulerBase(SchedulerBase):
    def cooperative_fetch(self, shared, sch, strides: Stride = Stride(), inner_step: int = 1, vectorize_inner=True):
        assert self.thread_per_block[2] == 1
        axes = sch[shared].op.axis
        if strides.is_valid():
            sch[shared].storage_align(axes[strides.ax], strides.stride - 1, strides.stride)
        fused = sch[shared].fuse(*axes)
        fused, tv = sch[shared].split(fused, factor=inner_step)
        _t, tx = sch[shared].split(fused, factor=self.thread_per_block[0])
        oo, ty = sch[shared].split(_t, factor=self.thread_per_block[1])
        sch[shared].reorder(oo, ty, tx)
        if vectorize_inner:
            sch[shared].vectorize(tv)
        else:
            sch[shared].unroll(tv)
        sch[shared].unroll(oo)
        sch[shared].bind(tx, te.thread_axis("threadIdx.x"))
        sch[shared].bind(ty, te.thread_axis("threadIdx.y"))

    def requires_cache(self, tensor, op):
        assert tensor in op.input_tensors
        if tensor in self.shared_inputs:
            return True
        cache = isinstance(tensor.op, te.PlaceholderOp) \
                    and len(op.output(0).shape) > len(tensor.shape) \
                    and np.prod(op.output(0).shape) > np.prod(tensor.shape) # is broadcast
        return cache

    def create_te_schedule(self) -> te.Schedule:
        sch = te.create_schedule([self.output_op])
        # use the op reference in te.schedule to avoid bugs
        self.ops = []
        for op in sch.stage_map:
            if isinstance(op, te.ComputeOp):
                if op == self.output_op:
                    self.output_op = op
                if op == self.reduce_op:
                    self.reduce_op = op
                self.ops.append(op)
        return sch
