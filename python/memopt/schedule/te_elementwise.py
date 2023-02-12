import numpy as np
from tvm import te

from ..config import Config, Stride
from .te_base import TESchedulerBase


class TEElementWiseScheduler(TESchedulerBase):
    def schedule(self) -> te.Schedule:
        sch, config = self.sche, self.config
        for op in self.ops:
            if op is not self.output_op:
                sch[op].compute_inline()
        out = self.output_op
        self.block_size[0] = int(np.prod(config.thread))

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for i, axis in enumerate(sch[out].op.axis):
            bx, _t = sch[out].split(axis, factor=config.block[i])
            vx, _t = sch[out].split(_t, factor=config.thread[i] * config.step[i])
            tx, tn = sch[out].split(_t, factor=config.step[i])
            blck_axis.append(bx)
            vthd_axis.append(vx)
            thrd_axis.append(tx)
            tile_axis.append(tn)
        vthd_axis = list(reversed(vthd_axis)) # inner virtual thread first
        axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis
        sch[out].reorder(*axis_order)
        blck_fused = sch[out].fuse(*blck_axis)
        thrd_fused = sch[out].fuse(*thrd_axis)
        sch[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
        for va in vthd_axis:
            sch[out].bind(va, te.thread_axis("vthread"))
        sch[out].bind(thrd_fused, te.thread_axis("threadIdx.x"))
        for tn in tile_axis:
            sch[out].unroll(tn)

        cache_plan = {}
        for op in self.none_reduce_ops:
            for tensor in op.input_tensors:
                if self.requires_cache(tensor, op):
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = sch.cache_read(tensor, "shared", consumers)
            sch[tensor_shared].compute_at(sch[out], thrd_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            self.cooperative_fetch(tensor_shared, strides)
            if len(self.shared_outputs) == 0: continue
            tensor_local = sch.cache_read(tensor_shared, "local", consumers)
            sch[tensor_local].compute_at(sch[out], thrd_fused)
        return sch
