import numpy as np
from tvm import te

from ..config import Config, Stride
from .te_base import TESchedulerBase


class TEReduceScheduler(TESchedulerBase):
    def schedule(self) -> te.Schedule:
        sch, config = self.sche, self.config
        for op in self.ops:
            if op is not self.output_op:
                sch[op].compute_inline()
        out = self.output_op
        self.block_size[0] = int(np.prod(config.thread))
        reg_tile = sch.cache_write(self.reduce_op.output(0), "local")

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

        sch[reg_tile].compute_at(sch[out], thrd_fused)

        reduce_outer_axis, reduce_inner_axis = [], []
        space_axis = list(sch[reg_tile].op.axis)

        for i in config.raxis_order:
            axis = sch[reg_tile].op.reduce_axis[i]
            ro, ri = sch[reg_tile].split(axis, factor=config.rstep[i])
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)
        axis_order = reduce_outer_axis + reduce_inner_axis + space_axis
        sch[reg_tile].reorder(*axis_order)
        space_fused = sch[reg_tile].fuse(*space_axis)
        sch[reg_tile].unroll(space_fused)

        for input_tensor in self.reduce_op.input_tensors:
            shared_tensor = sch.cache_read(input_tensor, "shared", [reg_tile])
            if input_tensor in self.shared_inputs:
                sch[shared_tensor].compute_at(sch[out], blck_fused)
                strides = self.shared_inputs_strides[input_tensor]
            else:
                sch[shared_tensor].compute_at(sch[reg_tile], reduce_outer_axis[-1])
                strides = Stride()
            if input_tensor.name in config.vectorize and not self._is_from_shared(input_tensor):
                vectorize = config.vectorize[input_tensor.name]
                if input_tensor not in self.args:
                    vectorize = min(4 , vectorize) # tvm not supporting ramp for 8 elements
            else:
                vectorize = 1
            self.cooperative_fetch(shared_tensor, strides, vectorize)

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
            # This is a hack, TVM cannot handle cached_local_read when padding on a shared input
            consumers = list(filter(lambda x: x.output(0) not in self.reduce_op.input_tensors, consumers))
            if len(consumers) == 0 or len(self.shared_outputs) == 0: continue
            tensor_local = sch.cache_read(tensor_shared, "local", consumers)
            sch[tensor_local].compute_at(sch[out], thrd_fused)

        return sch
