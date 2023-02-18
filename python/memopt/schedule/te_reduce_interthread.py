import numpy as np
from tvm import te

from ..config import Config, Stride
from .te_base import TESchedulerBase


class TEReduceInterThreadScheduler(TESchedulerBase):
    def schedule(self) -> te.Schedule:
        sch, config = self.sche, self.config
        for op in self.ops:
            if op is not self.output_op:
                sch[op].compute_inline()

        assert(self.reduce_op is not None)
        out = self.output_op
        reg_tile = self.reduce_op.output(0)
        self.block_size[0] = int(np.prod(self.config.reduce_thread))
        self.block_size[1] = int(np.prod(self.config.thread))

        # For inter thread reduction case, one thread must only compute one element
        assert self.config.thread == self.config.block

        blck_axis = []
        thrd_axis = []
        for i, axis in enumerate(sch[out].op.axis):
            bx, tx = sch[out].split(axis, factor=self.config.block[i])
            blck_axis.append(bx)
            thrd_axis.append(tx)
        axis_order = blck_axis + thrd_axis
        sch[out].reorder(*axis_order)
        blck_fused = sch[out].fuse(*blck_axis)
        thrd_fused = sch[out].fuse(*thrd_axis)
        sch[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
        sch[out].bind(thrd_fused, te.thread_axis("threadIdx.y"))
        if out is not self.reduce_op:
            sch[reg_tile].compute_at(sch[out], thrd_fused)

        reduce_outer_axis, reduce_inner_axis, reduce_inter_threads = [], [], []

        for i in self.config.raxis_order:
            axis = sch[reg_tile].op.reduce_axis[i]
            ro, _t = sch[reg_tile].split(axis, factor=self.config.rstep[i])
            ri, thd = sch[reg_tile].split(_t, factor=self.config.reduce_thread[i])
            reduce_inter_threads.append(thd)
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)
        axis_order = reduce_inter_threads + reduce_outer_axis + reduce_inner_axis
        sch[reg_tile].reorder(*axis_order)
        fused_reduce_inter_threads = sch[reg_tile].fuse(*reduce_inter_threads)
        sch[reg_tile].bind(fused_reduce_inter_threads, te.thread_axis("threadIdx.x"))

        for input_tensor in self.reduce_op.input_tensors:
            shared_tensor = sch.cache_read(input_tensor, "shared", [reg_tile])
            if input_tensor in self.shared_inputs:
                sch[shared_tensor].compute_at(sch[out], blck_fused)
                strides = self.shared_inputs_strides[input_tensor]
            else:
                sch[shared_tensor].compute_at(sch[reg_tile], reduce_outer_axis[-1])
                strides = Stride()
            if input_tensor.name in self.config.vectorize and not self._is_from_shared(input_tensor):
                vectorize = self.config.vectorize[input_tensor.name]
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
