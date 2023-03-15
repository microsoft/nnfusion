import numpy as np
from tvm import tir

from ..config import Config, Stride
from .tir_base import TIRSchedulerBase


class TIRSIMTScheduler(TIRSchedulerBase):
    def schedule(self) -> tir.Schedule:
        sch, config = self.sche, self.config
        self.block_size[0] = int(np.prod(config.thread))

        C = sch.get_block(self.reduce_op.name)
        CL = sch.cache_write(C, 0, "local")
        space_loops = sch.get_loops(C)[:len(self.reduce_op.axis)]
        reduce_loops = sch.get_loops(C)[-len(self.reduce_op.reduce_axis):]

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for i, loop in enumerate(space_loops):
            bx, _t = sch.split(loop, factors=[None, config.block[i]])
            blck_axis.append(bx)
            if config.step[i] > 1:
                _t, tn = sch.split(_t, factors=[None, config.step[i]])
                tile_axis.append(tn)
            if config.block[i] <= config.thread[i] * config.step[i]:
                tx = _t
            else:
                vx, tx = sch.split(_t, factors=[None, config.thread[i]])
                vthd_axis.append(vx)
            thrd_axis.append(tx)

        reduce_outer_axis, reduce_inner_axis = [], []
        for i in config.raxis_order:
            loop = reduce_loops[i]
            ro, ri = sch.split(loop, factors=[None, config.rstep[i]])
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)

        vthd_axis = list(reversed(vthd_axis)) # inner virtual thread first
        axis_order = blck_axis + vthd_axis + thrd_axis + reduce_outer_axis + reduce_inner_axis + tile_axis

        sch.reorder(*axis_order)
        blck_fused = sch.fuse(*blck_axis)
        thrd_fused = sch.fuse(*thrd_axis)
        sch.bind(blck_fused, "blockIdx.x")
        sch.bind(thrd_fused, "threadIdx.x")
        if len(vthd_axis) > 3:
            vthd_axis = vthd_axis[0:2] + [sch.fuse(*vthd_axis[2:])]
        for i, ax in enumerate(vthd_axis):
            sch.bind(ax, "vthread" + ['.x', '.y', '.z'][i])
        for ax in tile_axis:
            sch.unroll(ax)

        cached_stages = []
        for i, input_tensor in enumerate(self.reduce_op.input_tensors):
            SS = sch.cache_read(C, i, "shared")
            cached_stages.append(SS)
            if input_tensor in self.shared_inputs:
                sch.compute_at(SS, blck_fused)
                strides = self.shared_inputs_strides[input_tensor]
                dim_offset = 1
            else:
                sch.compute_at(SS, reduce_outer_axis[-1])
                strides = Stride()
                dim_offset = len(vthd_axis) + len(reduce_outer_axis) + 2 # outer loops are: blck_fused, thrd_fused, vthd_axis, reduce_outer_axis
            if input_tensor.name in config.vectorize and not self._is_from_shared(input_tensor):
                vectorize = config.vectorize[input_tensor.name]
            else:
                vectorize = 1
            self.cooperative_fetch(SS, dim_offset, strides, vectorize)

        sch.reverse_compute_at(CL, thrd_fused)
        if len(tile_axis) > 0:
            for ax in sch.get_loops(CL)[-len(tile_axis):]:
                sch.unroll(ax)
        sch.decompose_reduction(C, reduce_outer_axis[0])
        self.schedule_compute_inline()

        # ----- cache small tensors -----
        cache_plan = self.make_cache_plan()
        consumer_ops = {t.op for t in self.reduce_op.input_tensors}
        consumer_ops.add(self.output_op)
        op_input_map = self.detect_op_inputs(consumer_ops)
        for tensor in cache_plan:
            block = None
            if tensor.op in op_input_map[self.output_op]:
                block = CL
            else:
                for i, t in enumerate(self.reduce_op.input_tensors):
                    if tensor.op in op_input_map[t.op]:
                        block = cached_stages[i]
                        break
            assert block
            tensor_shared = sch.cache_read(block, tensor.name, "shared")
            if len(self.shared_outputs) > 0:
                tensor_local = sch.cache_read(block, tensor.name + "_shared", "local")
                sch.compute_at(tensor_local, thrd_fused)
                if len(tile_axis) > 0:
                    for ax in sch.get_loops(tensor_local)[-len(tile_axis):]:
                        sch.unroll(ax)
            sch.compute_at(tensor_shared, thrd_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            dim_offset = len(vthd_axis) + 2 # outer loops are: blck_fused vthd_axis thrd_fused
            self.cooperative_fetch(tensor_shared, dim_offset, strides)

        return sch.mod["main"]
