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
            vx, tx, tn = sch.split(_t, factors=[None, config.thread[i], config.step[i]])
            blck_axis.append(bx)
            vthd_axis.append(vx)
            thrd_axis.append(tx)
            tile_axis.append(tn)

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
        vthd_fused = sch.fuse(*vthd_axis)
        sch.bind(blck_fused, "blockIdx.x")
        sch.bind(thrd_fused, "threadIdx.x")
        sch.bind(vthd_fused, "vthread.x")
        # for tn in tile_axis:
        #     sch.unroll(tn)
        for i, input_tensor in enumerate(self.reduce_op.input_tensors):
            SS = sch.cache_read(C, i, "shared")
            if input_tensor in self.shared_inputs:
                sch.compute_at(SS, blck_fused)
                strides = self.shared_inputs_strides[input_tensor]
            else:
                sch.compute_at(SS, reduce_outer_axis[-1])
                strides = Stride()
            if input_tensor.name in config.vectorize and not self._is_from_shared(input_tensor):
                vectorize = config.vectorize[input_tensor.name]
            else:
                vectorize = 1
            dim_offset = len(reduce_outer_axis) + 3 # outer loops are: blck_fused, thrd_fused, vthd_fused, reduce_outer_axis
            self.cooperative_fetch(SS, dim_offset, strides, vectorize)

        sch.reverse_compute_at(CL, reduce_outer_axis[0])
        sch.decompose_reduction(C, reduce_outer_axis[0])
        self.schedule_compute_inline()

        return sch.mod["main"]
