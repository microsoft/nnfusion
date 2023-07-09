from typing import Dict, List

import numpy as np
import tvm
from tvm import te

from .config import Config, Stride


class Scheduler:
    def __init__(self):
        pass

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
        cache = isinstance(self.sche[tensor].op, tvm.te.PlaceholderOp) \
                    and len(op.output(0).shape) > len(tensor.shape) \
                    and np.prod(op.output(0).shape) > 16 * np.prod(tensor.shape) # is broadcast
        return cache

    # [Parameters]
    #   schedule: the original TVM schedule of an op
    #   config: apply this config onto the orginal schedule
    #   shared_inputs: inputs that are already in the shared memory, which is used for fusion case.
    # [Return]
    #   new_s: an optimized TVM schedule
    def rewrite_schedule(self, schedule: te.Schedule, config: Config, shared_inputs: List[te.Tensor] = [],
        shared_inputs_strides: Dict[te.Tensor, Stride] = {}, shared_outputs = []):
        self.config = config
        self.sche = schedule
        self.shared_inputs = shared_inputs
        self.shared_inputs_strides = {tensor: Stride() for tensor in shared_inputs}
        self.shared_inputs_strides.update(shared_inputs_strides)
        self.shared_outputs = shared_outputs

        self.reduce_op = None
        self.output_op = None
        self.elementwise_ops = []
        for op, stage in self.sche.stage_map.items():
            if isinstance(op, tvm.te.ComputeOp):
                if stage.is_output:
                    assert self.output_op is None
                    self.output_op = op
                else:
                    stage.compute_inline()
                if len(op.reduce_axis) > 0:
                    assert self.reduce_op is None
                    self.reduce_op = op
                else:
                    self.elementwise_ops.append(op)
        assert self.output_op is not None
        if self.reduce_op is not None:
            if self.config.use_tc:
                return self._rewrite_schedule_tc()
            if np.prod(self.config.reduce_thread) > 1:
                return self._rewrite_schedule_inter_thread()
            else:
                return self._rewrite_schedule_default()
        else:
            return self._rewrite_schedule_no_reduce()

    def _rewrite_schedule_no_reduce(self):
        assert(self.reduce_op is None)
        out = self.output_op
        self.thread_per_block = [int(np.prod(self.config.thread)), 1, 1]

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for i, axis in enumerate(self.sche[out].op.axis):
            bx, _t = self.sche[out].split(axis, factor=self.config.block[i])
            vx, _t = self.sche[out].split(_t, factor=self.config.thread[i] * self.config.step[i])
            tx, tn = self.sche[out].split(_t, factor=self.config.step[i])
            blck_axis.append(bx)
            vthd_axis.append(vx)
            thrd_axis.append(tx)
            tile_axis.append(tn)
        vthd_axis = list(reversed(vthd_axis)) # inner virtual thread first
        axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis
        self.sche[out].reorder(*axis_order)
        blck_fused = self.sche[out].fuse(*blck_axis)
        thrd_fused = self.sche[out].fuse(*thrd_axis)
        self.sche[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
        for va in vthd_axis:
            self.sche[out].bind(va, te.thread_axis("vthread"))
        self.sche[out].bind(thrd_fused, te.thread_axis("threadIdx.x"))
        for tn in tile_axis:
            self.sche[out].unroll(tn)

        cache_plan = {}
        for op in self.elementwise_ops:
            for tensor in op.input_tensors:
                if self.requires_cache(tensor, op):
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = self.sche.cache_read(tensor, "shared", consumers)
            self.sche[tensor_shared].compute_at(self.sche[out], thrd_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            self.cooperative_fetch(tensor_shared, self.sche, strides)
            if len(self.shared_outputs) == 0: continue
            tensor_local = self.sche.cache_read(tensor_shared, "local", consumers)
            self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)
        return self.sche

    def _rewrite_schedule_default(self):
        assert(self.reduce_op is not None)
        out = self.output_op
        self.thread_per_block = [int(np.prod(self.config.thread)), 1, 1]
        reg_tile = self.sche.cache_write(self.reduce_op.output(0), "local")

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for i, axis in enumerate(self.sche[out].op.axis):
            bx, _t = self.sche[out].split(axis, factor=self.config.block[i])
            vx, _t = self.sche[out].split(_t, factor=self.config.thread[i] * self.config.step[i])
            tx, tn = self.sche[out].split(_t, factor=self.config.step[i])
            blck_axis.append(bx)
            vthd_axis.append(vx)
            thrd_axis.append(tx)
            tile_axis.append(tn)
        vthd_axis = list(reversed(vthd_axis)) # inner virtual thread first
        axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis
        self.sche[out].reorder(*axis_order)
        blck_fused = self.sche[out].fuse(*blck_axis)
        thrd_fused = self.sche[out].fuse(*thrd_axis)
        self.sche[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
        for va in vthd_axis:
            self.sche[out].bind(va, te.thread_axis("vthread"))
        self.sche[out].bind(thrd_fused, te.thread_axis("threadIdx.x"))
        for tn in tile_axis:
            self.sche[out].unroll(tn)

        self.sche[reg_tile].compute_at(self.sche[out], thrd_fused)

        reduce_outer_axis, reduce_inner_axis = [], []
        space_axis = list(self.sche[reg_tile].op.axis)

        for i in self.config.raxis_order:
            axis = self.sche[reg_tile].op.reduce_axis[i]
            ro, ri = self.sche[reg_tile].split(axis, factor=self.config.rstep[i])
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)
        axis_order = reduce_outer_axis + reduce_inner_axis + space_axis
        self.sche[reg_tile].reorder(*axis_order)
        space_fused = self.sche[reg_tile].fuse(*space_axis)
        self.sche[reg_tile].unroll(space_fused)

        for input_tensor in self.reduce_op.input_tensors:
            shared_tensor = self.sche.cache_read(input_tensor, "shared", [reg_tile])
            if input_tensor in self.shared_inputs:
                self.sche[shared_tensor].compute_at(self.sche[out], blck_fused)
                strides = self.shared_inputs_strides[input_tensor]
            else:
                self.sche[shared_tensor].compute_at(self.sche[reg_tile], reduce_outer_axis[-1])
                strides = Stride()
            if input_tensor.name in self.config.vectorize and not self._is_from_shared(input_tensor):
                vectorize = self.config.vectorize[input_tensor.name]
            else:
                vectorize = 1
            self.cooperative_fetch(shared_tensor, self.sche, strides, vectorize)

        cache_plan = {}
        for op in self.elementwise_ops:
            for tensor in op.input_tensors:
                if self.requires_cache(tensor, op):
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = self.sche.cache_read(tensor, "shared", consumers)
            self.sche[tensor_shared].compute_at(self.sche[out], thrd_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            self.cooperative_fetch(tensor_shared, self.sche, strides)
            # This is a hack, TVM cannot handle cached_local_read when padding on a shared input
            consumers = list(filter(lambda x: x.output(0) not in self.reduce_op.input_tensors, consumers))
            if len(consumers) == 0 or len(self.shared_outputs) == 0: continue
            tensor_local = self.sche.cache_read(tensor_shared, "local", consumers)
            self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)

        return self.sche

    def _rewrite_schedule_inter_thread(self):
        assert(self.reduce_op is not None)
        out = self.output_op
        reg_tile = self.reduce_op.output(0)
        self.thread_per_block = [int(np.prod(self.config.reduce_thread)), int(np.prod(self.config.thread)), 1]
        # For inter thread reduction case, one thread must only compute one element
        assert self.config.thread == self.config.block

        blck_axis = []
        thrd_axis = []
        for i, axis in enumerate(self.sche[out].op.axis):
            bx, tx = self.sche[out].split(axis, factor=self.config.block[i])
            blck_axis.append(bx)
            thrd_axis.append(tx)
        axis_order = blck_axis + thrd_axis
        self.sche[out].reorder(*axis_order)
        blck_fused = self.sche[out].fuse(*blck_axis)
        thrd_fused = self.sche[out].fuse(*thrd_axis)
        self.sche[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
        self.sche[out].bind(thrd_fused, te.thread_axis("threadIdx.y"))
        if out is not self.reduce_op:
            self.sche[reg_tile].compute_at(self.sche[out], thrd_fused)

        reduce_outer_axis, reduce_inner_axis, reduce_inter_threads = [], [], []

        for i in self.config.raxis_order:
            axis = self.sche[reg_tile].op.reduce_axis[i]
            ro, _t = self.sche[reg_tile].split(axis, factor=self.config.rstep[i])
            ri, thd = self.sche[reg_tile].split(_t, factor=self.config.reduce_thread[i])
            reduce_inter_threads.append(thd)
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)
        axis_order = reduce_inter_threads + reduce_outer_axis + reduce_inner_axis
        self.sche[reg_tile].reorder(*axis_order)
        fused_reduce_inter_threads = self.sche[reg_tile].fuse(*reduce_inter_threads)
        self.sche[reg_tile].bind(fused_reduce_inter_threads, te.thread_axis("threadIdx.x"))

        for input_tensor in self.reduce_op.input_tensors:
            shared_tensor = self.sche.cache_read(input_tensor, "shared", [reg_tile])
            if input_tensor in self.shared_inputs:
                self.sche[shared_tensor].compute_at(self.sche[out], blck_fused)
                strides = self.shared_inputs_strides[input_tensor]
            else:
                self.sche[shared_tensor].compute_at(self.sche[reg_tile], reduce_outer_axis[-1])
                strides = Stride()
            if input_tensor.name in self.config.vectorize and not self._is_from_shared(input_tensor):
                vectorize = self.config.vectorize[input_tensor.name]
            else:
                vectorize = 1
            self.cooperative_fetch(shared_tensor, self.sche, strides, vectorize)

        cache_plan = {}
        for op in self.elementwise_ops:
            for tensor in op.input_tensors:
                if self.requires_cache(tensor, op):
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = self.sche.cache_read(tensor, "shared", consumers)
            self.sche[tensor_shared].compute_at(self.sche[out], thrd_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            self.cooperative_fetch(tensor_shared, self.sche, strides)
            # This is a hack, TVM cannot handle cached_local_read when padding on a shared input
            consumers = list(filter(lambda x: x.output(0) not in self.reduce_op.input_tensors, consumers))
            if len(consumers) == 0 or len(self.shared_outputs) == 0: continue
            tensor_local = self.sche.cache_read(tensor_shared, "local", consumers)
            self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)

        return self.sche

    # check whether a tensor is connected from shared inputs
    def _is_from_shared(self, tensor):
        if tensor in self.shared_inputs:
            return True
        return any(map(self._is_from_shared, tensor.op.input_tensors))

    def _rewrite_schedule_tc(self):
        assert(self.reduce_op is not None)
        out = self.output_op
        use_global = len(self.shared_outputs) == 0 and self.reduce_op == self.output_op
        # use_global = False
        assert (len(self.reduce_op.input_tensors) == 2)
        A, B = self.reduce_op.input_tensors
        C = self.reduce_op.output(0)
        AS = self.sche.cache_read(A, "shared", [C])
        BS = self.sche.cache_read(B, "shared", [C])
        AF = self.sche.cache_read(AS, "wmma.matrix_a", [C])
        BF = self.sche.cache_read(BS, "wmma.matrix_b", [C])
        CF = self.sche.cache_write(C, "wmma.accumulator")
        if use_global:
            CS = C
        else:
            CS = self.sche.cache_read(CF, "shared", [C])

        wmma_m, wmma_n, wmma_k = self.config.wmma
        assert (wmma_m, wmma_n, wmma_k) in [(16, 16, 16), (8, 32, 16), (32, 8, 16)]
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = self.config.tc_extra_conf.tc_axis
        A_ndim, B_ndim, C_ndim = len(A.shape), len(B.shape), len(C.shape)
        offset = 8
        AL_shape = [1 for _ in range(A_ndim)]
        AL_shape[A_ax_m] = wmma_m
        AL_shape[A_ax_k] = wmma_k
        BL_shape = [1 for _ in range(B_ndim)]
        BL_shape[B_ax_k] = wmma_k
        BL_shape[B_ax_n] = wmma_n
        CL_shape = [1 for _ in range(C_ndim)]
        CL_shape[C_ax_m] = wmma_m
        CL_shape[C_ax_n] = wmma_n
        C_high_ax = min(C_ax_m, C_ax_n)
        if use_global:
            CSstrideDef = Stride()
            CS_stride = CSstrideDef.compute_strides_from_shape(C.shape)
        else:
            CSstrideDef = Stride(int(np.prod(self.config.block[C_high_ax+1:])) + offset, C_high_ax)
            CS_stride = CSstrideDef.compute_strides_from_shape(self.config.block)
        A_high_ax = min(A_ax_m, A_ax_k)
        AS_shape = self.config.tc_extra_conf.AS_shape
        if A in self.shared_inputs:
            AS_shape[A_ax_k] = int(C.op.reduce_axis[0].dom.extent)
        ASstrideDef = Stride(int(np.prod(AS_shape[A_high_ax+1:])) + offset, A_high_ax)
        B_high_ax = min(B_ax_n, B_ax_k)
        BS_shape = self.config.tc_extra_conf.BS_shape
        if B in self.shared_inputs:
            BS_shape[B_ax_k] = int(C.op.reduce_axis[0].dom.extent)
        BSstrideDef = Stride(int(np.prod(BS_shape[B_high_ax+1:])) + offset, B_high_ax)
        AS_stride = ASstrideDef.compute_strides_from_shape(AS_shape)
        BS_stride = BSstrideDef.compute_strides_from_shape(BS_shape)
        AF_stride = [te.var() for _ in range(A_ndim)]
        BF_stride = [te.var() for _ in range(B_ndim)]
        CF_stride = [te.var() for _ in range(C_ndim)]

        if A in self.shared_inputs:
            self.config.tc_extra_conf.AS_shape[A_ax_k] = int(C.op.reduce_axis[-1].dom.extent) + offset

        self.thread_per_block = [32, 1, 1]
        for blk, warp in zip(self.config.block, self.config.warp):
            assert blk % warp == 0
            self.thread_per_block[1] *= (blk // warp)

        if use_global:
            blck_axis = []
            warp_axis = []
            CS_outer_axis = []
            CS_inner_axis = []
            for i, axis in enumerate(self.sche[out].op.axis):
                bx, _t = self.sche[out].split(axis, factor=self.config.block[i])
                wt, _t = self.sche[out].split(_t, factor=self.config.warp[i])
                ot, it = self.sche[out].split(_t, factor=CL_shape[i])
                blck_axis.append(bx)
                warp_axis.append(wt)
                CS_outer_axis.append(ot)
                CS_inner_axis.append(it)
            axis_order = blck_axis + warp_axis + CS_outer_axis + CS_inner_axis
            self.sche[out].reorder(*axis_order)
            blck_fused = self.sche[out].fuse(*blck_axis)
            warp_fused = self.sche[out].fuse(*warp_axis)
            self.sche[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
            self.sche[out].bind(warp_fused, te.thread_axis("threadIdx.y"))
        else:
            # schedule for output stage
            blck_axis = []
            thrd_axis = []
            for i, axis in enumerate(self.sche[out].op.axis):
                bx, tx = self.sche[out].split(axis, factor=self.config.block[i])
                blck_axis.append(bx)
                thrd_axis.append(tx)
            axis_order = blck_axis + thrd_axis
            self.sche[out].reorder(*axis_order)
            blck_fused = self.sche[out].fuse(*blck_axis)
            thrd_fused = self.sche[out].fuse(*thrd_axis)
            thrd_fused, tv = self.sche[out].split(thrd_fused, factor=8)
            _t, tx = self.sche[out].split(thrd_fused, factor=self.thread_per_block[0])
            _t, ty = self.sche[out].split(_t, factor=self.thread_per_block[1])
            self.sche[out].vectorize(tv)

            self.sche[out].bind(ty, te.thread_axis("threadIdx.y"))
            self.sche[out].bind(tx, te.thread_axis("threadIdx.x"))
            self.sche[out].bind(blck_fused, te.thread_axis("blockIdx.x"))

            # schedule for block
            self.sche[CS].compute_at(self.sche[out], blck_fused)
            self.sche[CS].storage_align(CS.op.axis[CSstrideDef.ax], CSstrideDef.stride - 1, CSstrideDef.stride)
            warp_axis = []
            CS_outer_axis = []
            CS_inner_axis = []
            for i, ax in enumerate(CS.op.axis):
                wt, _t = self.sche[CS].split(ax, factor=self.config.warp[i])
                ot, it = self.sche[CS].split(_t, factor=CL_shape[i])
                warp_axis.append(wt)
                CS_outer_axis.append(ot)
                CS_inner_axis.append(it)
            self.sche[CS].reorder(*warp_axis, *CS_outer_axis, *CS_inner_axis)
            warp_fused = self.sche[CS].fuse(*warp_axis)
            self.sche[CS].bind(warp_fused, te.thread_axis("threadIdx.y"))

        # Schedule for wmma computation
        self.sche[CF].compute_at(self.sche[CS], warp_fused)
        CF_outer_axis = []
        CF_inner_axis = []
        for i, ax in enumerate(CF.op.axis):
            ot, it = self.sche[CF].split(ax, factor=CL_shape[i])
            CF_outer_axis.append(ot)
            CF_inner_axis.append(it)
        ko, _k = self.sche[CF].split(CF.op.reduce_axis[0], factor=self.config.rstep[0])
        ki, _k = self.sche[CF].split(_k, factor=wmma_k)
        self.sche[CF].reorder(ko, ki, *CF_outer_axis, *CF_inner_axis, _k)

        # Schedule for  wmma_matrix_a load
        self.sche[AF].compute_at(self.sche[CF], ki)
        AF_outer_axis = []
        AF_inner_axis = []
        for i, ax in enumerate(AF.op.axis):
            ot, it = self.sche[AF].split(ax, factor=AL_shape[i])
            AF_outer_axis.append(ot)
            AF_inner_axis.append(it)
        self.sche[AF].reorder(*AF_outer_axis, *AF_inner_axis)

        # Schedule for  wmma_matrix_b load
        self.sche[BF].compute_at(self.sche[CF], ki)
        BF_outer_axis = []
        BF_inner_axis = []
        for i, ax in enumerate(BF.op.axis):
            ot, it = self.sche[BF].split(ax, factor=BL_shape[i])
            BF_outer_axis.append(ot)
            BF_inner_axis.append(it)
        self.sche[BF].reorder(*BF_outer_axis, *BF_inner_axis)

        # schedule shared
        if A in self.shared_inputs:
            self.sche[AS].compute_at(self.sche[out], blck_fused)
        else:
            self.sche[AS].compute_at(self.sche[CF], ko)
        if B in self.shared_inputs:
            self.sche[BS].compute_at(self.sche[out], blck_fused)
        else:
            self.sche[BS].compute_at(self.sche[CF], ko)
        # TVM sometimes errors when vectorize=8 for some inlining
        vectorize_A = 8 if isinstance(A.op, te.PlaceholderOp) else 4
        vectorize_B = 8 if isinstance(B.op, te.PlaceholderOp) else 4
        if self._is_from_shared(A): vectorize_A = 1
        if self._is_from_shared(B): vectorize_B = 1
        self.cooperative_fetch(AS, self.sche, ASstrideDef, vectorize_A)
        self.cooperative_fetch(BS, self.sche, BSstrideDef, vectorize_B)

        shape = (wmma_m, wmma_n, wmma_k)
        AL_gemm = te.placeholder(AL_shape, name="AL_gemm", dtype=A.dtype)
        BL_gemm = te.placeholder(BL_shape, name="BL_gemm", dtype=B.dtype)
        k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
        def CLfcompute(*args):
            A_slice = [0 for _ in range(A_ndim)]
            A_slice[A_ax_k] = k_gemm
            A_slice[A_ax_m] = args[C_ax_m]
            B_slice = [0 for _ in range(B_ndim)]
            B_slice[B_ax_k] = k_gemm
            B_slice[B_ax_n] = args[C_ax_n]
            return  te.sum(
                AL_gemm.__getitem__(tuple(A_slice)).astype(C.dtype) * BL_gemm.__getitem__(tuple(B_slice)).astype(C.dtype),
                axis=k_gemm,
            )
        CL_compute = te.compute(CL_shape, CLfcompute, name="CL_compute",)

        from .tc_intrin import (intrin_wmma_gemm, intrin_wmma_load_matrix_A,
                                intrin_wmma_load_matrix_W,
                                intrin_wmma_store_matrix)
        AF_layout = "row_major" if A_ax_m < A_ax_k else "col_major"
        self.sche[AF].tensorize(
            AF_inner_axis[0],
            intrin_wmma_load_matrix_A(
                AF_stride, AS_stride, shape, AF_layout, AL_shape, AL_shape, A.dtype
            ),
        )
        BF_layout = "row_major" if B_ax_k < B_ax_n else "col_major"
        self.sche[BF].tensorize(
            BF_inner_axis[0],
            intrin_wmma_load_matrix_W(
                BF_stride, BS_stride, shape, BF_layout, BL_shape, BL_shape, B.dtype
            ),
        )
        self.sche[CF].tensorize(
            CF_inner_axis[0], intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape)
        )
        self.sche[CS].tensorize(
            CS_inner_axis[0],
            intrin_wmma_store_matrix(
                CS_stride, CF_stride, shape, C.dtype, CL_shape, CL_shape, "global" if use_global else "shared"
            ),
        )

        cache_plan = {}
        for op in self.elementwise_ops:
            for tensor in op.input_tensors:
                if tensor in self.shared_inputs:
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = self.sche.cache_read(tensor, "shared", consumers)
            self.sche[tensor_shared].compute_at(self.sche[out], blck_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            self.cooperative_fetch(tensor_shared, self.sche, strides)
            # This is a hack, TVM cannot handle cached_local_read when padding on a shared input
            consumers = list(filter(lambda x: x.output(0) not in self.reduce_op.input_tensors, consumers))
            if len(consumers) == 0 or len(self.shared_outputs) == 0: continue
            tensor_local = self.sche.cache_read(tensor_shared, "local", consumers)
            self.sche[tensor_local].compute_at(self.sche[out], tx)

        return self.sche
