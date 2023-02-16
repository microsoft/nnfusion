import numpy as np
from tvm import te

from ..config import Config, Stride
from .te_base import TESchedulerBase


class TEWarpMMAScheduler(TESchedulerBase):
    def schedule(self) -> te.Schedule:
        sch, config = self.sche, self.config
        for op in self.ops:
            if op is not self.output_op:
                sch[op].compute_inline()
        out = self.output_op
        use_global = len(self.shared_outputs) == 0 and self.reduce_op == self.output_op
        # use_global = False
        assert (len(self.reduce_op.input_tensors) == 2)
        A, B = self.reduce_op.input_tensors
        C = self.reduce_op.output(0)
        AS = sch.cache_read(A, "shared", [C])
        BS = sch.cache_read(B, "shared", [C])
        AF = sch.cache_read(AS, "wmma.matrix_a", [C])
        BF = sch.cache_read(BS, "wmma.matrix_b", [C])
        CF = sch.cache_write(C, "wmma.accumulator")
        if use_global:
            CS = C
        else:
            CS = sch.cache_read(CF, "shared", [C])

        wmma_m, wmma_n, wmma_k = config.wmma
        assert (wmma_m, wmma_n, wmma_k) in [(16, 16, 16), (8, 32, 16), (32, 8, 16)]
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
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
            CSstrideDef = Stride(int(np.prod(config.block[C_high_ax+1:])) + offset, C_high_ax)
            CS_stride = CSstrideDef.compute_strides_from_shape(config.block)
        A_high_ax = min(A_ax_m, A_ax_k)
        AS_shape = config.tc_extra_conf.AS_shape
        if A in self.shared_inputs:
            AS_shape[A_ax_k] = int(C.op.reduce_axis[0].dom.extent)
        ASstrideDef = Stride(int(np.prod(AS_shape[A_high_ax+1:])) + offset, A_high_ax)
        B_high_ax = min(B_ax_n, B_ax_k)
        BS_shape = config.tc_extra_conf.BS_shape
        if B in self.shared_inputs:
            BS_shape[B_ax_k] = int(C.op.reduce_axis[0].dom.extent)
        BSstrideDef = Stride(int(np.prod(BS_shape[B_high_ax+1:])) + offset, B_high_ax)
        AS_stride = ASstrideDef.compute_strides_from_shape(AS_shape)
        BS_stride = BSstrideDef.compute_strides_from_shape(BS_shape)
        AF_stride = [te.var() for _ in range(A_ndim)]
        BF_stride = [te.var() for _ in range(B_ndim)]
        CF_stride = [te.var() for _ in range(C_ndim)]

        if A in self.shared_inputs:
            config.tc_extra_conf.AS_shape[A_ax_k] = int(C.op.reduce_axis[-1].dom.extent) + offset

        self.block_size[0] = 32
        for blk, warp in zip(config.block, config.warp):
            assert blk % warp == 0
            self.block_size[1] *= (blk // warp)

        if use_global:
            blck_axis = []
            warp_axis = []
            CS_outer_axis = []
            CS_inner_axis = []
            for i, axis in enumerate(sch[out].op.axis):
                bx, _t = sch[out].split(axis, factor=config.block[i])
                wt, _t = sch[out].split(_t, factor=config.warp[i])
                ot, it = sch[out].split(_t, factor=CL_shape[i])
                blck_axis.append(bx)
                warp_axis.append(wt)
                CS_outer_axis.append(ot)
                CS_inner_axis.append(it)
            axis_order = blck_axis + warp_axis + CS_outer_axis + CS_inner_axis
            sch[out].reorder(*axis_order)
            blck_fused = sch[out].fuse(*blck_axis)
            warp_fused = sch[out].fuse(*warp_axis)
            sch[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
            sch[out].bind(warp_fused, te.thread_axis("threadIdx.y"))
        else:
            # schedule for output stage
            blck_axis = []
            thrd_axis = []
            for i, axis in enumerate(sch[out].op.axis):
                bx, tx = sch[out].split(axis, factor=config.block[i])
                blck_axis.append(bx)
                thrd_axis.append(tx)
            axis_order = blck_axis + thrd_axis
            sch[out].reorder(*axis_order)
            blck_fused = sch[out].fuse(*blck_axis)
            thrd_fused = sch[out].fuse(*thrd_axis)
            thrd_fused, tv = sch[out].split(thrd_fused, factor=8)
            _t, tx = sch[out].split(thrd_fused, factor=self.block_size[0])
            _t, ty = sch[out].split(_t, factor=self.block_size[1])
            sch[out].vectorize(tv)

            sch[out].bind(ty, te.thread_axis("threadIdx.y"))
            sch[out].bind(tx, te.thread_axis("threadIdx.x"))
            sch[out].bind(blck_fused, te.thread_axis("blockIdx.x"))

            # schedule for block
            sch[CS].compute_at(sch[out], blck_fused)
            sch[CS].storage_align(CS.op.axis[CSstrideDef.ax], CSstrideDef.stride - 1, CSstrideDef.stride)
            warp_axis = []
            CS_outer_axis = []
            CS_inner_axis = []
            for i, ax in enumerate(CS.op.axis):
                wt, _t = sch[CS].split(ax, factor=config.warp[i])
                ot, it = sch[CS].split(_t, factor=CL_shape[i])
                warp_axis.append(wt)
                CS_outer_axis.append(ot)
                CS_inner_axis.append(it)
            sch[CS].reorder(*warp_axis, *CS_outer_axis, *CS_inner_axis)
            warp_fused = sch[CS].fuse(*warp_axis)
            sch[CS].bind(warp_fused, te.thread_axis("threadIdx.y"))

        # Schedule for wmma computation
        sch[CF].compute_at(sch[CS], warp_fused)
        CF_outer_axis = []
        CF_inner_axis = []
        for i, ax in enumerate(CF.op.axis):
            ot, it = sch[CF].split(ax, factor=CL_shape[i])
            CF_outer_axis.append(ot)
            CF_inner_axis.append(it)
        ko, _k = sch[CF].split(CF.op.reduce_axis[0], factor=config.rstep[0])
        ki, _k = sch[CF].split(_k, factor=wmma_k)
        sch[CF].reorder(ko, ki, *CF_outer_axis, *CF_inner_axis, _k)

        # Schedule for  wmma_matrix_a load
        sch[AF].compute_at(sch[CF], ki)
        AF_outer_axis = []
        AF_inner_axis = []
        for i, ax in enumerate(AF.op.axis):
            ot, it = sch[AF].split(ax, factor=AL_shape[i])
            AF_outer_axis.append(ot)
            AF_inner_axis.append(it)
        sch[AF].reorder(*AF_outer_axis, *AF_inner_axis)

        # Schedule for  wmma_matrix_b load
        sch[BF].compute_at(sch[CF], ki)
        BF_outer_axis = []
        BF_inner_axis = []
        for i, ax in enumerate(BF.op.axis):
            ot, it = sch[BF].split(ax, factor=BL_shape[i])
            BF_outer_axis.append(ot)
            BF_inner_axis.append(it)
        sch[BF].reorder(*BF_outer_axis, *BF_inner_axis)

        # schedule shared
        if A in self.shared_inputs:
            sch[AS].compute_at(sch[out], blck_fused)
        else:
            sch[AS].compute_at(sch[CF], ko)
        if B in self.shared_inputs:
            sch[BS].compute_at(sch[out], blck_fused)
        else:
            sch[BS].compute_at(sch[CF], ko)
        # TVM sometimes errors when vectorize=8 for some inlining
        vectorize_A = 8 if isinstance(A.op, te.PlaceholderOp) else 4
        vectorize_B = 8 if isinstance(B.op, te.PlaceholderOp) else 4
        if self._is_from_shared(A): vectorize_A = 1
        if self._is_from_shared(B): vectorize_B = 1
        self.cooperative_fetch(AS, ASstrideDef, vectorize_A)
        self.cooperative_fetch(BS, BSstrideDef, vectorize_B)

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

        from .wmma_intrin import (intrin_wmma_gemm, intrin_wmma_load_matrix_A,
                                  intrin_wmma_load_matrix_W,
                                  intrin_wmma_store_matrix)
        AF_layout = "row_major" if A_ax_m < A_ax_k else "col_major"
        sch[AF].tensorize(
            AF_inner_axis[0],
            intrin_wmma_load_matrix_A(
                AF_stride, AS_stride, shape, AF_layout, AL_shape, AL_shape, A.dtype
            ),
        )
        BF_layout = "row_major" if B_ax_k < B_ax_n else "col_major"
        sch[BF].tensorize(
            BF_inner_axis[0],
            intrin_wmma_load_matrix_W(
                BF_stride, BS_stride, shape, BF_layout, BL_shape, BL_shape, B.dtype
            ),
        )
        sch[CF].tensorize(
            CF_inner_axis[0], intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape, AF_layout, BF_layout)
        )
        sch[CS].tensorize(
            CS_inner_axis[0],
            intrin_wmma_store_matrix(
                CS_stride, CF_stride, shape, C.dtype, CL_shape, CL_shape, "global" if use_global else "shared"
            ),
        )

        cache_plan = {}
        for op in self.none_reduce_ops:
            for tensor in op.input_tensors:
                if tensor in self.shared_inputs:
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = sch.cache_read(tensor, "shared", consumers)
            sch[tensor_shared].compute_at(sch[out], blck_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            self.cooperative_fetch(tensor_shared, strides)
            # This is a hack, TVM cannot handle cached_local_read when padding on a shared input
            consumers = list(filter(lambda x: x.output(0) not in self.reduce_op.input_tensors, consumers))
            if len(consumers) == 0 or len(self.shared_outputs) == 0: continue
            tensor_local = sch.cache_read(tensor_shared, "local", consumers)
            sch[tensor_local].compute_at(sch[out], tx)

        return sch
