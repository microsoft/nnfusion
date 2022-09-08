import tvm
from tvm import te
import numpy as np
from typing import Dict, List, Any

class Scheduler:
    def __init__(self):
        pass

    def split_axis(self, op, axis):
        ret = []
        factors = self.tiling[axis.var.name]
        for i in range(0, len(factors)):
            ax0, ax1 = self.sche[op].split(axis, factor=int(np.prod(factors[i:])))
            ret.append(ax0)
            axis = ax1
        return ret + [axis]

    def cooperative_fetch(self, shared, sch):
        assert isinstance(self.thread_per_block, int)
        axes = sch[shared].op.axis
        fused = sch[shared].fuse(*axes)
        oo, ii = sch[shared].split(fused, factor=self.thread_per_block)
        sch[shared].reorder(oo, ii)
        sch[shared].unroll(oo)
        sch[shared].bind(ii, te.thread_axis("threadIdx.x"))

    def cooperative_fetch_2d(self, shared, sch, strides=None):
        assert len(self.thread_per_block) == 2
        axes = sch[shared].op.axis
        if strides is not None:
            sch[shared].storage_align(axes[-2], strides - 1, strides)
        fused = sch[shared].fuse(*axes)
        oo, _temp = sch[shared].split(fused, factor=self.thread_per_block[0] * self.thread_per_block[1])
        inner_y, inner_x = sch[shared].split(_temp, factor=self.thread_per_block[0])
        sch[shared].reorder(oo, inner_y, inner_x)
        sch[shared].unroll(oo)
        sch[shared].bind(inner_y, te.thread_axis("threadIdx.y"))
        sch[shared].bind(inner_x, te.thread_axis("threadIdx.x"))

    # [Parameters]
    #   schedule: the original TVM schedule of an op
    #   tile_dict: a dictionary holding split factors of each axis,
    #              e.g., {"i": [8, 16, 1], "j": [8, 16, 1], "k": [32]}.
    #              For spacial axes, the format is "axis_name": [thread_tile_size, thread_num, 1].
    #              For reduce axes, the format is "axis_name": [step_size, thread_num].
    # [Return]
    #   new_s: an optimized TVM schedule
    def rewrite_schedule(self, schedule: te.Schedule, tile_dict: Dict[str, Any], shared_inputs: List[str] = []):
        self.tiling = tile_dict.copy()
        self.sche = schedule
        self.shared_inputs = shared_inputs

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
            if "use_tc" in self.tiling and self.tiling["use_tc"] is True:
                return self.rewrite_schedule_tc()
            has_inter_thread_reduce = False
            for ax in self.reduce_op.reduce_axis:
                if self.tiling[str(ax.var.name)][1] > 1:
                    has_inter_thread_reduce = True
                    break
            if has_inter_thread_reduce:
                return self.rewrite_schedule_inter_thread()
            else:
                return self.recursive_schedule_up()
        else:
            return self.rewrite_schedule_no_reduce()

    def rewrite_schedule_no_reduce(self):
        assert(self.reduce_op is None)
        out = self.output_op
        self.thread_per_block = 1
        for axis in self.sche[out].op.axis:
            name = axis.var.name
            if len(self.tiling[name]) == 2:
                vthrd = self.tiling[name][1]
                thrd = self.tiling[name][0]
                self.tiling[name] = [vthrd, thrd, 1]
            self.thread_per_block *= self.tiling[name][1]

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for axis in self.sche[out].op.axis:
            bx, vx, tx, tn = self.split_axis(out, axis)
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

        cache_plan = {}
        for op in self.elementwise_ops:
            for tensor in op.input_tensors:
                cache = isinstance(self.sche[tensor].op, tvm.te.PlaceholderOp) \
                    and len(op.output(0).shape) > len(tensor.shape) \
                    and np.prod(op.output(0).shape) > np.prod(tensor.shape) # is broadcast
                if tensor.name in self.shared_inputs:
                    cache = True
                if cache:
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = self.sche.cache_read(tensor, "shared", consumers)
            self.sche[tensor_shared].compute_at(self.sche[out], thrd_fused)
            self.cooperative_fetch(tensor_shared, self.sche)
            tensor_local = self.sche.cache_read(tensor_shared, "local", consumers)
            self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)
        return self.sche

    def recursive_schedule_up(self):
        assert(self.reduce_op is not None)
        out = self.output_op

        self.thread_per_block = 1
        for axis in self.sche[out].op.axis:
            name = axis.var.name
            if len(self.tiling[name]) == 2:
                vthrd = self.tiling[name][1]
                thrd = self.tiling[name][0]
                self.tiling[name] = [vthrd, thrd, 1]
            self.thread_per_block *= self.tiling[name][1]

        reg_tile = self.sche.cache_write(self.reduce_op.output(0), "local")

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for axis in self.sche[out].op.axis:
            bx, vx, tx, tn = self.split_axis(out, axis)
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
        self.sche[reg_tile].compute_at(self.sche[out], thrd_fused)

        reduce_outer_axis, reduce_inner_axis = [], []
        space_axis = list(self.sche[reg_tile].op.axis)

        # reorder reduce axis
        if "raxis_order" not in self.tiling:
            ordered_reduce_axis = self.sche[reg_tile].op.reduce_axis
        else:
            ordered_reduce_axis = []
            for axis_name in self.tiling["raxis_order"]:
                for axis in self.sche[reg_tile].op.reduce_axis:
                    if str(axis.var.name) == axis_name:
                        ordered_reduce_axis.append(axis)

        for axis in ordered_reduce_axis:
            if axis.var.name not in self.tiling:
                ro, ri = self.sche[reg_tile].split(axis, nparts=1)
            else:
                factor = self.tiling[axis.var.name][0]
                ro, ri = self.sche[reg_tile].split(axis, factor=factor)
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)
        axis_order = reduce_outer_axis + reduce_inner_axis + space_axis
        self.sche[reg_tile].reorder(*axis_order)
        space_fused = self.sche[reg_tile].fuse(*space_axis)
        self.sche[reg_tile].unroll(space_fused)
        if "unroll" in self.tiling:
            for ax in reduce_inner_axis:
                self.sche[reg_tile].unroll(ax)

        for input_tensor in self.reduce_op.input_tensors:
            shared_tensor = self.sche.cache_read(input_tensor, "shared", [reg_tile])
            # local_tensor = self.sche.cache_read(shared_tensor, "local", [reg_tile])
            # self.sche[local_tensor].compute_at(self.sche[reg_tile], space_fused)
            if input_tensor.name in self.shared_inputs:
                self.sche[shared_tensor].compute_at(self.sche[out], blck_fused)
            else:
                self.sche[shared_tensor].compute_at(self.sche[reg_tile], reduce_outer_axis[-1])
            self.cooperative_fetch(shared_tensor, self.sche)

        cache_plan = {}
        for op in self.elementwise_ops:
            for tensor in op.input_tensors:
                cache = isinstance(self.sche[tensor].op, tvm.te.PlaceholderOp) \
                    and len(op.output(0).shape) > len(tensor.shape) \
                    and np.prod(op.output(0).shape) > np.prod(tensor.shape) # is broadcast
                if tensor.name in self.shared_inputs:
                    cache = True
                if cache:
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = self.sche.cache_read(tensor, "shared", consumers)
            self.sche[tensor_shared].compute_at(self.sche[out], thrd_fused)
            self.cooperative_fetch(tensor_shared, self.sche)
            # This is a hack, TVM cannot handle cached_local_read when padding on a shared input
            consumers = list(filter(lambda x: x.output(0) not in self.reduce_op.input_tensors, consumers))
            if len(consumers) == 0:continue
            tensor_local = self.sche.cache_read(tensor_shared, "local", consumers)
            self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)

        return self.sche

    def rewrite_schedule_inter_thread(self):
        assert(self.reduce_op is not None)
        out = self.output_op
        reg_tile = self.reduce_op.output(0)

        self.thread_per_block = [1, 1]
        for axis in self.sche[out].op.axis:
            name = axis.var.name
            if len(self.tiling[name]) == 2:
                assert self.tiling[name][1] == 1, "Virtual thread is not possible in inter-thread reduction"
                self.tiling[name] = [self.tiling[name][0]]
            elif len(self.tiling[name]) == 3:
                assert self.tiling[name][0] == 1, "Virtual thread is not possible in inter-thread reduction"
                assert self.tiling[name][2] == 1, "Virtual thread is not possible in inter-thread reduction"
                self.tiling[name] = [self.tiling[name][1]]
            else:
                raise ValueError(self.tiling)
            self.thread_per_block[1] *= self.tiling[name][0]

        for axis in self.sche[reg_tile].op.reduce_axis:
            name = axis.var.name
            if len(self.tiling[name]) == 2:
                self.thread_per_block[0] *= self.tiling[name][1]
            else:
                raise ValueError(self.tiling)

        blck_axis = []
        thrd_axis = []
        for axis in self.sche[out].op.axis:
            bx, tx = self.split_axis(out, axis)
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

        # reorder reduce axis
        if "raxis_order" not in self.tiling:
            ordered_reduce_axis = self.sche[reg_tile].op.reduce_axis
        else:
            ordered_reduce_axis = []
            for axis_name in self.tiling["raxis_order"]:
                for axis in self.sche[reg_tile].op.reduce_axis:
                    if str(axis.var.name) == axis_name:
                        ordered_reduce_axis.append(axis)

        for axis in ordered_reduce_axis:
            ro, ri, thd = self.split_axis(reg_tile, axis)
            reduce_inter_threads.append(thd)
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)
        axis_order = reduce_inter_threads + reduce_outer_axis + reduce_inner_axis
        self.sche[reg_tile].reorder(*axis_order)
        fused_reduce_inter_threads = self.sche[reg_tile].fuse(*reduce_inter_threads)
        self.sche[reg_tile].bind(fused_reduce_inter_threads, te.thread_axis("threadIdx.x"))

        for input_tensor in self.reduce_op.input_tensors:
            shared_tensor = self.sche.cache_read(input_tensor, "shared", [reg_tile])
            if input_tensor.name in self.shared_inputs:
                self.sche[shared_tensor].compute_at(self.sche[out], blck_fused)
            else:
                self.sche[shared_tensor].compute_at(self.sche[reg_tile], reduce_outer_axis[-1])
            self.cooperative_fetch_2d(shared_tensor, self.sche)

        cache_plan = {}
        for op in self.elementwise_ops:
            for tensor in op.input_tensors:
                cache = isinstance(self.sche[tensor].op, tvm.te.PlaceholderOp) \
                    and len(op.output(0).shape) > len(tensor.shape) \
                    and np.prod(op.output(0).shape) > np.prod(tensor.shape) # is broadcast
                if tensor.name in self.shared_inputs:
                    cache = True
                if cache:
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = self.sche.cache_read(tensor, "shared", consumers)
            self.sche[tensor_shared].compute_at(self.sche[out], thrd_fused)
            self.cooperative_fetch_2d(tensor_shared, self.sche)
            # This is a hack, TVM cannot handle cached_local_read when padding on a shared input
            consumers = list(filter(lambda x: x.output(0) not in self.reduce_op.input_tensors, consumers))
            if len(consumers) == 0:continue
            tensor_local = self.sche.cache_read(tensor_shared, "local", consumers)
            self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)

        return self.sche

    def rewrite_schedule_tc(self):
        assert(self.reduce_op is not None)
        out = self.output_op

        assert (len(self.reduce_op.input_tensors) == 2)
        A, B = self.reduce_op.input_tensors
        C = self.reduce_op.output(0)
        AS = self.sche.cache_read(A, "shared", [C])
        BS = self.sche.cache_read(B, "shared", [C])
        AF = self.sche.cache_read(AS, "wmma.matrix_a", [C])
        BF = self.sche.cache_read(BS, "wmma.matrix_b", [C])
        CF = self.sche.cache_write(C, "wmma.accumulator")
        CS = self.sche.cache_read(CF, "shared", [C])

        ax_M = C.op.axis[-2].var.name
        ax_N = C.op.axis[-1].var.name
        ax_K = C.op.reduce_axis[-1].var.name

        wmma_m, wmma_n, wmma_k = self.tiling[ax_M][-1], self.tiling[ax_N][-1], self.tiling[ax_K][-1]
        assert (wmma_m, wmma_n, wmma_k) in [(16, 16, 16), (8, 32, 16), (32, 8, 16)]

        offset = 8
        AF_stride = [wmma_k, 1]
        AS_stride = [self.tiling[ax_K][0] + offset, 1]
        BF_stride = [self.tiling[ax_N][1], 1]
        BS_stride = [self.tiling[ax_N][0] + offset, 1]
        CF_stride = [self.tiling[ax_N][1], 1]
        CS_stride = [self.tiling[ax_N][0] + offset, 1]
        if A.name in self.shared_inputs:
            AS_stride[0] = int(C.op.reduce_axis[-1].dom.extent) + offset

        self.thread_per_block = [32, 1]
        for axis in self.sche[out].op.axis:
            tile = self.tiling[axis.var.name]
            assert tile[0] % tile[1] == 0
            self.thread_per_block[1] *= (tile[0] // tile[1])

        # schedule for output stage
        blck_axis = []
        thrd_axis = []
        for axis in self.sche[out].op.axis:
            bx, tx = self.sche[out].split(axis, factor=self.tiling[axis.var.name][0])
            blck_axis.append(bx)
            thrd_axis.append(tx)
        axis_order = blck_axis + thrd_axis
        self.sche[out].reorder(*axis_order)
        blck_fused = self.sche[out].fuse(*blck_axis)
        thrd_fused = self.sche[out].fuse(*thrd_axis)
        _t, tx = self.sche[out].split(thrd_fused, factor=self.thread_per_block[0])
        _t, ty = self.sche[out].split(_t, factor=self.thread_per_block[1])

        self.sche[out].bind(ty, te.thread_axis("threadIdx.y"))
        self.sche[out].bind(tx, te.thread_axis("threadIdx.x"))
        self.sche[out].bind(blck_fused, te.thread_axis("blockIdx.x"))

        # schedule for block
        self.sche[CS].compute_at(self.sche[out], blck_fused)
        mc, nc = CS.op.axis
        self.sche[CS].storage_align(mc, CS_stride[0] - 1, CS_stride[0])
        mm, mmii = self.sche[CS].split(mc, factor=self.tiling[ax_M][1])
        nn, nnii = self.sche[CS].split(nc, factor=self.tiling[ax_N][1])
        mmii, mmi = self.sche[CS].split(mmii, factor=wmma_m)
        nnii, nni = self.sche[CS].split(nnii, factor=wmma_n)
        self.sche[CS].reorder(mm, nn, mmii, nnii, mmi, nni)
        warp_fused = self.sche[CS].fuse(mm, nn)
        self.sche[CS].bind(warp_fused, te.thread_axis("threadIdx.y"))

        # Schedule for wmma computation
        self.sche[CF].compute_at(self.sche[CS], warp_fused)
        warp_i, _ii = self.sche[CF].split(CF.op.axis[0], factor=wmma_m)
        warp_j, _jj = self.sche[CF].split(CF.op.axis[1], factor=wmma_n)
        ko, _k = self.sche[CF].split(CF.op.reduce_axis[0], factor=self.tiling[ax_K][0])
        ki, _k = self.sche[CF].split(_k, factor=wmma_k)
        self.sche[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

        # Schedule for  wmma_matrix_a load
        self.sche[AF].compute_at(self.sche[CF], ki)
        m, m_ii = self.sche[AF].split(AF.op.axis[0], factor=wmma_m)
        i, i_jj = self.sche[AF].split(AF.op.axis[1], factor=wmma_k)
        self.sche[AF].reorder(m, i, m_ii, i_jj)

        # Schedule for  wmma_matrix_b load
        self.sche[BF].compute_at(self.sche[CF], ki)
        i, i_ii = self.sche[BF].split(BF.op.axis[0], factor=wmma_k)
        n, n_ii = self.sche[BF].split(BF.op.axis[1], factor=wmma_n)
        self.sche[BF].reorder(i, n, i_ii, n_ii)

        # schedule shared
        if A.name in self.shared_inputs:
            self.sche[AS].compute_at(self.sche[out], blck_fused)
        else:
            self.sche[AS].compute_at(self.sche[CF], ko)
        if B.name in self.shared_inputs:
            self.sche[BS].compute_at(self.sche[out], blck_fused)
        else:
            self.sche[BS].compute_at(self.sche[CF], ko)
        self.cooperative_fetch_2d(AS, self.sche, AS_stride[0])
        self.cooperative_fetch_2d(BS, self.sche, BS_stride[0])

        shape = (wmma_m, wmma_n, wmma_k)
        AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=A.dtype)
        BL_gemm = te.placeholder((wmma_k, wmma_n), name="BL_gemm", dtype=B.dtype)
        k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
        CL_compute = te.compute(
            (wmma_m, wmma_n),
            lambda ii, jj: te.sum(
                AL_gemm[ii, k_gemm].astype(C.dtype) * BL_gemm[k_gemm, jj].astype(C.dtype),
                axis=k_gemm,
            ),
            name="CL_compute",
        )

        from tvm.topi.cuda.tensor_intrin import (
            intrin_wmma_load_matrix_A,
            intrin_wmma_load_matrix_W,
            intrin_wmma_store_matrix,
            intrin_wmma_gemm,
        )

        self.sche[AF].tensorize(
            m_ii,
            intrin_wmma_load_matrix_A(
                AF_stride, AS_stride, shape, "row_major", (wmma_m, wmma_k), (wmma_m, wmma_k), A.dtype
            ),
        )
        self.sche[BF].tensorize(
            i_ii,
            intrin_wmma_load_matrix_W(
                BF_stride, BS_stride, shape, "row_major", (wmma_k, wmma_n), (wmma_k, wmma_n), B.dtype
            ),
        )
        self.sche[CF].tensorize(
            _ii, intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape)
        )
        self.sche[CS].tensorize(
            mmi,
            intrin_wmma_store_matrix(
                CS_stride, CF_stride, shape, C.dtype, (wmma_m, wmma_n), (wmma_m, wmma_n)
            ),
        )

        return self.sche
