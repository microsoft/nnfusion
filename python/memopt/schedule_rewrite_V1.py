from audioop import bias
from functools import reduce
import tvm
from tvm import te
import numpy as np

def reverse_topo_order(sch):
    ready = set()
    dependency_count = {}

    result = []
    for op, stage in sch.stage_map.items():
        if stage.is_output:
            ready.add(op)
        for t in op.input_tensors:
            if t not in dependency_count:
                dependency_count[t] = 0
            dependency_count[t] += 1
    while len(ready) > 0:
        op = ready.pop()
        result.append(op)
        for t in op.input_tensors:
            dependency_count[t] -= 1
            assert(dependency_count[t] >= 0)
            if dependency_count[t] == 0:
                ready.add(sch[t].op)
    return result

class CodeGenerator:
    def __init__(self):
        self.storage_align_on = False

    def split_axis(self, op, axis, sche = None):
        ret = []
        factors = self.tiling[axis.var.name]
        for i in range(0, len(factors)):
            ax0, ax1 = self.sche[op].split(axis, factor=int(np.prod(factors[i:])))
            ret.append(ax0)
            axis = ax1
        return ret + [axis]

    def cooperative_fetch(self, shared, sch):
        axes = sch[shared].op.axis
        fused = sch[shared].fuse(*axes)
        oo, ii = sch[shared].split(fused, factor=self.thread_per_block)
        sch[shared].reorder(oo, ii)
        sch[shared].unroll(oo)
        sch[shared].bind(ii, te.thread_axis("threadIdx.x"))

    # [Parameters]
    #   schedule: the original TVM schedule of an op
    #   tile_dict: a dictionary holding split factors of each axis,
    #              e.g., {"i": [8, 16, 1], "j": [8, 16, 1], "k": [32]}.
    #              For spacial axes, the format is "axis_name": [thread_tile_size, thread_num, 1].
    #              For reduce axes, the format is "axis_name": [step_size].
    #   bind_dict: a dictionary indicating which GPU index an axis should be bound to.
    #              Since we'll fuse spatial and reduction axes respectively, it's sufficient
    #              to just provide binding information for spatial and reduction axes,
    #              e.g., {"space": ["blockIdx.x", "threadIdx.y", None], "reduce": [None, "threadIdx.x"]}.
    #   smem_bool: True if we need tiling at shared memory
    #   reg_bool: True if we need tiling at register files
    #
    # [Return]
    #   new_s: an optimized TVM schedule
    def rewrite_schedule(self, schedule, tile_dict, smem_bool, reg_bool, target_stage='compute', tile_blacklist=[]):
        self.tiling = tile_dict
        self.need_smem_tiling = smem_bool
        self.need_reg_tiling = reg_bool
        self.sche = schedule

        input_tensors = []
        out = None
        for tensor, stage in self.sche.stage_map.items():
            if stage.is_output:
                out = tensor.output(0)
            elif isinstance(stage.op, tvm.te.tensor.PlaceholderOp):
                input_tensors.append(tensor.output(0))

        assert(out is not None)

        # adjust format
        for axis in self.sche[out].op.axis:
            name = axis.var.name
            if len(self.tiling[name]) == 2:
                vthrd = self.tiling[name][1]
                thrd = self.tiling[name][0]
                self.tiling[name] = [vthrd, thrd, 1]
        #print('reduce:', self.sche[out].op.reduce_axis)
        #print('space:', self.sche[out].op.axis)

        self.thread_per_block = 1
        for axis in self.sche[out].op.axis:
            self.thread_per_block *= self.tiling[axis.var.name][1]

        shared_tensor_list = []
        local_tensor_list = []
        reg_tile = None
        # print("[Add cache stage]")
        if self.need_smem_tiling:
            for input_tensor in input_tensors:
                shared_tensor = self.sche.cache_read(input_tensor, "shared", [out])
                shared_tensor_list.append(shared_tensor)

        if self.need_reg_tiling:
            for shared_tensor in shared_tensor_list:
                local_tensor = self.sche.cache_read(shared_tensor, "local", [out])
                local_tensor_list.append(local_tensor)
            reg_tile = self.sche.cache_write(out, "local")

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for axis in self.sche[out].op.axis:
            bx, vx, tx, tn = self.split_axis(out, axis)
            # bx, tx, tn = self.split_axis(out, axis)
            blck_axis.append(bx)
            vthd_axis.append(vx)
            thrd_axis.append(tx)
            tile_axis.append(tn)
        axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis
        # print("[Split spatial axis]\n", axis_order)
        self.sche[out].reorder(*axis_order)
        blck_fused = self.sche[out].fuse(*blck_axis)
        thrd_fused = self.sche[out].fuse(*thrd_axis)
        self.sche[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
        for va in vthd_axis:
            self.sche[out].bind(va, te.thread_axis("vthread"))
        self.sche[out].bind(thrd_fused, te.thread_axis("threadIdx.x"))

        reduce_outer_axis, reduce_inner_axis = [], []

        if reg_tile is not None:
            self.sche[reg_tile].compute_at(self.sche[out], thrd_fused)
            space_axis = list(self.sche[reg_tile].op.axis)
            for axis in self.sche[reg_tile].op.reduce_axis:
                factor = self.tiling[axis.var.name][0]
                ro, ri = self.sche[reg_tile].split(axis, factor=factor)
                reduce_outer_axis.append(ro)
                reduce_inner_axis.append(ri)

            axis_order = reduce_outer_axis + reduce_inner_axis + space_axis
            self.sche[reg_tile].reorder(*axis_order)
            space_fused = self.sche[reg_tile].fuse(*space_axis)
            self.sche[reg_tile].unroll(space_fused)
        else:
            for axis in self.sche[out].op.reduce_axis:
                factor = self.tiling[axis.var.name][0]
                ro, ri = self.sche[out].split(axis, factor=factor)
                reduce_outer_axis.append(ro)
                reduce_inner_axis.append(ri)
                # bind_idx = te.thread_axis("threadIdx.x")
                # self.sche[out].bind(reduce_axis[1], bind_idx)
                # self.sche[out].set_store_predicate(bind_idx.var.equal(0))

        compute_stage = out if reg_tile is None else reg_tile
        for rt in local_tensor_list:
            self.sche[rt].compute_at(self.sche[compute_stage], space_fused)
        for st in shared_tensor_list:
            if st.name.endswith(".shared") and st.name[:-len(".shared")] in tile_blacklist:
                self.sche[st].compute_at(self.sche[out], blck_fused)
            else:
                self.sche[st].compute_at(self.sche[compute_stage], reduce_outer_axis[-1])
            self.cooperative_fetch(st, self.sche)

        return self.sche

    def recursive_schedule_up(self, schedule, tile_dict, tile_blacklist=[]):
        self.tiling = tile_dict
        self.sche = schedule

        out = None
        reduce_ops = []
        elementwise_op = []
        for op, stage in self.sche.stage_map.items():
            if isinstance(op, tvm.te.ComputeOp):
                if stage.is_output:
                    out = op.output(0)
                else:
                    stage.compute_inline()
                if len(op.reduce_axis) > 0:
                    reduce_ops.append(op)
                else:
                    elementwise_op.append(op)

        assert(len(reduce_ops) == 1)
        reduce_op = reduce_ops[0]
        # order = reverse_topo_order(self.sche)

        self.thread_per_block = 1
        for axis in self.sche[out].op.axis:
            name = axis.var.name
            if len(self.tiling[name]) == 2:
                vthrd = self.tiling[name][1]
                thrd = self.tiling[name][0]
                self.tiling[name] = [vthrd, thrd, 1]
            self.thread_per_block *= self.tiling[name][1]

        reg_tile = self.sche.cache_write(reduce_op.output(0), "local")

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
        for axis in self.sche[reg_tile].op.reduce_axis:
            factor = self.tiling[axis.var.name][0]
            ro, ri = self.sche[reg_tile].split(axis, factor=factor)
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)
        axis_order = reduce_outer_axis + reduce_inner_axis + space_axis
        self.sche[reg_tile].reorder(*axis_order)
        space_fused = self.sche[reg_tile].fuse(*space_axis)
        self.sche[reg_tile].unroll(space_fused)

        for input_tensor in reduce_op.input_tensors:
            shared_tensor = self.sche.cache_read(input_tensor, "shared", [reg_tile])
            local_tensor = self.sche.cache_read(shared_tensor, "local", [reg_tile])
            self.sche[local_tensor].compute_at(self.sche[reg_tile], space_fused)
            if input_tensor.name in tile_blacklist:
                self.sche[shared_tensor].compute_at(self.sche[out], thrd_fused)
            else:
                self.sche[shared_tensor].compute_at(self.sche[reg_tile], reduce_outer_axis[-1])
            self.cooperative_fetch(shared_tensor, self.sche)

        for op in elementwise_op:
            out_shape = op.output(0).shape
            for tensor in op.input_tensors:
                if isinstance(self.sche[tensor].op, tvm.te.PlaceholderOp) \
                and np.prod(out_shape) > np.prod(tensor.shape):
                    tensor_shared = self.sche.cache_read(tensor, "shared", [op])
                    self.sche[tensor_shared].compute_at(self.sche[out], thrd_fused)
                    self.cooperative_fetch(tensor_shared, self.sche)
                    # tensor_local = self.sche.cache_read(tensor_shared, "local", [op])
                    # self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)
        return self.sche
