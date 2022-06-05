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
        # bounds = tvm.te.schedule.InferBound(sch.normalize())

        # if axes[-1].dom.extent % 4 == 0 and isinstance(shared.op.body, tvm.ir.container.Array):
        #     print(np.prod([bounds[ax].extent for ax in axes]), [bounds[ax].extent for ax in axes])
        #     oo, mid = sch[shared].split(fused, factor=4 * self.thread_per_block)
        #     ii, vv = sch[shared].split(mid, factor=4)
        #     sch[shared].reorder(oo, ii, vv)
        #     sch[shared].vectorize(vv)
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
    def rewrite_schedule_no_reduce(self, schedule, tile_dict, shared_inputs=[]):
        self.tiling = tile_dict.copy()
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

        assert(len(reduce_ops) == 0)

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

        for op in elementwise_op:
            for tensor in op.input_tensors:
                cache = isinstance(self.sche[tensor].op, tvm.te.PlaceholderOp) \
                    and len(op.output(0).shape) > len(tensor.shape) \
                    and np.prod(op.output(0).shape) > np.prod(tensor.shape) # is broadcast
                if tensor.name in shared_inputs:
                    cache = True
                if cache:
                    tensor_shared = self.sche.cache_read(tensor, "shared", [op])
                    self.sche[tensor_shared].compute_at(self.sche[out], thrd_fused)
                    self.cooperative_fetch(tensor_shared, self.sche)
                    tensor_local = self.sche.cache_read(tensor_shared, "local", [op])
                    self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)
        return self.sche

    def recursive_schedule_up(self, schedule, tile_dict, shared_inputs=[]):
        self.tiling = tile_dict.copy()
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
        if "raxis_order" not in tile_dict:
            ordered_reduce_axis = self.sche[reg_tile].op.reduce_axis
        else:
            ordered_reduce_axis = []
            for axis_name in tile_dict["raxis_order"]:
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

        for input_tensor in reduce_op.input_tensors:
            shared_tensor = self.sche.cache_read(input_tensor, "shared", [reg_tile])
            # local_tensor = self.sche.cache_read(shared_tensor, "local", [reg_tile])
            # self.sche[local_tensor].compute_at(self.sche[reg_tile], space_fused)
            if input_tensor.name in shared_inputs:
                self.sche[shared_tensor].compute_at(self.sche[out], thrd_fused)
            else:
                self.sche[shared_tensor].compute_at(self.sche[reg_tile], reduce_outer_axis[-1])
            self.cooperative_fetch(shared_tensor, self.sche)

        for op in elementwise_op:
            for tensor in op.input_tensors:
                cache = isinstance(self.sche[tensor].op, tvm.te.PlaceholderOp) \
                    and len(op.output(0).shape) > len(tensor.shape) \
                    and np.prod(op.output(0).shape) > np.prod(tensor.shape) # is broadcast
                if tensor.name in shared_inputs:
                    cache = True
                if cache:
                    tensor_shared = self.sche.cache_read(tensor, "shared", [op])
                    self.sche[tensor_shared].compute_at(self.sche[out], thrd_fused)
                    self.cooperative_fetch(tensor_shared, self.sche)
                    tensor_local = self.sche.cache_read(tensor_shared, "local", [op])
                    self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)
        return self.sche
