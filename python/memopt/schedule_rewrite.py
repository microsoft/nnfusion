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

    def cooperative_fetch_2d(self, shared, sch):
        assert len(self.thread_per_block) == 2
        axes = sch[shared].op.axis
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
    def rewrite_schedule(self, schedule, tile_dict, shared_inputs=[]):
        self.tiling = tile_dict.copy()
        self.sche = schedule
        reduce_op = None
        self.shared_inputs = shared_inputs
        for op in self.sche.stage_map:
            if isinstance(op, tvm.te.ComputeOp) and len(op.reduce_axis) > 0:
                reduce_op = op
                break
        if reduce_op:
            has_inter_thread_reduce = False
            for ax in reduce_op.reduce_axis:
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

        cache_plan = {}
        for op in elementwise_op:
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
        out = None
        reduce_ops = []
        elementwise_op = []
        for op, stage in self.sche.stage_map.items():
            if isinstance(op, tvm.te.ComputeOp):
                if stage.is_output:
                    out = op
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

        for input_tensor in reduce_op.input_tensors:
            shared_tensor = self.sche.cache_read(input_tensor, "shared", [reg_tile])
            # local_tensor = self.sche.cache_read(shared_tensor, "local", [reg_tile])
            # self.sche[local_tensor].compute_at(self.sche[reg_tile], space_fused)
            if input_tensor.name in self.shared_inputs:
                self.sche[shared_tensor].compute_at(self.sche[out], thrd_fused)
            else:
                self.sche[shared_tensor].compute_at(self.sche[reg_tile], reduce_outer_axis[-1])
            self.cooperative_fetch(shared_tensor, self.sche)

        cache_plan = {}
        for op in elementwise_op:
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
            consumers = list(filter(lambda x: x.output(0) not in reduce_op.input_tensors, consumers))
            tensor_local = self.sche.cache_read(tensor_shared, "local", consumers)
            self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)

        return self.sche

    def rewrite_schedule_inter_thread(self):
        out = None
        reduce_ops = []
        elementwise_op = []
        for op, stage in self.sche.stage_map.items():
            if isinstance(op, tvm.te.ComputeOp):
                if stage.is_output:
                    out = op
                else:
                    stage.compute_inline()
                if len(op.reduce_axis) > 0:
                    reduce_ops.append(op)
                else:
                    elementwise_op.append(op)

        assert(len(reduce_ops) == 1)
        reduce_op = reduce_ops[0]
        reg_tile = reduce_op.output(0)

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
        if out is not reduce_op:
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

        for input_tensor in reduce_op.input_tensors:
            shared_tensor = self.sche.cache_read(input_tensor, "shared", [reg_tile])
            if input_tensor.name in self.shared_inputs:
                self.sche[shared_tensor].compute_at(self.sche[out], thrd_fused)
            else:
                self.sche[shared_tensor].compute_at(self.sche[reg_tile], reduce_outer_axis[-1])
            self.cooperative_fetch_2d(shared_tensor, self.sche)

        cache_plan = {}
        for op in elementwise_op:
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
            consumers = list(filter(lambda x: x.output(0) not in reduce_op.input_tensors, consumers))
            tensor_local = self.sche.cache_read(tensor_shared, "local", consumers)
            self.sche[tensor_local].compute_at(self.sche[out], thrd_fused)

        return self.sche

