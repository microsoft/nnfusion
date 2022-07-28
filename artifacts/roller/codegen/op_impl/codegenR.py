"""
    codegen implementation for rprog
"""

from os import close
import tvm
from tvm import te
import numpy as np
import math
import copy
from .tc_intrin import (
    init_intrin_strides,
    intrin_wmma_load_matrix,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)
"""
from tvm.topi.cuda.tensor_intrin import (
    intrin_wmma_load_matrix_A,
    intrin_wmma_load_matrix_W,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)
"""

class CodeGeneratorR:
    def get_codegen_dict(self, rprog):
        """
            convert a rprog to tiling, results stored to self.tiling
        """
        self.tiling = {}
        for axis_name in rprog.saxis:
            self.tiling[axis_name] = []
            axis_cfg = rprog.GetAxisConfig(axis_name)
            for i in range(rprog.num_level):
                self.tiling[axis_name].append(math.ceil(axis_cfg[i] / axis_cfg[i + 1]))
        for axis_name in rprog.raxis:
            self.tiling[axis_name] = []
            axis_cfg = rprog.GetAxisConfig(axis_name)
            for i in range(rprog.num_level):
                self.tiling[axis_name].append(math.ceil(axis_cfg[i] / axis_cfg[i + 1]))
        return self.tiling

    def split_axis(self, op, axis, sche = None):
        if sche == None:
            sche = self.sche
        ret = []
        factors = self.tiling[axis.var.name]
        for i in range(0, len(factors)):
            ax0, ax1 = sche[op].split(axis, factor=int(np.prod(factors[i:])))
            ret.append(ax0)
            axis = ax1
        return ret + [axis]

    def update_thread_per_block(self, stage, sche = None, vthread=True):
        if sche == None:
            sche = self.sche
        num = 1
        print(self.tiling)
        for axis in sche[stage].op.axis:
            num = num * self.tiling[axis.var.name][1 if vthread else 0]
        self.thread_per_block = num

    def cooperative_fetch(self, shared, sch):
        axes = sch[shared].op.axis
        fused = sch[shared].fuse(*axes)
        fused, ii_n = sch[shared].split(fused, factor=self.bank_size // 4)
        oo, ii = sch[shared].split(fused, factor=self.thread_per_block)
        #ii, ii_n = sch[shared].split(ii, factor=2)
        sch[shared].vectorize(ii_n)
        sch[shared].reorder(oo, ii, ii_n)
        sch[shared].unroll(oo)
        # sch[shared].unroll(ii_n)
        sch[shared].bind(ii, te.thread_axis("threadIdx.x"))

    def calc_grid(self, reduce_iters, space_iters, vthread=True):
        blck_dict = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
        thrd_dict = {"threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1}

        for iter in space_iters:
            if iter.var.name in self.tiling:
                factors = self.tiling[iter.var.name]
                length = iter.dom.extent
                blck = max(length // int(np.prod(factors[0:])), 1)
                thrd = factors[1 if vthread else 0]
                if self.binding["space"][0] in blck_dict:
                    blck_dict[self.binding["space"][0]] *= blck
                if self.binding["space"][2 if vthread else 1] in thrd_dict:
                    thrd_dict[self.binding["space"][2 if vthread else 1]] *= thrd

        for iter in reduce_iters:
            if iter.var.name in self.tiling:
                factors = self.tiling[iter.var.name]
                length = iter.dom.extent
                blck = max(length // int(np.prod(factors[0:])), 1)
                thrd = factors[0]
                if self.binding["reduce"][0] in blck_dict:
                    blck_dict[self.binding["reduce"][0]] *= blck
                if self.binding["reduce"][1] in thrd_dict:
                    thrd_dict[self.binding["reduce"][1]] *= thrd

        self.blck_grid = [blck_dict["blockIdx.x"], blck_dict["blockIdx.y"], blck_dict["blockIdx.z"]]
        self.thrd_grid = [thrd_dict["threadIdx.x"], thrd_dict["threadIdx.y"], thrd_dict["threadIdx.z"]]
        # print("blck_grid: ", self.blck_grid, "thrd_grid: ", self.thrd_grid)

    def adjust_format(self, out):
        for axis in self.sche[out].op.axis:
            name = axis.var.name
            if len(self.tiling[name]) == 2:
                vthrd = self.tiling[name][1]
                thrd = self.tiling[name][0]
                self.tiling[name] = [vthrd, thrd, 1]
        #print("Config:", self.tiling)

    # [Parameters]
    #   schedule: the original TVM schedule of an op
    #   rprog: roller's rprog configuration
    #   smem_bool: True if we need tiling at shared memory
    #   reg_bool: True if we need tiling at register files
    #
    # [Return]
    #   new_s: an optimized TVM schedule
    def rewrite_schedule(self, schedule, rprog, smem_bool, reg_bool, target_stage='compute', align_info = [], bank_size = 4):
        # self.storage_align_on = st_align
        self.bank_size = bank_size
        # self.bank_number = bank_number
        self.binding = {"space": ["blockIdx.x", "vthread", "threadIdx.x"], "reduce": [None, None]}
        self.get_codegen_dict(rprog)
        print(self.tiling)
        self.need_smem_tiling = smem_bool
        self.need_reg_tiling = reg_bool
        self.sche = schedule
        # align_info = self.get_align_info(schedule, rprog, smem_bool, reg_bool, target_stage, st_align, bank_size, bank_number)

        input_tensors = []
        output_num = 0
        output_tensors = []

        for item in self.sche.stage_map.items():
            if isinstance(item[0], tvm.te.tensor.ComputeOp):
                output_num = item[0].num_outputs
                for i in range(output_num):
                    if item[0].name != target_stage:
                        out = item[0].output(i)
                        self.sche[out].compute_inline()
                    else:
                        input_tensors = list(item[0].input_tensors)
                        output_tensors.append(item[0].output(i))
        # print("Input: ", input_tensors)
        # print("Output: ", output_tensors)
        for out in output_tensors:
            # print('reduce:', self.sche[out].op.reduce_axis)
            # print('space:', self.sche[out].op.axis)
            self.adjust_format(out)
            # TVM only allows binding reduce axis if it's the only one
            if self.binding["reduce"][1] is not None:
                assert len(self.sche[out].op.reduce_axis) == 1

            self.update_thread_per_block(out)
            all_iters = self.sche[out].all_iter_vars
            reduce_iters = out.op.reduce_axis
            space_iters = list(set(all_iters) - set(reduce_iters))
            self.calc_grid(reduce_iters, space_iters)
            # print("Target: {}\nSpace Iters: {}\nReduce Iters: {}\n".format(out, space_iters, reduce_iters))

            smem_tensor = []
            reg_tensor = []
            reg_tile = None
            # print("[Add cache stage]")
            if self.need_smem_tiling:
                for input_tensor in input_tensors:
                    shared_tensor = self.sche.cache_read(input_tensor, "shared", [out])
                    smem_tensor.append(shared_tensor)

            if self.need_reg_tiling:
                for shared_tensor in smem_tensor:
                    local_tensor = self.sche.cache_read(shared_tensor, "local", [out])
                    reg_tensor.append(local_tensor)
                reg_tile = self.sche.cache_write(out, "local")

            blck_axis = []
            vthd_axis = []
            thrd_axis = []
            tile_axis = []
            for axis in self.sche[out].op.axis:
                # adjust self.tiling's space axis for proper smem load
                # TODO: what if not two-level tiling structure?
                if self.bank_size != 4:
                    assert len(self.tiling[axis.var.name]) == 3
                    if self.tiling[axis.var.name][-3] >= (self.bank_size // 4):
                        self.tiling[axis.var.name][-3] = self.tiling[axis.var.name][-3] // (self.bank_size // 4)
                        self.tiling[axis.var.name][-1] = self.tiling[axis.var.name][-1] * (self.bank_size // 4)
                    else:
                        print('shared mem tiling is too small.')
                        self.tiling[axis.var.name][-1] = self.tiling[axis.var.name][-1] * self.tiling[axis.var.name][-3]
                        self.tiling[axis.var.name][-3] = 1
                    print('updated self.tiling: ', self.tiling)

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
            if self.binding["space"][0] is not None:
                self.sche[out].bind(blck_fused, te.thread_axis(self.binding["space"][0]))
            if self.binding["space"][1] is not None:
                for va in vthd_axis:
                    self.sche[out].bind(va, te.thread_axis(self.binding["space"][1]))
            if self.binding["space"][2] is not None:
                self.sche[out].bind(thrd_fused, te.thread_axis(self.binding["space"][2]))

            reduce_axis = []
            if reg_tile is not None:
                self.sche[reg_tile].compute_at(self.sche[out], thrd_fused)
                space_axis = []
                for axis in self.sche[reg_tile].op.axis:
                    space_axis.append(axis)
                for axis in self.sche[reg_tile].op.reduce_axis:
                    res = self.split_axis(reg_tile, axis)
                    reduce_axis = reduce_axis + res
                axis_order = reduce_axis + space_axis
                # print('axis_order', axis_order)
                # print("[Split reduction axis]\n", axis_order)
                self.sche[reg_tile].reorder(*axis_order)
                space_fused = self.sche[reg_tile].fuse(*space_axis)
                self.sche[reg_tile].unroll(space_fused)
            else:
                for axis in self.sche[out].op.reduce_axis:
                    res = self.split_axis(out, axis)
                    reduce_axis = reduce_axis + res
                if self.binding["reduce"][1] is not None:
                    bind_idx = te.thread_axis(self.binding["reduce"][1])
                    self.sche[out].bind(reduce_axis[1], bind_idx)
                    self.sche[out].set_store_predicate(bind_idx.var.equal(0))

            # print("[Cooperative fetching]")
            if reg_tile is not None:
                for rt in reg_tensor:
                    self.sche[rt].compute_at(self.sche[reg_tile], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche[st].compute_at(self.sche[reg_tile], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche)
            else:
                for rt in reg_tensor:
                    self.sche[rt].compute_at(self.sche[out], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche[st].compute_at(self.sche[out], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche)

        for info in align_info:
            idx, factor, offset = info
            st = smem_tensor[idx]
            # st_size = tvm.runtime.DataType(st.dtype).bits // 8
            # num_ele = bank_size // st_size
            # assert num_ele > 0
            # factor = factor * num_ele
            # offset = math.ceil(offset/num_ele) * num_ele
            self.sche[st].storage_align(st.op.axis[-2], factor, offset)

        return self.sche

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

    def rewrite_schedule_fuse(self, schedule, rprog, smem_bool, reg_bool, input_tensors, output_tensors, write_tensor, target_stage="conv2d_nchw_implicit_gemm", write_stage="output", align_info = [], bank_size = 4):
        # self.storage_align_on = st_align
        self.bank_size = bank_size
        # self.bank_number = bank_number
        self.binding = {"space": ["blockIdx.x", "vthread", "threadIdx.x"], "reduce": [None, None]}
        self.get_codegen_dict(rprog)
        print(self.tiling)
        self.need_smem_tiling = smem_bool
        self.need_reg_tiling = reg_bool
        self.sche = schedule
        # align_info = self.get_align_info_fuse(schedule, rprog, smem_bool, reg_bool, target_stage, st_align, bank_size, bank_number)

        for out in output_tensors:
            #print('reduce:', self.sche[out].op.reduce_axis)
            #print('space:', self.sche[out].op.axis)
            self.adjust_format(out)
            # TVM only allows binding reduce axis if it's the only one
            if self.binding["reduce"][1] is not None:
                assert len(self.sche[out].op.reduce_axis) == 1

            self.update_thread_per_block(out)
            all_iters = self.sche[out].all_iter_vars
            reduce_iters = out.op.reduce_axis
            space_iters = list(set(all_iters) - set(reduce_iters))
            self.calc_grid(reduce_iters, space_iters)
            # print("Target: {}\nSpace Iters: {}\nReduce Iters: {}\n".format(out, space_iters, reduce_iters))

            smem_tensor = []
            reg_tensor = []
            reg_tile = self.sche.cache_write(out, "local")
            # print("[Add cache stage]")
            if self.need_smem_tiling:
                for input_tensor in input_tensors:
                    self.sche[input_tensor].compute_inline()
                    shared_tensor = self.sche.cache_read(input_tensor, "shared", [reg_tile])
                    smem_tensor.append(shared_tensor)

                for shared_tensor in smem_tensor:
                    local_tensor = self.sche.cache_read(shared_tensor, "local", [reg_tile])
                    reg_tensor.append(local_tensor)

            blck_axis = []
            vthd_axis = []
            thrd_axis = []
            tile_axis = []
            self.sche[out].compute_inline()
            out = write_tensor
            for axis in self.sche[out].op.axis:
                if self.bank_size != 4:
                    assert len(self.tiling[axis.var.name]) == 3
                    if self.tiling[axis.var.name][-3] >= (self.bank_size // 4):
                        self.tiling[axis.var.name][-3] = self.tiling[axis.var.name][-3] // (self.bank_size // 4)
                        self.tiling[axis.var.name][-1] = self.tiling[axis.var.name][-1] * (self.bank_size // 4)
                    else:
                        print('shared mem tiling is too small.')
                        self.tiling[axis.var.name][-1] = self.tiling[axis.var.name][-1] * self.tiling[axis.var.name][-3]
                        self.tiling[axis.var.name][-3] = 1
                    print('updated self.tiling: ', self.tiling)
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
            if self.binding["space"][0] is not None:
                self.sche[out].bind(blck_fused, te.thread_axis(self.binding["space"][0]))
            if self.binding["space"][1] is not None:
                for va in vthd_axis:
                    self.sche[out].bind(va, te.thread_axis(self.binding["space"][1]))
            if self.binding["space"][2] is not None:
                self.sche[out].bind(thrd_fused, te.thread_axis(self.binding["space"][2]))

            reduce_axis = []
            if reg_tile is not None:
                self.sche[reg_tile].compute_at(self.sche[out], thrd_fused)
                space_axis = []
                for axis in self.sche[reg_tile].op.axis:
                    space_axis.append(axis)
                for axis in self.sche[reg_tile].op.reduce_axis:
                    res = self.split_axis(reg_tile, axis)
                    reduce_axis = reduce_axis + res
                axis_order = reduce_axis + space_axis
                self.sche[reg_tile].reorder(*axis_order)
                reg_fused = self.sche[reg_tile].fuse(*space_axis)
                self.sche[reg_tile].unroll(reg_fused)
            else:
                for axis in self.sche[out].op.reduce_axis:
                    res = self.split_axis(out, axis)
                    reduce_axis = reduce_axis + res
                if self.binding["reduce"][1] is not None:
                    bind_idx = te.thread_axis(self.binding["reduce"][1])
                    self.sche[out].bind(reduce_axis[1], bind_idx)
                    self.sche[out].set_store_predicate(bind_idx.var.equal(0))

            # print("[Cooperative fetching]")
            if reg_tile is not None:
                for rt in reg_tensor:
                    self.sche[rt].compute_at(self.sche[reg_tile], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche[st].compute_at(self.sche[reg_tile], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche)
            else:
                for rt in reg_tensor:
                    self.sche[rt].compute_at(self.sche[out], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche[st].compute_at(self.sche[out], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche)
        for info in align_info:
            idx, factor, offset = info
            st = smem_tensor[idx]
            # st_size = tvm.runtime.DataType(st.dtype).bits // 8
            # num_ele = bank_size // st_size
            # assert num_ele > 0
            # factor = factor * num_ele
            # offset = math.ceil(offset/num_ele) * num_ele
            self.sche[st].storage_align(st.op.axis[-2], factor, offset)
        # assert False
        return self.sche

'''
    def get_align_info(self, schedule, rprog, smem_bool=False, reg_bool=False, target_stage='compute', st_align=False, bank_size=4, bank_number=32):
        if not smem_bool or not reg_bool or not st_align:
            return []
        self.storage_align_on = st_align
        self.bank_size = bank_size
        self.bank_number = bank_number
        self.get_codegen_dict(rprog)
        self.need_smem_tiling = smem_bool
        self.need_reg_tiling = reg_bool
        self.sche_simu = copy.deepcopy(schedule)
        binding = {"space": ["blockIdx.x", "threadIdx.x"], "reduce": [None, None]}

        input_tensors = []
        output_num = 0
        output_tensors = []

        for item in self.sche_simu.stage_map.items():
            if isinstance(item[0], tvm.te.tensor.ComputeOp):
                output_num = item[0].num_outputs
                for i in range(output_num):
                    if item[0].name != target_stage:
                        out = item[0].output(i)
                        self.sche_simu[out].compute_inline()
                    else:
                        input_tensors = list(item[0].input_tensors)
                        output_tensors.append(item[0].output(i))

        for out in output_tensors:
            #print('reduce:', self.sche_simu[out].op.reduce_axis)
            #print('space:', self.sche_simu[out].op.axis)
            # self.adjust_format(out)
            # TVM only allows binding reduce axis if it's the only one
            if binding["reduce"][1] is not None:
                assert len(self.sche_simu[out].op.reduce_axis) == 1
            self.update_thread_per_block(out, self.sche_simu, False)
            # all_iters = self.sche_simu[out].all_iter_vars
            # reduce_iters = out.op.reduce_axis
            # space_iters = list(set(all_iters) - set(reduce_iters))
            # self.calc_grid(reduce_iters, space_iters, False)
            # print("Target: {}\nSpace Iters: {}\nReduce Iters: {}\n".format(out, space_iters, reduce_iters))

            smem_tensor = []
            reg_tensor = []
            reg_tile = None
            # print("[Add cache stage]")
            if self.need_smem_tiling:
                for input_tensor in input_tensors:
                    shared_tensor = self.sche_simu.cache_read(input_tensor, "shared", [out])
                    smem_tensor.append(shared_tensor)

            if self.need_reg_tiling:
                for shared_tensor in smem_tensor:
                    local_tensor = self.sche_simu.cache_read(shared_tensor, "local", [out])
                    reg_tensor.append(local_tensor)
                reg_tile = self.sche_simu.cache_write(out, "local")

            blck_axis = []
            # vthd_axis = []
            thrd_axis = []
            tile_axis = []
            for axis in self.sche_simu[out].op.axis:
                # bx, vx, tx, tn = self.split_axis(out, axis)
                bx, tx, tn = self.split_axis(out, axis, self.sche_simu)
                blck_axis.append(bx)
                # vthd_axis.append(vx)
                thrd_axis.append(tx)
                tile_axis.append(tn)
            # axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis
            axis_order = blck_axis + thrd_axis + tile_axis
            # print("[Split spatial axis]\n", axis_order)
            self.sche_simu[out].reorder(*axis_order)
            blck_fused = self.sche_simu[out].fuse(*blck_axis)
            thrd_fused = self.sche_simu[out].fuse(*thrd_axis)
            if binding["space"][0] is not None:
                self.sche_simu[out].bind(blck_fused, te.thread_axis(binding["space"][0]))
            # if self.binding["space"][1] is not None:
            #     for va in vthd_axis:
            #         self.sche_simu[out].bind(va, te.thread_axis(self.binding["space"][1]))
            if binding["space"][1] is not None:
                self.sche_simu[out].bind(thrd_fused, te.thread_axis(binding["space"][1]))

            reduce_axis = []
            if reg_tile is not None:
                self.sche_simu[reg_tile].compute_at(self.sche_simu[out], thrd_fused)
                space_axis = []
                for axis in self.sche_simu[reg_tile].op.axis:
                    space_axis.append(axis)
                for axis in self.sche_simu[reg_tile].op.reduce_axis:
                    res = self.split_axis(reg_tile, axis, self.sche_simu)
                    reduce_axis = reduce_axis + res
                axis_order = reduce_axis + space_axis
                # print('axis_order', axis_order)
                # print("[Split reduction axis]\n", axis_order)
                self.sche_simu[reg_tile].reorder(*axis_order)
                space_fused = self.sche_simu[reg_tile].fuse(*space_axis)
                #self.sche_simu[reg_tile].unroll(space_fused)
            else:
                for axis in self.sche_simu[out].op.reduce_axis:
                    res = self.split_axis(out, axis, self.sche_simu)
                    reduce_axis = reduce_axis + res
                if binding["reduce"][1] is not None:
                    bind_idx = te.thread_axis(binding["reduce"][1])
                    self.sche_simu[out].bind(reduce_axis[1], bind_idx)
                    self.sche_simu[out].set_store_predicate(bind_idx.var.equal(0))

            # print("[Cooperative fetching]")
            if reg_tile is not None:
                for rt in reg_tensor:
                    self.sche_simu[rt].compute_at(self.sche_simu[reg_tile], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche_simu[st].compute_at(self.sche_simu[reg_tile], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche_simu)
            else:
                for rt in reg_tensor:
                    self.sche_simu[rt].compute_at(self.sche_simu[out], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche_simu[st].compute_at(self.sche_simu[out], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche_simu)
        return self.add_storage_align(smem_tensor, reg_tensor)

    def get_align_info_fuse(self, schedule, rprog, smem_bool=False, reg_bool=False, target_stage='conv2d_nchw_implicit_gemm', write_stage="output", st_align=False, bank_size=4, bank_number=32):
        if not smem_bool or not reg_bool or not st_align:
            return []
        self.storage_align_on = st_align
        self.bank_size = bank_size
        self.bank_number = bank_number
        self.get_codegen_dict(rprog)
        self.need_smem_tiling = smem_bool
        self.need_reg_tiling = reg_bool
        self.sche_simu = copy.deepcopy(schedule)
        binding = {"space": ["blockIdx.x", "threadIdx.x"], "reduce": [None, None]}

        input_tensors = []
        output_num = 0
        output_tensors = []
        write_tensor = None

        for item in self.sche_simu.stage_map.items():
            if isinstance(item[0], tvm.te.tensor.ComputeOp):
                output_num = item[0].num_outputs
                for i in range(output_num):
                    if item[0].name == target_stage:
                        input_tensors = list(item[0].input_tensors)
                        output_tensors.append(item[0].output(i))
                    elif item[0].name == write_stage:
                        write_tensor = item[0].output(i)
        i=0
        smem_tensor = []
        reg_tensor = []
        for out in output_tensors:
            #print('reduce:', self.sche_simu[out].op.reduce_axis)
            #print('space:', self.sche_simu[out].op.axis)
            # self.adjust_format(out)
            # TVM only allows binding reduce axis if it's the only one
            if binding["reduce"][1] is not None:
                assert len(self.sche_simu[out].op.reduce_axis) == 1
            self.update_thread_per_block(out, self.sche_simu, False)
            # all_iters = self.sche_simu[out].all_iter_vars
            # reduce_iters = out.op.reduce_axis
            # space_iters = list(set(all_iters) - set(reduce_iters))
            # self.calc_grid(reduce_iters, space_iters, False)
            # print("Target: {}\nSpace Iters: {}\nReduce Iters: {}\n".format(out, space_iters, reduce_iters))

            smem_tensor = []
            reg_tensor = []
            reg_tile = None
            # print("[Add cache stage]")
            if self.need_smem_tiling:
                for input_tensor in input_tensors:
                    self.sche_simu[input_tensor].compute_inline()
                    shared_tensor = self.sche_simu.cache_read(input_tensor, "shared", [reg_tile])
                    smem_tensor.append(shared_tensor)

            if self.need_reg_tiling:
                for shared_tensor in smem_tensor:
                    local_tensor = self.sche_simu.cache_read(shared_tensor, "local", [reg_tile])
                    reg_tensor.append(local_tensor)
                reg_tile = self.sche_simu.cache_write(out, "local")


            blck_axis = []
            # vthd_axis = []
            thrd_axis = []
            tile_axis = []
            self.sche_simu[out].compute_inline()
            out = write_tensor
            for axis in self.sche_simu[out].op.axis:
                # bx, vx, tx, tn = self.split_axis(out, axis)
                bx, tx, tn = self.split_axis(out, axis, self.sche_simu)
                blck_axis.append(bx)
                # vthd_axis.append(vx)
                thrd_axis.append(tx)
                tile_axis.append(tn)
            # axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis
            axis_order = blck_axis + thrd_axis + tile_axis
            # print("[Split spatial axis]\n", axis_order)
            self.sche_simu[out].reorder(*axis_order)
            blck_fused = self.sche_simu[out].fuse(*blck_axis)
            thrd_fused = self.sche_simu[out].fuse(*thrd_axis)
            if binding["space"][0] is not None:
                self.sche_simu[out].bind(blck_fused, te.thread_axis(binding["space"][0]))
            # if self.binding["space"][1] is not None:
            #     for va in vthd_axis:
            #         self.sche_simu[out].bind(va, te.thread_axis(self.binding["space"][1]))
            if binding["space"][1] is not None:
                self.sche_simu[out].bind(thrd_fused, te.thread_axis(binding["space"][1]))

            reduce_axis = []
            self.sche_simu[reg_tile].compute_at(self.sche_simu[out], thrd_fused)
            space_axis = []
            if reg_tile is not None:
                for axis in self.sche_simu[reg_tile].op.axis:
                    space_axis.append(axis)
                for axis in self.sche_simu[reg_tile].op.reduce_axis:
                    res = self.split_axis(reg_tile, axis, self.sche_simu)
                    reduce_axis = reduce_axis + res
                axis_order = reduce_axis + space_axis
                self.sche_simu[reg_tile].reorder(*axis_order)
                reg_fused = self.sche_simu[reg_tile].fuse(*space_axis)
                self.sche_simu[reg_tile].unroll(reg_fused)
            else:
                for axis in self.sche_simu[out].op.reduce_axis:
                    res = self.split_axis(out, axis, self.sche_simu)
                    reduce_axis = reduce_axis + res
                if binding["reduce"][1] is not None:
                    bind_idx = te.thread_axis(binding["reduce"][1])
                    self.sche_simu[out].bind(reduce_axis[1], bind_idx)
                    self.sche_simu[out].set_store_predicate(bind_idx.var.equal(0))

            # print("[Cooperative fetching]")
            if reg_tile is not None:
                for rt in reg_tensor:
                    self.sche_simu[rt].compute_at(self.sche_simu[reg_tile], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche_simu[st].compute_at(self.sche_simu[reg_tile], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche_simu)
            else:
                for rt in reg_tensor:
                    self.sche_simu[rt].compute_at(self.sche_simu[out], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche_simu[st].compute_at(self.sche_simu[out], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche_simu)

        return self.add_storage_align(smem_tensor, reg_tensor)

    def add_storage_align(self, smem_tensor, reg_tensor):
        res = []
        assert len(smem_tensor) == len(reg_tensor)
        factor = self.bank_number
        self.sche_simu = self.sche_simu.normalize()
        bounds = tvm.te.schedule.InferBound(self.sche_simu)
        for j in range(len(reg_tensor)):
            outer = 1
            axes = reg_tensor[j].op.axis
            l = len(axes)
            if l > 1:
                for i in range(len(axes) - 1):
                    outer *= bounds[axes[i]].extent
                # print("outer,", outer)
                if outer > 1:
                    inner_most = bounds[axes[-1]].extent
                    # print("inner most, ", inner_most)
                    st = smem_tensor[j]
                    if len(st.op.axis) > 1:
                        st_size = tvm.runtime.DataType(st.dtype).bits // 8
                        num_ele = self.bank_size // st_size
                        assert num_ele > 0
                        factor = factor * num_ele
                        offset = math.ceil(int(inner_most)/num_ele) * num_ele
                        res.append((j, factor, int(offset)))
        return res

'''
