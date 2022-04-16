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

class CodeGenerator:
    def __init__(self):
        self.storage_align_on = False

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
        for axis in sche[stage].op.axis:
            num = num * self.tiling[axis.var.name][1 if vthread else 0]
        self.thread_per_block = num

    def cooperative_fetch(self, shared, sch):
        axes = sch[shared].op.axis
        fused = sch[shared].fuse(*axes)
        oo, ii = sch[shared].split(fused, factor=self.thread_per_block)
        sch[shared].reorder(oo, ii)
        sch[shared].unroll(oo)
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
    def rewrite_schedule(self, schedule, tile_dict, smem_bool, reg_bool, target_stage='compute', st_align=True, bank_size=4):
        self.tiling = tile_dict
        self.binding = {"space": ["blockIdx.x", "vthread", "threadIdx.x"], "reduce": [None, None]}
        self.need_smem_tiling = smem_bool
        self.need_reg_tiling = reg_bool
        self.sche = schedule
        align_info = []
        if self.need_smem_tiling and self.need_reg_tiling and self.storage_align_on and st_align:
            align_info = self.get_align_info(schedule, target_stage)

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
            st_size = tvm.runtime.DataType(st.dtype).bits // 8
            num_ele = bank_size // st_size
            assert num_ele > 0
            factor = factor * num_ele
            offset = math.ceil(offset/num_ele) * num_ele
            self.sche[st].storage_align(st.op.axis[-2], factor, offset)

        return self.sche


    def get_align_info(self, schedule, target_stage):
        # self.tiling = tile_dict
        self.sche_simu = copy.deepcopy(schedule)
        binding = {"space": ["blockIdx.x", "threadIdx.x"], "reduce": [None, None]}
        # self.need_smem_tiling = smem_bool
        # self.need_reg_tiling = reg_bool
        # self.sche_simu = schedule

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

    def get_align_info_fuse(self, schedule, target_stage="conv2d_nchw_implicit_gemm", write_stage="output"):
        # self.tiling = tile_dict
        self.sche_simu = copy.deepcopy(schedule)
        binding = {"space": ["blockIdx.x", "threadIdx.x"], "reduce": [None, None]}
        # self.need_smem_tiling = smem_bool
        # self.need_reg_tiling = reg_bool
        # self.sche_simu = schedule

        input_tensors = []
        output_num = 0
        output_tensors = []
        write_tensor = None
        # for item in in_tensors:
        #     input_tensors.append(copy.deepcopy(item))
        # for item in out_tensors:
        #     output_tensors.append(copy.deepcopy(item))    
        # write_tensor = copy.deepcopy(w_tensor)
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
            reg_tile = self.sche_simu.cache_write(out, "local")
            # print("[Add cache stage]")
            for input_tensor in input_tensors:
                self.sche_simu[input_tensor].compute_inline()
                shared_tensor = self.sche_simu.cache_read(input_tensor, "shared", [reg_tile])
                smem_tensor.append(shared_tensor)
            
            for shared_tensor in smem_tensor:
                local_tensor = self.sche_simu.cache_read(shared_tensor, "local", [reg_tile])
                reg_tensor.append(local_tensor)
            
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
            for axis in self.sche_simu[reg_tile].op.axis:
                space_axis.append(axis)
            for axis in self.sche_simu[reg_tile].op.reduce_axis:
                res = self.split_axis(reg_tile, axis, self.sche_simu)
                reduce_axis = reduce_axis + res
            axis_order = reduce_axis + space_axis
            self.sche_simu[reg_tile].reorder(*axis_order)
            reg_fused = self.sche_simu[reg_tile].fuse(*space_axis)
            self.sche_simu[reg_tile].unroll(reg_fused)

            # print("[Cooperative fetching]")
 
            for rt in reg_tensor:
                self.sche_simu[rt].compute_at(self.sche_simu[reg_tile], reduce_axis[-1])
            for st in smem_tensor:
                self.sche_simu[st].compute_at(self.sche_simu[reg_tile], reduce_axis[0])
                self.cooperative_fetch(st, self.sche_simu)

        return self.add_storage_align(smem_tensor, reg_tensor)


    def add_storage_align(self, smem_tensor, reg_tensor):
        res = []
        assert len(smem_tensor) == len(reg_tensor)
        factor = 32
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
                        res.append((j, factor, int(inner_most)))
        return res 

    def calc_grid_(self):
        blck_dict = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
        thrd_dict = {"threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1}

        for iter, length in self.new_spatial_axis:
            if iter.var.name in self.tiling:
                factors = self.tiling[iter.var.name]
                blck = (length - 1) // int(np.prod(factors[0:])) + 1
                thrd = factors[1]
                if self.spatial_binding[0] in blck_dict:
                    blck_dict[self.spatial_binding[0]] *= blck
                if self.spatial_binding[2] in thrd_dict:
                    thrd_dict[self.spatial_binding[2]] *= thrd

        self.blck_grid = [blck_dict["blockIdx.x"], blck_dict["blockIdx.y"], blck_dict["blockIdx.z"]]
        self.thrd_grid = [thrd_dict["threadIdx.x"], thrd_dict["threadIdx.y"], thrd_dict["threadIdx.z"]]
        print("blck_grid: ", self.blck_grid, "thrd_grid: ", self.thrd_grid)
    
    def update_thread_per_block_(self, spatial_axis, vthread=True):
        num = 1
        for axis in spatial_axis:
            num = num * self.tiling[axis.var.name][1 if vthread else 0]
        self.thread_per_block = num
    
    def update_spatial_axis(self, tensor, fused_spatial_axis_base_delta):
        tvm_spatial_axis = self.sche[tensor].op.axis
        spatial_axis = []
        idx = 0
        if len(fused_spatial_axis_base_delta) == 0:
            for iter in tvm_spatial_axis:
                #spatial_axis.append((iter, iter.dom.extent))
                spatial_axis.append(iter)
        else:
            for base, delta in fused_spatial_axis_base_delta:
                while idx < base:
                    #spatial_axis.append((tvm_spatial_axis[idx], tvm_spatial_axis[idx].dom.extent))
                    spatial_axis.append(tvm_spatial_axis[idx])
                    idx += 1
                fused_axis_list = tvm_spatial_axis[base:base+delta]
                new_axis = self.sche[tensor].fuse(*list(fused_axis_list))
                #full_len = 1
                #for a in fused_axis_list:
                #    full_len *= a.dom.extent
                #spatial_axis.append((new_axis, full_len))
                spatial_axis.append(new_axis)
                idx += delta
        return spatial_axis

    def update_reduce_axis(self, tensor, fused_reduce_axis_base_delta):
        tvm_reduce_axis = self.sche[tensor].op.reduce_axis
        reduce_axis = []
        idx = 0
        if len(fused_reduce_axis_base_delta) == 0:
            for iter in tvm_reduce_axis:
                #reduce_axis.append((iter, iter.dom.extent))
                reduce_axis.append(iter)
        else:
            for base, delta in fused_reduce_axis_base_delta:
                while idx < base:
                    #reduce_axis.append((tvm_axis[idx], tvm_axis[idx].dom.extent))
                    reduce_axis.append(tvm_reduce_axis[idx])
                    idx += 1
                fused_axis_list = tvm_reduce_axis[base:base+delta]
                new_axis = self.sche[tensor].fuse(*list(fused_axis_list))
                #full_len = 1
                #for a in fused_axis_list:
                #    full_len *= a.dom.extent
                #reduce_axis.append((new_axis, full_len))
                reduce_axis.append(new_axis)
                idx += delta
        return reduce_axis

    def adjust_format_(self, spatial_axis):
        for axis in spatial_axis:
            name = axis.var.name
            if len(self.tiling[name]) == 2:
                vthrd = self.tiling[name][1]
                thrd = self.tiling[name][0]
                self.tiling[name] = [vthrd, thrd, 1]
        # print("Config:", self.tiling)

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

    def rewrite_schedule_fuse(self, schedule, sokoban_schedule, input_tensors, output_tensors, write_tensor, target_stage="conv2d_nchw_implicit_gemm", write_stage="output", st_align=True, bank_size=4):
        self.tiling = sokoban_schedule.to_codegen_dict()
        self.binding = {"space": ["blockIdx.x", "vthread", "threadIdx.x"], "reduce": [None, None]}
        self.sche = schedule
        # print("Config: {}".format(self.tiling))

        #print("Input: ", input_tensors)
        #print("Output: ", output_tensors)
        align_info = []
        if st_align and self.storage_align_on:
            align_info = self.get_align_info_fuse(schedule, target_stage, write_stage)
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
            for rt in reg_tensor:
                self.sche[rt].compute_at(self.sche[reg_tile], reduce_axis[-1])
            for st in smem_tensor:
                self.sche[st].compute_at(self.sche[reg_tile], reduce_axis[0])
                self.cooperative_fetch(st, self.sche)
        
        for info in align_info:
            idx, factor, offset = info
            st = smem_tensor[idx]
            st_size = tvm.runtime.DataType(st.dtype).bits // 8
            num_ele = bank_size // st_size
            assert num_ele > 0
            factor = factor * num_ele
            offset = math.ceil(offset/num_ele) * num_ele
            self.sche[st].storage_align(st.op.axis[-2], factor, offset)
        # assert False
        return self.sche


    def schedule_tensorcore(self, schedule, sokoban_schedule, C, verbose=False):
        """
            Schedule dense operator using Tensorcore
        """
        #if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
        #    s[B].compute_inline()
        #batch, out_dim = get_const_tuple(C.shape)
        Mdim, Ndim = C.shape
        s = schedule
        A, B = s[C].op.input_tensors
        data_dtype = A.dtype
        out_dtype = C.dtype

        # Explicit memory access
        AS = s.cache_read(A, "shared", [C])
        BS = s.cache_read(B, "shared", [C])
        AF = s.cache_read(AS, "wmma.matrix_a", [C])
        BF = s.cache_read(BS, "wmma.matrix_b", [C])
        CF = s.cache_write(C, "wmma.accumulator")
        #CS = s.cache_read(CF, "shared", [C])

        if verbose:
            print("0========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")

        # Extract Sokoban scheduling information
        warp_size = 32
        wmma_m, wmma_n = sokoban_schedule.get_tile(2)[0]
        wmma_k = sokoban_schedule.get_tile(2)[1]['k']
        warp_m, warp_n = sokoban_schedule.get_tile(1)[0]
        block_m, block_n = sokoban_schedule.get_tile(0)[0]
        rstep_size = sokoban_schedule.get_tile(0)[1]['k'] // wmma_k

        block_row_warps = block_m // warp_m
        block_col_warps = block_n // warp_n
        warp_row_tiles = warp_m // wmma_m
        warp_col_tiles = warp_n // wmma_n
        offset = 8
        offsetCS = 0
        vec = 1

        # Define the stride of intrin functions
        #CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
        #AS_stride = [AS_align, 1]
        #BS_stride = [BS_align, 1]
        #AF_stride = [wmma_k, 1]
        #BF_stride = [wmma_k, 1]
        AF_stride, AS_stride = init_intrin_strides([wmma_m, wmma_k], warp_row_tiles, block_row_warps, rstep_size, offset, "row_major")
        BF_stride, BS_stride = init_intrin_strides([wmma_k, wmma_n], warp_col_tiles, block_col_warps, rstep_size, offset, "col_major")
        AS_align = AS_stride[0]
        BS_align = BS_stride[0]
        CF_stride = [warp_col_tiles * wmma_n, 1]
        C_stride = [Ndim, 1]

        block_x = te.thread_axis("blockIdx.x")
        block_y = te.thread_axis("blockIdx.y")
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")
        thread_z = te.thread_axis("threadIdx.z")

        # Schedule for dense computation
        block_factor_m = wmma_m * warp_row_tiles * block_row_warps
        block_factor_n = wmma_n * warp_col_tiles * block_col_warps
        m, n = C.op.axis
        
        block_i, mc = s[C].split(m, factor=block_factor_m)
        block_j, nc = s[C].split(n, factor=block_factor_n)
        mm, mmi = s[C].split(mc, factor=wmma_m)
        nn, nni = s[C].split(nc, factor=wmma_n)
        mm, mmii = s[C].split(mm, factor=warp_row_tiles)
        nn, nnii = s[C].split(nn, factor=warp_col_tiles)
        s[C].reorder(block_i, block_j, mm, nn, mmii, nnii, mmi, nni)
        s[C].bind(block_i, block_x)
        s[C].bind(block_j, block_y)
        if verbose:
            print("i========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")
        s[C].bind(mm, thread_y)
        s[C].bind(nn, thread_z)
        s[C].unroll(mmii)
        s[C].unroll(nnii)

        if verbose:
            print("ii========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")

        # Schedule for wmma computation
        s[CF].compute_at(s[C], nn)
        warp_i, warp_j = CF.op.axis
        warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
        warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
        (k,) = CF.op.reduce_axis
        k, _k = s[CF].split(k, factor=wmma_k)
        ko, ki = s[CF].split(k, factor=rstep_size)
        s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)
        s[CF].unroll(ki)
        s[CF].unroll(warp_i)
        s[CF].unroll(warp_j)

        if verbose:
            print("iii========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")
        # Schedule for  wmma_matrix_a load
        s[AF].compute_at(s[CF], ki)
        m, i = AF.op.axis
        m, m_ii = s[AF].split(m, factor=wmma_m)
        i, i_jj = s[AF].split(i, factor=wmma_k)
        s[AF].reorder(m, i, m_ii, i_jj)
        s[AF].unroll(m)
        s[AF].unroll(i)

        if verbose:
            print("iv========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")
        # Schedule for  wmma_matrix_b load
        s[BF].compute_at(s[CF], ki)
        i, n = BF.op.axis
        i, i_ii = s[BF].split(i, factor=wmma_k)
        n, n_ii = s[BF].split(n, factor=wmma_n)
        s[BF].reorder(i, n, i_ii, n_ii)
        s[BF].unroll(i)
        s[BF].unroll(n)

        if verbose:
            print("v========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")
        # Schedule for A's(B's) shared memory load
        
        def shared_shedule(stage, strides):
            s[stage].compute_at(s[CF], ko)
            xo, yo = stage.op.axis
            s[stage].storage_align(xo, strides - 1, strides)
            t = s[stage].fuse(xo, yo)
            t, vi = s[stage].split(t, factor=vec)
            t, tx = s[stage].split(t, factor=warp_size)
            t, ty = s[stage].split(t, factor=block_row_warps)
            t, tz = s[stage].split(t, factor=block_col_warps)
            s[stage].bind(ty, thread_y)
            s[stage].bind(tz, thread_z)
            s[stage].bind(tx, thread_x)
            s[stage].unroll(t)
            s[stage].vectorize(vi)

        shared_shedule(AS, AS_align)
        shared_shedule(BS, BS_align)

        if verbose:
            print("vi========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")

        shape = (wmma_m, wmma_n, wmma_k)
        AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=data_dtype)
        BL_gemm = te.placeholder((wmma_k, wmma_n), name="BL_gemm", dtype=data_dtype)
        k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
        CL_compute = te.compute(
            (wmma_m, wmma_n),
            lambda ii, jj: te.sum(
                AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[k_gemm, jj].astype(out_dtype),
                axis=k_gemm,
            ),
            name="CL_compute",
        )

        # lower the computation loops down to TensorCore hardware intrinsics
        # by mapping the dense tensorcore to tensor intrinsics
        s[AF].tensorize(
            m_ii,
            intrin_wmma_load_matrix(
                (wmma_m, wmma_n, wmma_k), (warp_row_tiles, warp_col_tiles), (block_row_warps, block_col_warps), rstep_size, AF_stride, AS_stride, "wmma.matrix_a", "row_major", data_dtype
            ),
        )
        if verbose:
            print("vii========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")

        s[BF].tensorize(
            i_ii,
            intrin_wmma_load_matrix(
                (wmma_m, wmma_n, wmma_k), (warp_row_tiles, warp_col_tiles), (block_row_warps, block_col_warps), rstep_size, BF_stride, BS_stride, "wmma.matrix_b", "col_major", data_dtype
            ),
        )
        if verbose:
            print("viii========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")

        s[CF].tensorize(
            _ii, intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape)
        )
        if verbose:
            print("ix========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")

        s[C].tensorize(
            mmi,
            intrin_wmma_store_matrix(
                C_stride, CF_stride, shape, out_dtype
            ),
        )
        if verbose:
            print("x========================================================================")
            print(tvm.lower(s, [A, B, C]))
            print("========================================================================")
        return s
