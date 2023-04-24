import tvm
import copy
import math
import numpy as np
from tvm import te
class PolicyBase():
    def emit_raw_configs(self):
        # emit all configs without any
        #    i) pruning
        #    ii) optimization using cost model
        raise NotImplementedError

    def emit_configs_without_trails(self):
        # generate a list of configs:
        #     i) out of raw configs self.emit_raw_configs()
        #     ii) without using any trials
        raise NotImplementedError

    def update_rtile_storage_padding(self, rprog, arch, mem_level, smem_tiling, reg_tiling, st_align):
        if not st_align:
            return
        if mem_level == 0: #smem
            rtile = rprog.GetTile(mem_level)
            expr = rprog.Expression()
            shape = rprog.Dimensions()
            expr_out = expr(shape)
            in_tensors, out_tensors = expr_out[0], expr_out[1]
            if len(expr_out) == 3:
                ori_in = []
                pad_in = []
                for ins in in_tensors:
                    if '_pad' in ins.name:
                        pad_in.append(ins)
                    else:
                        ori_in.append(ins)
                out_tensor = out_tensors[0]
                write_tensor = out_tensors[-1]
                align_info = self.get_align_info_fuse(rprog, arch, smem_tiling, reg_tiling, target_stage=out_tensor.name, write_stage=write_tensor.name, st_align=st_align)
            else:
                out_tensor = out_tensors[0]
                align_info = self.get_align_info(rprog, arch, smem_tiling, reg_tiling, target_stage=out_tensor.name, st_align=st_align)
            rtile.UpdateStoragePadding(align_info)

    def get_align_info(self, rprog, arch, smem_bool=False, reg_bool=False, target_stage='compute', st_align=False):
        if not smem_bool or not st_align:
            return []
        # self.storage_align_on = st_align
        self.get_codegen_dict(rprog)
        # self.need_smem_tiling = smem_bool
        # self.need_reg_tiling = reg_bool
        schedule = rprog.GetTVMSchedule()
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
            if smem_bool:
                for input_tensor in input_tensors:
                    shared_tensor = self.sche_simu.cache_read(input_tensor, "shared", [out])
                    smem_tensor.append(shared_tensor)

            if reg_bool:
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
        return self.add_storage_align(smem_tensor, reg_tensor, arch)

    def get_align_info_fuse(self, rprog, arch, smem_bool=False, reg_bool=False, target_stage='conv2d_nchw_implicit_gemm', write_stage="output", st_align=False):
        if not smem_bool or not st_align:
            return []
        # self.storage_align_on = st_align
        # self.bank_size = bank_size
        # self.bank_number = bank_number
        self.get_codegen_dict(rprog)
        # self.need_smem_tiling = smem_bool
        # self.need_reg_tiling = reg_bool
        schedule = rprog.GetTVMSchedule()
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
            reg_tile = self.sche_simu.cache_write(out, "local")
            # print("[Add cache stage]")
            if smem_bool:
                for input_tensor in input_tensors:
                    self.sche_simu[input_tensor].compute_inline()
                    shared_tensor = self.sche_simu.cache_read(input_tensor, "shared", [reg_tile])
                    smem_tensor.append(shared_tensor)
            
            if reg_bool:
                for shared_tensor in smem_tensor:
                    local_tensor = self.sche_simu.cache_read(shared_tensor, "local", [reg_tile])
                    reg_tensor.append(local_tensor)
                # reg_tile = self.sche_simu.cache_write(out, "local")

            
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

        return self.add_storage_align(smem_tensor, reg_tensor, arch)

    def add_storage_align(self, smem_tensor, reg_tensor, arch):
        res = []
        assert len(smem_tensor) == len(reg_tensor)
        factor = arch._bank_number()
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
                        num_ele = arch._smem_bank_size() // st_size
                        assert num_ele > 0
                        factor = factor * num_ele
                        offset = math.ceil(int(inner_most)/num_ele) * num_ele
                        res.append((j, factor, int(offset)))
        return res 

    def update_thread_per_block(self, stage, sche = None, vthread=True):
        if sche == None:
            sche = self.sche
        num = 1
        # print(self.tiling)
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