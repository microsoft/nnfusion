import inspect
import json
from arch import *
from utils import *
import tvm
from tvm import te
import os
from config import *

'''
expr_db.json is a json file that contains map from expr to db json file path.

the db json file contains compute perf profiling result.

the schema of the content of the file 
should be look like:
{
    "max_compute_perf" : zzz,
    hashkey1:{
        "gflops": xxx,
        "item_throughput": yyy,
        ...
    },
    hashkey2:{
        "gflops": xxx,
        "item_throughput": yyy,
        ...
    },
    ...
}

'''

class ComputeDB():
    def __init__(self, expr, arch, compute_loop_times = 10000, fused_redu_size = 1, test_metrics = False, debug_mode = False, tiling_configs = tuple()): 
        self.expr = expr
        self.arch = arch
        self.compute_loop_times = compute_loop_times
        self.fused_redu_size = fused_redu_size
        self.test_metrics = test_metrics
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.shape_enum = tiling_configs[0]
            self.warp_size_list = tiling_configs[1]
            self.fused_redu_size_list = tiling_configs[2]
            self.reg_limit = tiling_configs[3]
            self.grid_size_list = tiling_configs[4]

        with open ('compute_db/expr_db.json', 'r') as f:
            self.expr_db = json.loads(f.read())

        key = expr.__name__
        if key not in self.expr_db:
            db_file, self.db = self.profile()
            if not self.debug_mode:
                self.expr_db[key] = db_file
                with open('compute_db/expr_db.json', 'w') as f:
                    json.dump(self.expr_db, f)
        else:
            db_file = self.expr_db[key]
            with open (db_file, 'r') as f:
                self.db = json.loads(f.read())
            
    def ComputePerf(self, rprog):

        reg_rtile = rprog.GetTile(1)
        reg_sdim = reg_rtile.SDimensions()
        reg_area = 1
        for d in reg_sdim:
            reg_area *= d
        reg_area = str(reg_area)
        warp_size = 32   
        key = "(" + reg_area + ", " + str(warp_size) + ")"
        if key not in self.db:
            print("key not in db: ", key)
            return -1
        latency = self.db[key]

        while latency == 1e+96:
            if warp_size > 4:
                warp_size -= 4
            else:
                warp_size -= 1
            key = "(" + reg_area + ", " + str(warp_size) + ")"
            if key not in self.db:
                print("key not in db: ", key)
                return -1
            latency = self.db[key]
        
        perf = warp_size / latency     
        return perf

    def MaxComputePerf(self):
        return self.db['max_compute_perf']

    def profile(self):
        # todo
        '''
        profile the compute perf of the expr,
        and save the result in a json file.
        return the path of json file.
        '''
        self.proc_expr()

        def get_tensor_size_list(tensor_list):
            size = []
            for t in tensor_list:
                s = 1
                for i in t.shape:
                    if i <= 0 and '_pad' in t.op.name:
                        assert i > 0 # shape of input tensor must be positive
                    s *= int(i)
                size.append(s)
            return size

        def enum_tiling_configs(self):
            reg_limit = self.arch._reg_cap(1)
            # from math import log2, ceil
            # maxloglimit = ceil(log2(reg_limit))
            # ini_reg_list = [2**i for i in range(maxloglimit)]
            ini_reg_list = [i+1 for i in range(reg_limit)]

            def full_permutation(input,ini_list):
                output = []
                for n in ini_list:
                    for i in input:
                        in_arr = i.copy()
                        in_arr.append(n)
                        output.append(in_arr)
                return output
            
            reg_enum = [[]]
            for _ in range(self.spatial_dim):
                reg_enum = full_permutation(reg_enum, ini_reg_list)
            
            shape_enum = []
            for reg in reg_enum:
                redu = [1 for _ in range(self.reduce_dim)]
                shape = reg + redu
                shape_enum.append(shape)
                # rt = rTile(self.expr,shape)
                # op = Op(self.expr, shape)
                # if op.RegUsage(rt) <= reg_limit:
                #     shape_enum.append(shape)
            warp_size_list = [i+1 for i in range(32)]
            fused_redu_enum = [self.fused_redu_size]
            grid_enum = [1]
            return shape_enum, warp_size_list, fused_redu_enum, reg_limit, grid_enum

        def fusedshape_to_realshape(self,key_shape_enum):
            shape_list = []
            for fused_shape in key_shape_enum:
                real_shape = [1 for i in range(self.shape_dim)]
                for i in range(self.spatial_dim):
                    real_shape[self.fused_init_axis_dict['space_{}'.format(i)][-1]] = fused_shape[i]
                for i in range(self.reduce_dim):
                    real_shape[self.fused_init_axis_dict['redu_{}'.format(i)][-1]] = fused_shape[i+self.spatial_dim]
                shape_list.append(real_shape)
            return shape_list

        def codegen_c(schedule, schedule_args, target_stage):
            '''
            Codegen input TVM schedule to cpu c code with full loop unrolling.

            Parameters
            ----------
            schedule: TVM schedule
            schedule_args: input tensor list(list, include padding tensor) + output tensor list(list, restriction: len=1)
            target_stage: output tensor's name.

            Returns
            -------
            str: cpu c code.
            '''
            output_tensors = []
            for item in schedule.stage_map.items():
                if isinstance(item[0], tvm.te.tensor.ComputeOp):
                    output_num = item[0].num_outputs
                    for i in range(output_num):
                        if item[0].name != target_stage:
                            out = item[0].output(i)
                            schedule[out].compute_inline()
                        else:
                            input_tensors = list(item[0].input_tensors)
                            output_tensors.append(item[0].output(i))
            # print(output_tensors)
            # print(input_tensors)
            # print(type(input_tensors[0]))
            # exit(0)
            for out in output_tensors:
                output_cache = schedule.cache_write(out,'global')
                if self.ispad:
                    for i in range(len(self.input_ispad)):
                        if self.input_ispad[i]:
                            schedule[schedule_args[i]].compute_inline()
                            schedule.cache_read(schedule_args[i],'global',[output_cache])
                space_axis = []
                reduce_axis = []
                for axis in schedule[output_cache].op.axis:
                    space_axis.append(axis)
                for axis in schedule[output_cache].op.reduce_axis:
                    # res = self.split_axis(reg_tile, axis)
                    # reduce_axis = reduce_axis + res
                    reduce_axis.append(axis)
                axis_order = reduce_axis + space_axis
                # print(space_axis,out)
                # continue
                schedule[output_cache].reorder(*axis_order)
                space_fused = schedule[output_cache].fuse(*space_axis)
                schedule[output_cache].unroll(space_fused)
                # print(space_fused)
                reduce_fused = schedule[output_cache].fuse(*reduce_axis)
                schedule[output_cache].unroll(reduce_fused)
                # print(reduce_fused)
                continue

            func = tvm.build(schedule,schedule_args,target='c',target_host='c')
            return func.get_source()
        
        def finish_kernel(input_tensor_list, output_tensor_list, fused_redu_size, src, warp_size, kernel_repeat_time, grid_size = 1):
            '''
            Given cpu c code and register tile, shared memory tile, generate full cuda test code.

            Parameters
            ----------
            input_tensor_list: input tensor list. Type: list()
            output_tensor_list: output tensor list. Type: list(). Restriction: length should be 1.
            fused_redu_size: fused reduction axis size.
            src: cpu c code.
            warp_size: warp number.
            kernel_repeat_time: kernel repeat times.

            Returns
            -------
            str: CUDA test code.
            '''
            
            input_size_list = get_tensor_size_list(input_tensor_list)
            output_size_list = get_tensor_size_list(output_tensor_list)
            output_tensor_name = output_tensor_list[0].op.name
            # print(input_size_list, output_size_list, output_tensor_name)

            if len(output_size_list) != 1:
                print('more output data. function finish_kernel should be modified.')
                exit(0)
            
            # allocate
            allocate_str = ''
            if self.ispad:
                for i in range(len(input_size_list)):
                    if self.input_ispad[i]:
                        allocate_str += '  float {}_global[{}];\n'.format(self.input_tensor_name_list[i],input_size_list[i])
            else:
                for i in range(len(input_size_list)):
                    allocate_str += '  float {}[{}];\n'.format(self.input_tensor_name_list[i],input_size_list[i])
            # TODO: Q: how to deal with larger than 1 output tensor?
            allocate_str += '  float {}_global[{}];\n'.format(output_tensor_name,output_size_list[0])
            
            # initialization
            def c_kernel_extract(src_c, output_tensor_name):
                compute_name = output_tensor_name + '_global'
                src_list = src_c.split('\n')
                initial_begin_pos = -1
                initial_end_pos = -1
                compute_begin_pos = -1
                compute_end_pos = -1

                for i in range(len(src_list)):
                    if '0.000000e+00f' in src_list[i] and compute_name in src_list[i]:
                        if initial_begin_pos == -1:
                            initial_begin_pos = i
                        initial_end_pos = i + 1
                        # print(src_list[i])
                    elif compute_name in src_list[i] and '=' in src_list[i] and initial_begin_pos != -1 \
                        and output_tensor_name+')' not in src_list[i] and output_tensor_name+'[' not in src_list[i]:
                        if compute_begin_pos == -1:
                            compute_begin_pos = i
                        compute_end_pos = i + 1
                    elif compute_end_pos != -1:
                        break
                initial_list = src_list[initial_begin_pos:initial_end_pos]
                compute_list = src_list[compute_begin_pos:compute_end_pos]
                initial_str = '\n'.join(initial_list)
                compute_str = '\n'.join(compute_list)
                return initial_str,compute_str

            initial_str, compute_str = c_kernel_extract(src, output_tensor_name)
            
            # compute loop
            compute_kernel_args_list = []
            compute_kernel_call_args_list = []
            if self.ispad:
                for i in range(len(self.input_tensor_name_list)):
                    if self.input_ispad[i]:
                        name = self.input_tensor_name_list[i]
                        compute_kernel_args_list.append('float* __restrict__ {}_global'.format(name)) 
                        compute_kernel_call_args_list.append('{}_global'.format(name))
                # for name in self.input_tensor_name_list:
                #     if self.input_ispad[i]:
                #         compute_kernel_args_list.append('float* __restrict__ {}_global'.format(name)) 
                #         compute_kernel_call_args_list.append(name)
            else:
                for name in self.input_tensor_name_list:
                    compute_kernel_args_list.append('float* __restrict__ {}'.format(name)) 
                    compute_kernel_call_args_list.append(name)
            compute_kernel_args_list.append('float* __restrict__ {}_global'.format(output_tensor_name)) 
            compute_kernel_call_args_list.append((output_tensor_name+ '_global'))
            compute_kernel_args = ', '.join(compute_kernel_args_list)
            compute_kernel_call_args = ', '.join(compute_kernel_call_args_list)
            
            compute_loop_str = '  for (int compute_loop = 0; compute_loop < {}; ++compute_loop) {{\n' \
                '    compute_kernel({});\n' \
                '    __syncthreads();\n' \
                '  }}\n'.format(self.compute_loop_times,compute_kernel_call_args)

            # compute kernel
            compute_kernel_unroll_k_list = [compute_str for _ in range(fused_redu_size)]
            compute_kernel_unroll_k = '\n'.join(compute_kernel_unroll_k_list)
            compute_kernel_str = '__device__ void compute_kernel({}) {{\n' \
                '{}\n' \
                '}}\n'.format(compute_kernel_args, compute_kernel_unroll_k)

            # write back
            # ad-hoc here
            if grid_size == 1:
                wb_str = '  for (int i = 0; i < {}; ++i) {{\n' \
                    '    compute[((((int)threadIdx.x) * {}) + i)] = {}_global[(i)];\n' \
                    '  }}\n'.format(output_size_list[0], output_size_list[0], output_tensor_name)
            else:
                wb_str = '  for (int i = 0; i < {}; ++i) {{\n' \
                    '    compute[((((int)blockIdx.x) * {}) + (((int)threadIdx.x) * {}) + i)] = {}_global[(i)];\n' \
                    '  }}\n'.format(output_size_list[0], output_size_list[0] * warp_size * 32, output_size_list[0], output_tensor_name)

            # full kernel
            full_kernel_code = '{}\n' \
                'extern "C" __global__ void default_function_kernel0(float* __restrict__ compute) {{\n' \
                '{}\n' \
                '{}\n' \
                '{}\n' \
                '{}\n' \
                '}}\n'.format(compute_kernel_str,allocate_str,initial_str,compute_loop_str,wb_str)
            
            # full code
            full_test_code = '#include <cuda_runtime.h>\n' \
                '#include <stdio.h>\n' \
                '#include <stdlib.h>\n' \
                '#include "cu_helper.h"\n' \
                '#include <cuda_fp16.h>\n' \
                '#include <mma.h>\n' \
                '#include <string>\n' \
                '\n' \
                '{}' \
                '\n' \
                'int main(int argc, char *argv[])\n' \
                '{{\n' \
                '    checkCudaErrors(cuInit(0));\n' \
                '    CUdevice device;\n' \
                '    checkCudaErrors(cuDeviceGet(&device, 0));\n' \
                '    CUcontext context;\n' \
                '    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));\n' \
                '\n' \
                '   float *compute_d;\n' \
                '   cudaMalloc((void **)&compute_d, {} * sizeof(float));\n' \
                '   srand(1);\n' \
                '\n' \
                '\n' \
                '    int grid_size = {};\n' \
                '    int block_size = {};\n' \
                '    dim3 grid(grid_size, 1, 1);\n' \
                '    dim3 block(block_size, 1, 1);\n' \
                '\n' \
                '    for (int i = 0; i < {}; ++i)\n' \
                '    {{\n' \
                '        default_function_kernel0<<<grid, block>>>((float*)compute_d);\n' \
                '        cudaDeviceSynchronize();\n' \
                '    }}\n' \
                '}}\n'.format(full_kernel_code, output_size_list[0] * warp_size * 32 * grid_size, grid_size, warp_size * 32, kernel_repeat_time)

            return full_test_code

        if self.debug_mode:
            key_shape_enum = self.shape_enum
            warp_size_list = self.warp_size_list
            fused_redu_size_list = self.fused_redu_size_list
            reg_limit = self.reg_limit
            grid_size_list = self.grid_size_list
        else: 
            key_shape_enum, warp_size_list, fused_redu_size_list, reg_limit, grid_size_list = enum_tiling_configs(self)
        if self.isfuse:
            shape_enum = fusedshape_to_realshape(self,key_shape_enum)
        else: shape_enum = key_shape_enum
        time_dict = {}
        time_output_list = []
        idx = 0
        shape_idx = -1
        for shape in shape_enum:
            shape_idx += 1
            if self.isfuse: input_tensor_list, output_tensor_list, _ = self.expr(shape)
            else: input_tensor_list, output_tensor_list = self.expr(shape)
            if self.ispad:  real_output_tensor_list = [output_tensor_list[self.output_pad_index]]
            else: real_output_tensor_list = [output_tensor_list[0]]
            # judge register size (can be extracted as a function or other)
            try:
                input_t_size = get_tensor_size_list(input_tensor_list)
                output_t_size = get_tensor_size_list(real_output_tensor_list)
            except AssertionError:
                print('Tensor size is negative. Input tensor:{}, Output tensor:{}'.format(input_tensor_list,output_tensor_list))
                continue
            reg_size = 0
            for i in range(len(input_t_size)):
                if self.ispad and (not self.input_ispad[i]): continue
                reg_size += input_t_size[i]
            reg_size += output_t_size[0]
            if reg_size > reg_limit: 
                print('idx = {} reg size = {} shape = {} out of max {}'.format(idx, reg_size, shape, reg_limit))
                if not self.debug_mode:
                    continue
            # print(idx, input_t_size, output_t_size, shape, key_shape_enum[shape_idx])
            # print(input_tensor_list, output_tensor_list)
            s = te.create_schedule(real_output_tensor_list[0].op)
            schedule_args = input_tensor_list + real_output_tensor_list
            if self.ispad:
                kernel_src = codegen_c(s,schedule_args,self.output_tensor_name_list[self.output_pad_index])
            else:
                kernel_src = codegen_c(s,schedule_args,self.output_tensor_name_list[0])
            # with open('{}.cc'.format('{}_{}_twooutput'.format(self.expr_name,idx)),'w') as f:
            #     f.write(kernel_src)
            #     f.close()
            # idx += 1
            # continue

            for warp_size in warp_size_list:
                for fused_redu in fused_redu_size_list:
                    for grid_size in grid_size_list:
                        if self.test_metrics:
                            full_src = finish_kernel(input_tensor_list,real_output_tensor_list,fused_redu,kernel_src,warp_size,1,grid_size)
                        else:
                            full_src = finish_kernel(input_tensor_list,real_output_tensor_list,fused_redu,kernel_src,warp_size,20,grid_size)
                        
                        file_name = '{}_{}'.format(self.expr_name,idx)
                        with open('{}.cu'.format(file_name),'w') as f:
                            f.write(full_src)
                            f.close()
                        # idx += 1
                        # continue

                        # ad-hoc here
                        if isinstance(self.arch,V100):
                            gencode = '-gencode=arch=compute_70,code=compute_70'
                        elif isinstance(self.arch,K80):
                            gencode = '-gencode=arch=compute_37,code=compute_37'
                        os.system("/usr/local/cuda-10.2/bin/nvcc {}.cu -lcuda {} -o {}".format(file_name, gencode, file_name))
                        if self.test_metrics:
                            os.system("/usr/local/cuda-10.2/bin/nvprof --metrics all --events all ./{} 2>&1 |tee {}_metrics.log".format(file_name, file_name))
                            os.system("/usr/local/cuda-10.2/bin/nvprof --print-gpu-trace ./{} 2>&1 |tee {}_gpu_trace.log".format(file_name, file_name))
                            os.system('rm {}'.format(file_name))
                        else:
                            os.system("/usr/local/cuda-10.2/bin/nvprof ./{} 2>&1 |tee _tmp_{}".format(file_name, file_name))

                            kernel_time = get_time_from_nvprof_file('_tmp_{}'.format(file_name))
                            os.system('rm _tmp_{}'.format(file_name))
                            os.system('rm {}'.format(file_name))
                            os.system('rm {}.cu'.format(file_name))
                            
                            real_spatial_key = key_shape_enum[shape_idx][:self.spatial_dim]
                            time_dict_key = str(tuple(real_spatial_key + [fused_redu, warp_size]))
                            time_dict[time_dict_key] = kernel_time / self.compute_loop_times
                            print(time_dict_key, kernel_time / self.compute_loop_times)

                            if self.debug_mode:
                                time_list = key_shape_enum[shape_idx] + [fused_redu, warp_size, grid_size, kernel_time]
                                time_output_list.append(time_list)

                        idx += 1
                        # if idx == 10:
                        #     if not self.test_metrics:
                        #         with open('compute_db/{}.json'.format(self.expr_name),'w') as f:
                        #             json.dump(time_dict, f)
                            
                        #     if self.debug_mode and (not self.test_metrics):
                        #         import pandas as pd
                        #         df = pd.DataFrame(time_output_list)
                        #         df.to_csv('compute_db/{}.csv'.format(self.expr_name))
                        #     exit(1)
                        # if idx == 5: break
        # exit(0)
        if not self.test_metrics:
            with open('compute_db/{}.json'.format(self.expr_name),'w') as f:
                json.dump(time_dict, f)
        
        if self.debug_mode and (not self.test_metrics):
            import pandas as pd
            df = pd.DataFrame(time_output_list)
            df.to_csv('compute_db/{}.csv'.format(self.expr_name))
        
        return 'compute_db/{}.json'.format(self.expr_name), time_dict

    def proc_expr(self):
        # ad-hoc here: 
        # 1. There must be 'a, b, c = shape' in expr.
        # 2. Use the first output tensor without 'unpad' for codegen and other operation (only need compute part here, no need to care about writeback.)
        # TODO: fix this question for elw expr.
        expr_str = inspect.getsource(self.expr)
        expr_str_list = expr_str.split('\n')
        self.shape_dim = -1
        expr_index = 0
        while expr_index < len(expr_str_list):
            if 'def ' in expr_str_list[expr_index]:
                self.expr_name = expr_str_list[expr_index].split('(')[0].split()[-1]
                print(self.expr_name)
            elif 'if for_rtile:' in expr_str_list[expr_index]:
                expr_index += 1
                while 'return' not in expr_str_list[expr_index]: expr_index += 1
            elif ' = shape' in expr_str_list[expr_index]:
                self.shape_dim = len(expr_str_list[expr_index].split(','))
                # print(expr_str_list[expr_index])
                break
            expr_index += 1
        assert self.shape_dim != -1

        def get_tensor_opname(tensor_list):
            name = []
            for t in tensor_list:
                # assert type(t) != 
                name.append(t.op.name)
            return name

        test_shape = [1 for _ in range(self.shape_dim)]
        out = self.expr(test_shape)
        self.input_tensor_name_list = get_tensor_opname(out[0])
        self.output_tensor_name_list = get_tensor_opname(out[1])
        # need to know the index of 
        # pad: for input size(key of compute db)
        # output without unpad: for compile
        self.ispad = False
        input_ispad = []
        for name in self.input_tensor_name_list:
            if 'pad' in name:
                self.ispad = True
                input_ispad.append(True)
            else:
                input_ispad.append(False)
        if self.ispad: 
            # input tensor is pad or not (boolean list) 
            self.input_ispad = input_ispad
            # output first pad tensor
            for i in range(len(self.output_tensor_name_list)):
                if 'unpad' not in self.output_tensor_name_list[i]:
                    self.output_pad_index = i
                    break
            # name
            saxis, raxis = get_axis_names(out[1][self.output_pad_index])
        else: 
            assert len(self.output_tensor_name_list) == 1
            saxis, raxis = get_axis_names(out[1][0])
        # TODO question: with input pad tensor, the c code can be compiled or not?
        # Fusion case:
        self.isfuse = False
        self.spatial_dim = len(saxis)
        self.reduce_dim = len(raxis)
        if self.spatial_dim + self.reduce_dim != self.shape_dim:
            self.isfuse = True
            self.fused_init_axis_dict = {}
            for i in range(len(saxis)):
                self.fused_init_axis_dict['space_{}'.format(i)] = out[2][saxis[i]]
            for i in range(len(raxis)):
                self.fused_init_axis_dict['redu_{}'.format(i)] = out[2][raxis[i]]
        #     print(self.fused_init_axis_dict)
        # print(self.spatial_dim,self.reduce_dim)
        # print(self.shape_dim)
        print(saxis,raxis)
