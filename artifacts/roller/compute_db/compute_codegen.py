import tvm
from tvm import te
import os

def codegen_c(schedule, schedule_args, target_stage):
    '''
    Codegen input TVM schedule to cpu c code with full loop unrolling.

    Parameters
    ----------
    schedule: TVM schedule
    schedule_args: for tvm.build()'s args use.
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
    print(output_tensors)
    # print(input_tensors)
    # print(type(input_tensors[0]))
    # exit(0)
    for out in output_tensors:
        output_cache = schedule.cache_write(out,'global')
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
        print(space_fused)
        reduce_fused = schedule[output_cache].fuse(*reduce_axis)
        schedule[output_cache].unroll(reduce_fused)
        print(reduce_fused)
        continue

    func = tvm.build(schedule,schedule_args,target='c',target_host='c')
    return func.get_source()


def finish_kernel(input_size_list, input_name_list, output_size_list, output_name_list, src, reg_tile_size, warp_size, loop_times, kernel_repeat_time):
    '''
    Given cpu c code and register tile, shared memory tile, generate full cuda test code.

    Parameters
    ----------
    input_size_list: input parameters' size. Type: list()
    input_name_list: input parameters' name. Type: list()
    output_size_list: output parameters' size. Type: list(). Restriction: length should be 1.
    output_name_list: output parameters' name. Type: list(). Restriction: length should be 1.
    src: cpu c code.
    reg_tile_size: register tile size.
    warp_size: warp number.
    loop_times: compute loop times.
    kernel_repeat_time: kernel repeat times.

    Returns
    -------
    str: CUDA test code.
    '''
    if len(output_size_list) != 1 or len(output_name_list) != 1:
        print('more output data. function finish_kernel should be modified.')
        exit(0)

    # allocate
    allocate_str = ''
    for i in range(len(input_size_list)):
        allocate_str += '  float {}[{}];\n'.format(input_name_list[i],input_size_list[i])
    # TODO: Q: how to deal with larger than 1 output tensor?
    for i in range(len(output_name_list)):
        allocate_str += '  float {}[{}];\n'.format(output_name_list[i],output_size_list[i])
    
    # initialization
    def c_kernel_extract(src_c, compute_name):
        src_list = src_c.split('\n')
        initial_begin_pos = -1
        initial_end_pos = -1
        compute_begin_pos = -1
        compute_end_pos = -1

        for i in range(len(src_list)):
            if '0.000000e+00f' in src_list[i]:
                if compute_name not in src_list[i]:
                    print('ERROR in c_kernel_extract: compute name is not compute_global, need to rewrite the code.')
                    exit(1)
                else:
                    if initial_begin_pos == -1:
                        initial_begin_pos = i
                    initial_end_pos = i + 1
                # print(src_list[i])
            elif compute_name in src_list[i] and '=' in src_list[i] and initial_begin_pos != -1:
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

    initial_str, compute_str = c_kernel_extract(src, compute_name=output_name_list[0])
    
    # compute loop
    compute_kernel_args_list = []
    compute_kernel_call_args_list = []
    for name in input_name_list:
        compute_kernel_args_list.append('float* __restrict__ {}'.format(name)) 
        compute_kernel_call_args_list.append(name)
    for name in output_name_list:
        compute_kernel_args_list.append('float* __restrict__ {}'.format(name)) 
        compute_kernel_call_args_list.append(name)
    compute_kernel_args = ', '.join(compute_kernel_args_list)
    compute_kernel_call_args = ', '.join(compute_kernel_call_args_list)
    
    compute_loop_str = '  for (int compute_loop = 0; compute_loop < {}; ++compute_loop) {{\n' \
        '    compute_kernel({});\n' \
        '    __syncthreads();\n' \
        '  }}\n'.format(loop_times,compute_kernel_call_args)

    # compute kernel
    compute_kernel_str = '__device__ void compute_kernel({}) {{\n' \
        '{}\n' \
        '}}\n'.format(compute_kernel_args, compute_str)

    # write back
    # ad-hoc here
    wb_str = '  for (int i = 0; i < {}; ++i) {{\n' \
        '    compute[((((int)threadIdx.x) * {}) + i)] = {}[(i)];\n' \
        '  }}\n'.format(output_size_list[0], output_size_list[0], output_name_list[0])

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
        '#include "../cu_helper.h"\n' \
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
        '    int grid_size = 1;\n' \
        '    int block_size = {};\n' \
        '    dim3 grid(grid_size, 1, 1);\n' \
        '    dim3 block(block_size, 1, 1);\n' \
        '\n' \
        '    for (int i = 0; i < {}; ++i)\n' \
        '    {{\n' \
    	'        default_function_kernel0<<<grid, block>>>((float*)compute_d);\n' \
        '        cudaDeviceSynchronize();\n' \
        '    }}\n' \
        '}}\n'.format(full_kernel_code, reg_tile_size * warp_size * 32, warp_size * 32, kernel_repeat_time)

    return full_test_code

# ========================================
# enum regtile, warp(smemtile), reduction
warp = 2
reg_tile = [8,4]

# ========================================
# tvm schedule for op code
# TODO: input should be expr + enumrated config
# M = 4
# K = 4
# N = 4
# A = te.placeholder((M, K), name="A") # input name here is the same as c code.
# B = te.placeholder((K, N), name="B")
# k = te.reduce_axis((0, K), name="k")
# C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k))
# mys = te.create_schedule(C.op)

# asize = 8
# bsize = 16
# A = te.placeholder((asize,),name='A')
# B = te.placeholder((bsize,),name='B')
# C = te.compute((asize,bsize), lambda x,y: A[x] * B[y])
# mys = te.create_schedule(C.op)


N, F, HO, WO, C, KH, KW = 8,4,4,8,1,7,7
# N, F, HO, WO, C, KH, KW = 128,96,83,83,1,7,7
S, D, P = 1, 1, 0
H = (HO - 1) * S + KH - 2 * P
W = (WO - 1) * S + KW - 2 * P

data = te.placeholder((N, C, H, W), name="data")
kernel = te.placeholder((F, C, KH, KW), name="kernel")

c = te.reduce_axis((0, C), name='c')
kh = te.reduce_axis((0, KH), name='kh')
kw = te.reduce_axis((0, KW), name='kw')

conv = te.compute((N, F, HO, WO), lambda n, f, ho, wo:\
            te.sum(data[n, c, ho * S + kh * D, wo * S + kw * D] *
                    kernel[f, c, kh, kw],
                    axis=[c, kh, kw])
                    , name='conv')
mys = te.create_schedule(conv.op)

print(type(data))
print(data.shape)
print(int(data.shape[0] * data.shape[2]))
print(type(data.shape[0]))
print(type(data.shape))
exit(0)

# ========================================
# schedule codegen to c code with loop unroll
# input: tvm schedule; input + output ops (in tvm.te.placeholder format)
schedule_args = [data,kernel,conv]
# schedule_args = [A,B,C]
src = codegen_c(mys,schedule_args,'conv')

with open('test.cc','w') as ouf:
    ouf.write(src)
    ouf.close()



# ========================================
# finish the kernel code and host code
# input: input ops with size, output ops with size; reg, warp, reduction size

# kernel

# full_kernel_code = finish_kernel([16,16],['A','B'],[16],['compute_global'],1000,src)
# with open('kernel_test.cc','w') as ouf:
#     ouf.write(full_kernel_code)
#     ouf.close()

# full_code = finish_test_code(full_kernel_code,4*4,2,1)
# with open('full_test.cu','w') as ouf:
#     ouf.write(full_code)
#     ouf.close()

full_kernel_code = finish_kernel([16,16],['A','B'],[16],['conv_global'],src, 4*4, 3, 100, 1)
with open('full_test.cu','w') as ouf:
    ouf.write(full_kernel_code)
    ouf.close()

exit(0)




target_stage = 'compute'
output_tensors = []
for item in mys.stage_map.items():
    if isinstance(item[0], tvm.te.tensor.ComputeOp):
        output_num = item[0].num_outputs
        for i in range(output_num):
            if item[0].name != target_stage:
                out = item[0].output(i)
                mys[out].compute_inline()
            else:
                input_tensors = list(item[0].input_tensors)
                output_tensors.append(item[0].output(i))
# print(output_tensors)
# exit(0)
for out in output_tensors:
    output_cache = mys.cache_write(out,'global')
    space_axis = []
    reduce_axis = []
    for axis in mys[output_cache].op.axis:
        space_axis.append(axis)
    for axis in mys[output_cache].op.reduce_axis:
        # res = self.split_axis(reg_tile, axis)
        # reduce_axis = reduce_axis + res
        reduce_axis.append(axis)
    axis_order = reduce_axis + space_axis
    # print(space_axis,out)
    # continue
    mys[output_cache].reorder(*axis_order)
    space_fused = mys[output_cache].fuse(*axis_order)
    # print(space_fused)
    # continue
    mys[output_cache].unroll(space_fused)
# reg_tile = s.cache_write()
# s.unroll()
func = tvm.build(mys, [A, B, C],target='c',target_host='c')
# print(tvm.lower(s,[A,B],simple_mode=False))
# func = te.build(s, [A, B, C])
# print(func.imported_modules[0].get_source())
# print(type(func))

# print(func.get_source())

# D = tvm.sum(A * B + C) 
# D = [2, 4]
# A = [2, 1]
# B = [1, 4]
# C = [2, 4]
# wrap = 1 Reg = [2,4] Share=[64, 4], [128, 4]
# s = tvm.schedule (D)
# s.unroll()
# =>
# void kenrel(float *A, float *B, C, D)
# {
#     D[0] = A[x] dd;
#     dd
#     D;
# }
# =>
# __device__ kenrel(float *A, float *B, C, D)
# {
#     D[0] = A[x] dd;
#     dd
#     D;
# }
# __global__ kernel_entry(float *D)
# {
#     float A_local[2];
#     ..
#     ..
#     for 
#      kernel()
#     for i : reg_size:
#         D[threadIdx.x * reg_size + i] = D_local[x];
# }
# void main()
# {
#     float D[8*32];
#     kernel_entry<<1, 32>>(D)
# }