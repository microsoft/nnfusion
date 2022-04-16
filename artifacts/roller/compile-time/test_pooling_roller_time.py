from arch import *
from op import *
from config import Schedule
import codegen.op_impl.codegen
from codegen.op_impl.codegen import *
import tvm
from tvm.topi import nn
import sys
from tvm import te, auto_scheduler
from tvm.contrib import nvcc
import time
import os
from policy import *
from cost_model import WarpBasedCostModel
from utils import *

profile_mode = "profile_time"
BACKEND = "tvm"
tmp_file = "_tmp_pool"
do_smem_tiling=False
do_reg_tiling=False

GPUs = 8
topk = 10

def pooling2d(Input, filter_height, filter_width, stride, padding, pool_type, dilation=1, out_dtype=None):
    """Pooling2d forward operator.
    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height_padded, in_width_padded]
    stride : int or a list/tuple of two ints
        The spatial stride, or (stride_height, stride_width).
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype: str, optional
        Output data type
    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch * in_channel, out_height, out_width]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch_channel_fused, in_height, in_width = Input.shape
    # shape of dilated kernel

    dilated_kernel_h = (filter_height - 1) * dilation_h + 1
    dilated_kernel_w = (filter_width - 1) * dilation_w + 1
    #pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
    #    padding, (dilated_kernel_h, dilated_kernel_w)
    #)
    out_height = simplify((in_height - dilated_kernel_h) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w) // stride_w + 1)
    
    k = te.reduce_axis((0, filter_height * filter_width), name="k")
    
    if pool_type == "max":
        Pooling = te.compute(
            (batch_channel_fused, out_height, out_width),
            lambda b, i, j: te.max(
                Input[
                    b,
                    i * stride_h + (k // filter_width) * dilation_h,
                    j * stride_w + (k % filter_width) * dilation_w,
                ].astype(out_dtype),
                axis=[k],
            ),
            name="Pool2d",
            tag="Pool2d_max",
        )

    if pool_type == "avg":
        Pooling = te.compute(
            (batch_channel_fused, out_height, out_width),
            lambda b, i, j: te.sum(
                Input[
                    b,
                    i * stride_h + (k // filter_width) * dilation_h,
                    j * stride_w + (k % filter_width) * dilation_w,
                ].astype(out_dtype) / (filter_width * filter_width),
                axis=[k],
            ),
            name="Pool2d",
            tag="Pool2d_avg",
        )

    return Pooling, out_height, out_width


def main_template(source, N, C, K, S, H, W, P, grid_size_x, grid_size_y, block_size_x, block_size_y, times):
    if BACKEND == "antares":
        kernel_name = "template_op_kernel0"
    if BACKEND == "tvm":
        kernel_name = "default_function_kernel0"
    if isinstance(P,str):
        Pline = 'std::string P = \"{}\";'.format(P)
        Outsizeline = 'int output_size;\n'\
        '   if (P == std::string(\"VALID\")){\n'\
        '       output_size = N * C * ((NH - KH + 1) / S_height + 1) * ((NW - KW + 1) / S_width + 1);\n'\
        '   } else if (P == std::string(\"SAME\")){\n'\
        '       output_size = N * C * (NH / S_height + 1) * (NW / S_width + 1);\n'\
        '   }'
    elif isinstance(P,int):
        Pline = 'int P = \'{}\';'.format(P)
        Outsizeline = 'int output_size = N * C * ((NH - KH + 2 * P) / S_height + 1) * ((NW - KW + 2 * P) / S_width + 1);'
    if isinstance(S,tuple):
        Sline = 'int S_height = {}, S_width = {};'.format(S[0],S[1])
    elif isinstance(S,int):
        Sline = 'int S_height = {}, S_width = {};'.format(S,S)

    return '#include <cuda_runtime.h>\n' \
    '#include <stdio.h>\n' \
    '#include <stdlib.h>\n' \
    '#include "cu_helper.h"\n' \
    '#include <cuda_fp16.h>\n' \
    '#include <mma.h>\n' \
    '#include <string>\n' \
    '\n' \
    'int N = {};\n' \
    'int C = {};\n' \
    '{}\n' \
    '{}\n' \
    'int NH = {}, KH = {};\n' \
    'int NW = {}, KW = {};\n' \
    '\n' \
    '{}' \
    'int main(int argc, char *argv[])\n' \
    '{{\n' \
    '    int input_size0 = N * C * (NH + KH - 1) * (NW + KW - 1);\n' \
    '    {}\n' \
    '\n' \
    '    checkCudaErrors(cuInit(0));\n' \
    '    CUdevice device;\n' \
    '    checkCudaErrors(cuDeviceGet(&device, 0));\n' \
    '    CUcontext context;\n' \
    '    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));\n' \
    '\n' \
    '    float *Ah;\n' \
    '    float *Ad, *Cd;\n' \
    '    Ah = (float*)malloc(input_size0 * sizeof(float));\n' \
    '\n' \
    '    cudaMalloc((void **)&Ad, input_size0 * sizeof(float));\n' \
    '    cudaMalloc((void **)&Cd, output_size * sizeof(float));\n' \
    '\n' \
    '    srand(1);\n' \
    '    for (int i = 0; i < input_size0; ++ i)\n' \
    '        Ah[i] = 1;\n' \
    '\n' \
    '    cudaMemcpy(Ad, Ah, input_size0 * sizeof(float), cudaMemcpyHostToDevice);\n' \
    '\n' \
    '    dim3 grid({}, {}, 1);\n' \
    '    dim3 block({}, {}, 1);\n' \
    '    for (int i = 0; i < {}; ++i)\n' \
    '    {{\n' \
	'        {}<<<grid, block>>>((float*)Cd, (float*)Ad);\n' \
    '        cudaDeviceSynchronize();\n' \
    '    }}\n' \
    '}}\n'.format(N, C,Pline,Sline, H, K, W, K, source,Outsizeline, grid_size_x, grid_size_y, block_size_x, block_size_y, times, kernel_name)


def get_pooling_tvm_source(data, pooling, config):
    s = te.create_schedule(pooling.op)
    cgen = CodeGenerator()
    #print(tvm.lower(s, [data, pooling]))
    #cgen.rewrite_schedule(s, config, [data], [pooling])
    cgen.rewrite_schedule(s, config.to_codegen_dict(), do_smem_tiling, do_reg_tiling, target_stage="Pool2d")
    #print(tvm.lower(s, [data, pooling]))
    func = tvm.build(s, [data, pooling], "cuda")
    return func.imported_modules[0].get_source()


def compile_and_run_kernel(configs, configs_idx, device_id, 
    op, data, pooling, N, C, H, W, K, S, P):

    log_name = tmp_file + str(device_id)

    best_time = 1e100
    best_idx = 0
    for idx in range(configs_idx[device_id], configs_idx[device_id + 1]):
        config = configs[idx]

        block_size = config.subtile_count(0, 1)
        grid_size = op.get_grid_size(config.get_tile(0)[0])
        source = get_pooling_tvm_source(data, pooling, config)
        file_name = "pooling-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(idx, N, C, H, W, K, S, P, grid_size, block_size)
 
        with open('{}.cu'.format(file_name), 'w') as ouf:
            main_source = main_template(source, N, C, K, S, H, W, P, grid_size, 1, block_size, 1, 10)
            ouf.write(main_source)
        
        os.system("/usr/local/cuda/bin/nvcc {}.cu -lcuda -gencode=arch=compute_70,code=compute_70 -o {} && " \
            "export CUDA_VISIBLE_DEVICES={} && "\
            "/usr/local/cuda/bin/nvprof ./{} > {} 2>&1 && " \
            "rm {} && " \
            "rm {}.cu".format(file_name, file_name, device_id, file_name, log_name, file_name, file_name))

        this_time = get_time_from_nvprof_file(log_name)
        if this_time < best_time:
            best_idx = idx
            best_time = this_time

    os.system("rm {}".format(log_name))

    return best_time, best_idx


if __name__ == "__main__":
    pool_type, N, C, H, W, K, S, P = 'avg',128,672,21,21,3,2,'SAME'
    
    if len(sys.argv) == 9:
        pool_type = sys.argv[1]
        N = int(sys.argv[2])
        C = int(sys.argv[3])
        H = int(sys.argv[4])
        W = int(sys.argv[5])
        K = int(sys.argv[6])
        S = int(sys.argv[7])
        P = str(sys.argv[8])
    print("N, C, H, W, K, S, P:", N, C, H, W, K, S, P)

    pt, pl, pd, pr = get_pad_tuple(P, (K, K))
    HI = H + pt + pd
    WI = W + pl + pr
    data = te.placeholder((N * C, HI, WI), name="data")
    pooling, ho, wo = pooling2d(data, K, K, S, P, pool_type)
    s = te.create_schedule(pooling.op)

    saxis_names = [axis.var.name for axis in s[pooling].op.axis]
    raxis_names = [axis.var.name for axis in s[pooling].op.reduce_axis]

    if isinstance(S, tuple):
        S = S[0]

    op = Pooling2dOp(N, C, K, S, H, W, 1, P)
    arch = V100()

    Tiling_Policy = ConstructionPolicyPlain(op, arch, saxis_names, raxis_names, smem_tiling=do_smem_tiling)

    start_time = time.time()
    configs = Tiling_Policy.emit_config_without_trails(50)[:topk]
    emit_time = time.time() - start_time

    configs_idx = alloc_configs_for_subprocess(GPUs, len(configs))
    threads = []
    for device_id in range(GPUs):
        thread = MyThread(func=compile_and_run_kernel, args=(configs, configs_idx, device_id, 
                                op, data, pooling, N, C, H, W, K, S, P))
        threads.append(thread)
        thread.start()
    
    best_time = 1e100
    for thread in threads:
        thread.join()
        local_best_time, local_best_idx = thread.get_result()
        if local_best_time < best_time:
            best_time = local_best_time
            best_idx = local_best_idx
    
    eval_time = time.time() - start_time

    print("best time: {} ms".format(best_time))
    print("best idx: {}".format(best_idx))
    print("best config: {}".format(configs[best_idx].dump_to_string()))
    print("top1 time: {} s".format(emit_time))
    print("top10 time: {} s".format(eval_time))
    print("total time: {} s".format(time.time() - start_time))
    