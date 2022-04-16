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
tmp_file = "_tmp_dc"
do_smem_tiling = True
do_reg_tiling = True

GPUs = 8
topk = 10

def depthwise_conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Depthwise convolution nchw forward operator.
    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    Filter : tvm.te.Tensor
        4-D with shape [in_channel, channel_multiplier, filter_height, filter_width]
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
        4-D with shape [batch, out_channel, out_height, out_width]
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

    batch, in_channel, in_height, in_width = Input.shape
    # shape of dilated kernel
    filter_channel, channel_multiplier, filter_height, filter_width = Filter.shape

    dilated_kernel_h = (filter_height - 1) * dilation_h + 1
    dilated_kernel_w = (filter_width - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = simplify(in_channel * channel_multiplier)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # padding stage
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    PaddedInput = nn.pad(Input, pad_before, pad_after, name="PaddedInput")
    Kernel_4d = te.compute(
        (filter_channel, channel_multiplier, filter_height, filter_width),
        lambda c, m, x, y: Filter[c, m, x, y], tag="kernel_4d"
    )
    # depthconv stage
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, filter_height * filter_width), name="k")
    #di = te.reduce_axis((0, filter_height), name="di")
    #dj = te.reduce_axis((0, filter_width), name="dj")
    Conv = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j: te.sum(
            (
                PaddedInput[
                    b,
                    idxdiv(c, channel_multiplier),
                    #i * stride_h + di * dilation_h,
                    #j * stride_w + dj * dilation_w,
                    i * stride_h + (k // filter_width) * dilation_h,
                    j * stride_w + (k % filter_height) * dilation_w,
                ].astype(out_dtype)
                * Kernel_4d[
                    idxdiv(c, channel_multiplier), idxmod(c, channel_multiplier), k // filter_width, k % filter_height
                ].astype(out_dtype)
            ),
            axis=[k],
        ),
        name="DepthwiseConv2d",
        tag="depthwise_conv2d_nchw",
    )
    Out = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j: Conv[b, c, i, j], tag="output"
    )
    return PaddedInput, Kernel_4d, Conv, Out, out_height, out_width


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
    '    int input_size1 = C * KH * KW;\n' \
    '    {}\n' \
    '\n' \
    '    checkCudaErrors(cuInit(0));\n' \
    '    CUdevice device;\n' \
    '    checkCudaErrors(cuDeviceGet(&device, 0));\n' \
    '    CUcontext context;\n' \
    '    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));\n' \
    '\n' \
    '    float *Ah, *Bh;\n' \
    '    float *Ad, *Bd, *Cd;\n' \
    '    Ah = (float*)malloc(input_size0 * sizeof(float));\n' \
    '    Bh = (float*)malloc(input_size1 * sizeof(float));\n' \
    '\n' \
    '    cudaMalloc((void **)&Ad, input_size0 * sizeof(float));\n' \
    '    cudaMalloc((void **)&Bd, input_size1 * sizeof(float));\n' \
    '    cudaMalloc((void **)&Cd, output_size * sizeof(float));\n' \
    '\n' \
    '    srand(1);\n' \
    '    for (int i = 0; i < input_size0; ++ i)\n' \
    '        Ah[i] = 1;\n' \
    '    for (int i = 0; i < input_size1; ++ i)\n' \
    '        Bh[i] = 1;\n' \
    '\n' \
    '    cudaMemcpy(Ad, Ah, input_size0 * sizeof(float), cudaMemcpyHostToDevice);\n' \
    '    cudaMemcpy(Bd, Bh, input_size1 * sizeof(float), cudaMemcpyHostToDevice);\n' \
    '\n' \
    '    dim3 grid({}, {}, 1);\n' \
    '    dim3 block({}, {}, 1);\n' \
    '    for (int i = 0; i < {}; ++i)\n' \
    '    {{\n' \
	'        {}<<<grid, block>>>((float*)Ad, (float*)Bd, (float*)Cd);\n' \
    '        cudaDeviceSynchronize();\n' \
    '    }}\n' \
    '}}\n'.format(N, C,Pline,Sline, H, K, W, K, source,Outsizeline, grid_size_x, grid_size_y, block_size_x, block_size_y, times, kernel_name)


def get_depthwise_conv_tvm_source(data_padded, kernel_4d, data, kernel, conv, out, config):
    s = te.create_schedule(out.op)
    cgen = CodeGenerator()
    #print(tvm.lower(s, [data, kernel, conv, out]))
    cgen.rewrite_schedule_fuse(s, config, [data_padded, kernel_4d], [conv], out)
    #print(tvm.lower(s, [data, kernel, conv, out]))
    func = tvm.build(s, [data, kernel, conv, out], "cuda")
    return func.imported_modules[0].get_source()


def compile_and_run_kernel(configs, configs_idx, device_id, 
    op, data_padded, kernel_4d, data, kernel, conv, out, N, C, H, W, K, S, D, P):

    log_name = tmp_file + str(device_id)

    best_time = 1e100
    best_idx = 0
    for idx in range(configs_idx[device_id], configs_idx[device_id + 1]):
        config = configs[idx]

        block_size = config.subtile_count(0, 1)
        smem_tile_size = config.get_tile(0)[0]
        grid_size = op.get_grid_size(smem_tile_size)
        source = get_depthwise_conv_tvm_source(data_padded, kernel_4d, data, kernel, conv, out, config)
        file_name = "depthwise-conv-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(N, C, H, W, K, S, D, P, idx, grid_size, block_size)
 
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
    N, C, H, W, K, S, D, P = 128,96,165,165,7,2,1,'SAME'
    
    if len(sys.argv) == 9:
        N = int(sys.argv[1])
        C = int(sys.argv[2])
        H = int(sys.argv[3])
        W = int(sys.argv[4])
        K = int(sys.argv[5])
        S = int(sys.argv[6])
        D = int(sys.argv[7])
        P = str(sys.argv[8])
    print("N, C, H, W, K, S, D, P:", N, C, H, W, K, S, D, P)

    data = te.placeholder((N, C, H, W), name="data")
    kernel = te.placeholder((C, 1, K, K), name="kernel")
    data_padded, kernel_4d, conv, out, HO, WO = depthwise_conv2d_nchw(data, kernel, S, P, D)
    s = te.create_schedule(conv.op)

    saxis_names = [axis.var.name for axis in s[conv].op.axis]
    raxis_names = [axis.var.name for axis in s[conv].op.reduce_axis]

    if isinstance(S, tuple):
        S = S[0]

    op = DepthwiseConvOp(N, C, K, S, H, W, D, P)
    arch = V100()

    Tiling_Policy = ConstructionPolicyPlain(op, arch, saxis_names, raxis_names)
    
    start_time = time.time()
    configs = Tiling_Policy.emit_config_without_trails(50)[:topk]
    emit_time = time.time() - start_time

    configs_idx = alloc_configs_for_subprocess(GPUs, len(configs))
    threads = []
    for device_id in range(GPUs):
        thread = MyThread(func=compile_and_run_kernel, args=(configs, configs_idx, device_id, 
                                op, data_padded, kernel_4d, data, kernel, conv, out, N, C, H, W, K, S, D, P))
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
    