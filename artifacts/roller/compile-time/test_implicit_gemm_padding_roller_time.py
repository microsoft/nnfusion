from arch import *
from op import *
from config import Schedule
import codegen.op_impl.codegen
from codegen.op_impl.codegen import *
import tvm
from tvm.topi import nn
import sys
from tvm import te
from tvm.contrib import nvcc
import time
import os
from policy import *
from cost_model import WarpBasedCostModel
from utils import *

BACKEND = "tvm"
tmp_file = "_tmp_conv"

GPUs = 8
topk = 10

def main_template(source, N, C, F, K, S, H, W, P, with_bias, grid_size_x, grid_size_y, block_size_x, block_size_y, times):
    if BACKEND == "antares":
        kernel_name = "template_op_kernel0"
    if BACKEND == "tvm":
        kernel_name = "default_function_kernel0"
    if isinstance(P,str):
        Pline = 'std::string P = \"{}\";'.format(P)
        Outsizeline = 'int output_size;\n'\
        '   if (P == std::string(\"VALID\")){\n'\
        '       output_size = N * F * ((NH - KH + 1) / S_height + 1) * ((NW - KW + 1) / S_width + 1);\n'\
        '   } else if (P == std::string(\"SAME\")){\n'\
        '       output_size = N * F * (NH / S_height + 1) * (NW / S_width + 1);\n'\
        '   }'
    elif isinstance(P,int):
        Pline = 'int P = \'{}\';'.format(P)
        Outsizeline = 'int output_size = N * F * ((NH - KH + 2 * P) / S_height + 1) * ((NW - KW + 2 * P) / S_width + 1);'
    if isinstance(S,tuple):
        Sline = 'int S_height = {}, S_width = {};'.format(S[0],S[1])
    elif isinstance(S,int):
        Sline = 'int S_height = {}, S_width = {};'.format(S,S)
    bias_line = ''
    call_name = '((float*)Ad, (float*)Bd, (float*)Cd)'
    if with_bias:
        bias_line = '    float *Dh, *Dd;\n'\
        '    Dh = (float*)malloc(output_size * sizeof(float));\n'\
        '    cudaMalloc((void **)&Dd, output_size * sizeof(float));\n'\
        '    for (int i = 0; i < output_size; ++ i)\n'\
        '        Dh[i] = 1;\n'\
        '    cudaMemcpy(Dd, Dh, output_size * sizeof(float), cudaMemcpyHostToDevice);\n'
        call_name = '((float*)Ad, (float*)Bd, (float*)Cd, (float*)Dd)'
            
    return '#include <cuda_runtime.h>\n' \
    '#include <stdio.h>\n' \
    '#include <stdlib.h>\n' \
    '#include "cu_helper.h"\n' \
    '#include <cuda_fp16.h>\n' \
    '#include <mma.h>\n' \
    '#include <string>\n' \
    '\n' \
    'int N = {}, F = {};\n' \
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
    '    int input_size1 = F * C * KH * KW;\n' \
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
    '{}\n' \
    '    dim3 grid({}, {}, 1);\n' \
    '    dim3 block({}, {}, 1);\n' \
    '    for (int i = 0; i < {}; ++i)\n' \
    '    {{\n' \
	'        {}<<<grid, block>>>{};\n' \
    '        cudaDeviceSynchronize();\n' \
    '    }}\n' \
    '}}\n'.format(N, F, C,Pline,Sline, H, K, W, K, source, Outsizeline, bias_line, grid_size_x, grid_size_y, block_size_x, block_size_y, times, kernel_name, call_name)
    

def conv2d_nchw(Input, Filter, stride, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.
    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]
    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w) // stride_w + 1)

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: te.sum(
            Input[nn, rc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w].astype(
                out_dtype
            )
            * Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx],
        ),
        tag="conv2d_nchw",
    )


def conv2d_nchw_implict_gemm(Input, Filter, stride, dilation, with_bias=False, act_relu=False, shrd_tile=(1,1,1), out_dtype=None):
    """Convolution operator in NCHW layout.
    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]
    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    Returns
    -------
    Output : tvm.te.Tensor
        2-D with shape [out_channel, batch * out_height * out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    N, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    FH = (kernel_h - 1) * dilation_h + 1
    FW = (kernel_w - 1) * dilation_w + 1

    OC = num_filter
    OH = simplify((in_height - FH) // stride_h + 1)
    OW = simplify((in_width - FW) // stride_w + 1)
    k = te.reduce_axis((0, in_channel * kernel_h * kernel_w), name="k")

    # Auto padding
    tm, tn, tk = shrd_tile
    mpad = ((OC - 1) // tm + 1) * tm
    npad = ((N * OH * OW - 1) // tn + 1) * tn
    kpad = ((in_channel*FH*FW - 1) // tk + 1) * tk
    # print(mpad, npad, kpad)


    #Filter=>4D
    # data_2d = te.compute(
    #     (in_channel*FH*FW, N * OH * OW),
    #     lambda k, j: Input[j//(OH*OW), k//(FH*FW), j % (OH*OW)//OW * stride_h + k % (FH*FW)//FW *
    #                        dilation_h, j % (OH*OW) % OW * stride_w + k % (FH*FW) % FW * dilation_w].astype(out_dtype),
    #     tag="data_2d",
    # )
    data_2d = te.compute(
        (kpad, npad),
        lambda k, j: te.if_then_else(te.all(k<in_channel*FH*FW, j < N*OH*OW), Input[j//(OH*OW), k//(FH*FW), j % (OH*OW)//OW * stride_h + k % (FH*FW)//FW *
                           dilation_h, j % (OH*OW) % OW * stride_w + k % (FH*FW) % FW * dilation_w].astype(out_dtype), 0.0), 
        tag="data_2d",
    )
    # kernel_2d = te.compute(
    #     (OC, in_channel*FH*FW),
    #     lambda i, k: Filter[i, k//(FH*FW), k % (FH*FW)//FW, k %
    #                         (FH*FW) % FW].astype(out_dtype),
    #     tag="kernel_2d",
    # )
    kernel_2d = te.compute(
        (mpad, kpad),
        lambda i, k: te.if_then_else(te.all(i<OC, k<in_channel*FH*FW), Filter[i, k//(FH*FW), k % (FH*FW)//FW, k %
                            (FH*FW) % FW].astype(out_dtype),0.0),
        tag="kernel_2d",
    )
    conv = te.compute(
        (mpad, npad),
        lambda i, j: te.sum(
            data_2d[k, j].astype(out_dtype)
            * kernel_2d[i, k].astype(out_dtype),
            axis=[k],
        ),
        tag="conv2d_nchw_implicit_gemm",
    )

    bias = None
    if with_bias:
        bias = te.placeholder((OC, N * OH * OW), name="bias")
    if with_bias and not act_relu:
        out = te.compute((OC, N * OH * OW), lambda i, j: conv[i, j] + bias[i, j], tag="output")
    elif not with_bias and act_relu:
        out = te.compute((OC, N * OH * OW), lambda i, j: te.max(conv[i, j], tvm.tir.const(0, conv.dtype)), tag="output")
    elif with_bias and act_relu:
        out = te.compute((OC, N * OH * OW), lambda i, j: te.max(conv[i, j] + bias[i, j], tvm.tir.const(0, conv.dtype)), tag="output")
    else:
        out = te.compute((OC, N * OH * OW), lambda i, j: conv[i, j], tag="output")
    return data_2d, kernel_2d, conv, out, bias


def get_implicit_gemm_tvm_source(data_2d, kernel_2d, data, kernel, conv, out, bias, config):
    s = te.create_schedule(out.op)
    cgen = CodeGenerator()
    cgen.rewrite_schedule_fuse(s, config, [data_2d, kernel_2d], [conv], out)
    if bias != None:
        func = tvm.build(s, [data, kernel, bias, conv, out], "cuda")  
    else:  
        func = tvm.build(s, [data, kernel, conv, out], "cuda")
    #print(tvm.lower(s, [data, kernel, conv, out]))
    return func.imported_modules[0].get_source()


def compile_and_run_kernel(configs, configs_idx, device_id, 
    data, kernel, S, D, with_bias, act_relu, MM, NN, N, C, H, W, F, K, P):

    log_name = tmp_file + str(device_id)

    best_time = 1e100
    best_idx = 0
    for idx in range(configs_idx[device_id], configs_idx[device_id + 1]):
        config = configs[idx]

        st = config.get_tile(0)
        data_2d, kernel_2d, conv, out, bias = conv2d_nchw_implict_gemm(
            data, kernel, S, D, with_bias, act_relu, (st[0][0], st[0][1], st[1]["k"]))

        block_size = config.subtile_count(0, 1)
        smem_tile_size = config.get_tile(0)[0]
        grid_size = ((MM - 1) // smem_tile_size[0] + 1) * ((NN - 1) // smem_tile_size[1] + 1)
        if isinstance(S,tuple):
            file_name = "implicit-gemm-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(N, C, H, W, F, K, '{}_{}'.format(S[0],S[1]), D, P, idx, grid_size, block_size)
        elif isinstance(S,int):
            file_name = "implicit-gemm-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(N, C, H, W, F, K, S, D, P, idx, grid_size, block_size)
        
        source = get_implicit_gemm_tvm_source(data_2d, kernel_2d, data, kernel, conv, out, bias, config)
        
        with open('{}.cu'.format(file_name), 'w') as ouf:
            main_source = main_template(source, N, C, F, K, S, H, W, P, with_bias, grid_size, 1, block_size, 1, 10)
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
    N, C, H, W, F, K, S, D, P = 128,336,21,21,336,1,1,1,'VALID'

    with_bias, act_relu = False, False
    if len(sys.argv) >= 11:
        N = int(sys.argv[1])
        C = int(sys.argv[2])
        H = int(sys.argv[3])
        W = int(sys.argv[4])
        F = int(sys.argv[5])
        K = int(sys.argv[6])
        K = int(sys.argv[7])
        if '(' in sys.argv[8]:
            S = tuple(list(map(int,str(sys.argv[7])[1:-1].split(';'))))
        else:
            S = int(sys.argv[8])
        D = int(sys.argv[9])
        P = str(sys.argv[10])

    if len(sys.argv) == 13:
        with_bias = str(sys.argv[11]) == 'True'
        act_relu = str(sys.argv[12]) == 'True'

    print("N, C, H, W, F, K, S, D, P:", N, C, H, W, F, K, S, D, P)
    print("With bias: {}, Fused with relu: {}".format(with_bias, act_relu))

    t0 = time.time()
    pt, pl, pd, pr = get_pad_tuple(P, (K, K))
    HI = H + pt + pd
    WI = W + pl + pr
    data = te.placeholder((N, C, HI, WI), name="data")
    kernel = te.placeholder((F, C, K, K), name="kernel")
    data_2d, kernel_2d, conv, out, bias = conv2d_nchw_implict_gemm(
        data, kernel, S, D, with_bias, act_relu)# TODO
    s = te.create_schedule(conv.op)

    saxis_names = [axis.var.name for axis in s[conv].op.axis]
    raxis_names = [axis.var.name for axis in s[conv].op.reduce_axis]

    if isinstance(S, tuple):
        S = S[0]
    HO = (HI - ((K - 1) * D + 1)) // S + 1
    WO = (WI - ((K - 1) * D + 1)) // S + 1

    MM = F
    KK = K * K * C
    NN = N * HO * WO

    op = ImplicitGemmOpV1(N, C, F, K, S, H, W, D, P)
    arch = V100()

    Tiling_Policy = ConstructionPolicyV2(op, arch, saxis_names, raxis_names)

    start_time = time.time()
    configs = Tiling_Policy.emit_config_without_trails(50)[:topk]
    emit_time = time.time() - start_time

    configs_idx = alloc_configs_for_subprocess(GPUs, len(configs))
    threads = []
    for device_id in range(GPUs):
        thread = MyThread(func=compile_and_run_kernel, args=(configs, configs_idx, device_id, 
                                data, kernel, S, D, with_bias, act_relu, MM, NN, N, C, H, W, F, K, P))
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
