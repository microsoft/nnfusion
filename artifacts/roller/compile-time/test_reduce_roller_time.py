from arch import *
from op import *
from config import *
import codegen.op_impl.codegen
from codegen.op_impl.codegen import *
from tvm.contrib import nvcc
import sys
from tvm import te
import time
from policy import *
import os
from cost_model import WarpBasedCostModel
from utils import *

BACKEND = "tvm"
profile_mode = "profile_time"
tmp_file = "_tmp_reduce"
do_smem_tiling = True
do_reg_tiling = True

GPUs = 8
topk = 10

def reduce_layer(spatial_len, reduce_len):
    A = te.placeholder((spatial_len, reduce_len), name="A")
    k = te.reduce_axis((0, reduce_len), name="k")
    C = te.compute((spatial_len, 1), lambda x: te.sum(A[x, k], axis=k))
    return A, C

def main_template(source, M, N, grid_size, block_size, times):
    if BACKEND == "antares":
        kernel_name = "template_op_kernel0"
    if BACKEND == "tvm":
        kernel_name = "default_function_kernel0"
    return '#include <cuda_runtime.h>\n' \
    '#include <stdio.h>\n' \
    '#include <stdlib.h>\n' \
    '#include "cu_helper.h"\n' \
    '#include <cuda_fp16.h>\n' \
    '#include <mma.h>\n' \
    '#include <string>\n' \
    '\n' \
    'int M = {}, N = {};\n' \
    '\n' \
    '{}' \
    '\n' \
    'int main(int argc, char *argv[])\n' \
    '{{\n' \
    '    std::string path;\n' \
    '    int input_size = M * N;\n' \
    '    int output_size = M;\n' \
    '\n' \
    '    checkCudaErrors(cuInit(0));\n' \
    '    CUdevice device;\n' \
    '    checkCudaErrors(cuDeviceGet(&device, 0));\n' \
    '    CUcontext context;\n' \
    '    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));\n' \
    '\n' \
    '    float *Ah, *Ch;\n' \
    '    float *Ad, *Cd;\n' \
    '    Ah = (float*)malloc(input_size * sizeof(float));\n' \
    '    Ch = (float*)malloc(output_size * sizeof(float));\n' \
    '\n' \
    '    cudaMalloc((void **)&Ad, input_size * sizeof(float));\n' \
    '    cudaMalloc((void **)&Cd, output_size * sizeof(float));\n' \
    '\n' \
    '    srand(1);\n' \
    '    for (int i = 0; i < input_size; ++ i)\n' \
    '        Ah[i] = 1;\n' \
    '\n' \
    '    cudaMemcpy(Ad, Ah, input_size * sizeof(float), cudaMemcpyHostToDevice);\n' \
    '    cudaMemcpy(Cd, Ch, output_size * sizeof(float), cudaMemcpyHostToDevice);\n' \
    '\n' \
    '    int grid_size = {};\n' \
    '    int block_size = {};\n' \
    '    dim3 grid(grid_size, 1, 1);\n' \
    '    dim3 block(block_size, 1, 1);\n' \
    '\n' \
    '    int numBlocks;\n' \
    '    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, {}, block_size, 0);\n' \
    '    fprintf(stderr, \"Active blocks per SM = %d\\n\", numBlocks);\n ' \
    '\n' \
    '    for (int i = 0; i < {}; ++i)\n' \
    '    {{\n' \
	'        {}<<<grid, block>>>((float*)Ad, (float*)Cd);\n' \
    '        cudaDeviceSynchronize();\n' \
    '    }}\n' \
    '}}\n'.format(M, N, source, grid_size, block_size, kernel_name, times, kernel_name)

def get_reduce_tvm_source(A, C, config):
    s = te.create_schedule(C.op)
    codegen_dict = config.to_codegen_dict()
    cgen = CodeGenerator()
    cgen.rewrite_schedule(s, codegen_dict, do_smem_tiling, do_reg_tiling)
    func = tvm.build(s, [A, C], "cuda")
    return func.imported_modules[0].get_source()

def fused_axis(Dims, Reduced):
    spatial_len = 1
    reduce_len = 1
    for i in range(len(Dims) - Reduced):
        spatial_len *= Dims[i]
    for j in range(Reduced):
        reduce_len *= Dims[j + len(Dims) - Reduced]
    return spatial_len, reduce_len


def compile_and_run_kernel(configs, configs_idx, device_id, 
    op, dim_str, A, C, spatial_len, reduce_len):

    log_name = tmp_file + str(device_id)

    best_time = 1e100
    best_idx = 0
    for idx in range(configs_idx[device_id], configs_idx[device_id + 1]):
        config = configs[idx]

        block_size = config.subtile_count(0, 1)
        grid_size = op.get_grid_size(config.get_tile(0)[0])
        file_name = "reduce-{}-{}-{}-{}".format(idx, dim_str, grid_size, block_size)
        source = get_reduce_tvm_source(A, C, config)
        
        with open('{}.cu'.format(file_name), 'w') as ouf:
            main_source = main_template(source, spatial_len, reduce_len, grid_size, block_size, 10)
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
    Dims = [128, 512, 1024]
    Reduced = 1
    if len(sys.argv) > 1:
        Dims = [int(sys.argv[i + 1]) for i in range(len(sys.argv) - 2)]
        Reduced = int(sys.argv[-1])
    dim_str = str(Dims)[1:-1]
    dim_str = dim_str.replace(", ", "_")
    dim_str += "_" + str(Reduced)
    print(dim_str)
    print("Input shape: {}, Reduce Indices: {}".format(Dims, Reduced))

    t0 = time.time()
    spatial_len, reduce_len = fused_axis(Dims, Reduced)
    op = ReduceOp(spatial_len, reduce_len)
    arch = V100(True)
    A, C = reduce_layer(spatial_len, reduce_len)

    s = te.create_schedule(C.op)
    saxis_names = [axis.var.name for axis in s[C].op.axis]
    raxis_names = [axis.var.name for axis in s[C].op.reduce_axis]
    cgen = CodeGenerator()

    Tiling_Policy = ConstructionPolicyPlain(op, arch, saxis_names, raxis_names, data_type="float", smem_tiling=do_smem_tiling, tile_tensor="output")
    start_time = time.time()
    configs = Tiling_Policy.emit_config_without_trails(50)[:topk]
    emit_time = time.time() - start_time

    configs_idx = alloc_configs_for_subprocess(GPUs, len(configs))
    threads = []
    for device_id in range(GPUs):
        thread = MyThread(func=compile_and_run_kernel, args=(configs, configs_idx, device_id, 
                                op, dim_str, A, C, spatial_len, reduce_len))
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
