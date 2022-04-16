from arch import *
from op import *
from config import Schedule
import codegen.op_impl.codegen
from codegen.op_impl.codegen import *
import tvm
from tvm.topi import nn
import sys
from tvm import te, topi
from tvm.contrib import nvcc
import time
import os
from policy import *
from cost_model import WarpBasedCostModel
from utils import *

profile_mode = "profile_time"
BACKEND = "tvm"
tmp_file = "_tmp_ele"
do_smem_tiling = False
do_reg_tiling = False

GPUs = 8
topk = 10

def unitary_main_template(source, Size, grid_size, block_size, times):
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
    'int Size = {};\n' \
    '\n' \
    '{}' \
    '\n' \
    'int main(int argc, char *argv[])\n' \
    '{{\n' \
    '    std::string path;\n' \
    '    int input_size0 = Size;\n' \
    '    int output_size = Size;\n' \
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
    '    int grid_size = {};\n' \
    '    int block_size = {};\n' \
    '    dim3 grid(grid_size, 1, 1);\n' \
    '    dim3 block(block_size, 1, 1);\n' \
    '\n' \
    '    for (int i = 0; i < {}; ++i)\n' \
    '    {{\n' \
	'        {}<<<grid, block>>>((float*)Ad, (float*)Cd);\n' \
    '        cudaDeviceSynchronize();\n' \
    '    }}\n' \
    '}}\n'.format(Size, source, grid_size, block_size, times, kernel_name)

def binary_main_template(source, SizeA, SizeB, grid_size, block_size, times):
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
    'int SizeA = {};\n' \
    'int SizeB = {};\n' \
    '\n' \
    '{}' \
    '\n' \
    'int main(int argc, char *argv[])\n' \
    '{{\n' \
    '    std::string path;\n' \
    '    int input_size0 = SizeA;\n' \
    '    int input_size1 = SizeB;\n' \
    '    int output_size = SizeA;\n' \
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
    '    int grid_size = {};\n' \
    '    int block_size = {};\n' \
    '    dim3 grid(grid_size, 1, 1);\n' \
    '    dim3 block(block_size, 1, 1);\n' \
    '\n' \
    '    for (int i = 0; i < {}; ++i)\n' \
    '    {{\n' \
	'        {}<<<grid, block>>>((float*)Ad, (float*)Bd, (float*)Cd);\n' \
    '        cudaDeviceSynchronize();\n' \
    '    }}\n' \
    '}}\n'.format(SizeA, SizeB, source, grid_size, block_size, times, kernel_name)

def Area(dims):
    ret = 1
    for dim in dims:
        ret *= dim
    return (ret, 1)

def add_layer(shape):
    area = Area(shape)
    A = te.placeholder(area, name="data1")
    B = te.placeholder(area, name="data2")
    C = te.compute((area, 1), lambda x: A[x] + B[x])
    return [A, B], C

def mul_layer(shape):
    area = Area(shape)
    A = te.placeholder(area, name="data1")
    B = te.placeholder(area, name="data2")
    C = te.compute((area, 1), lambda x: A[x] * B[x])
    return [A, B], C

def biasadd_layer(M, N):
    A = te.placeholder((M, N), name="A")
    B = te.placeholder((N,), name="B")
    C = te.compute((M, N), lambda y, x: A[y, x] + B[x])
    return [A, B], C

def tanh_layer(shape):
    area = Area(shape)
    data = te.placeholder(area, name="data")
    tanh = topi.nn.tanh(data)
    return [data], tanh

def sigmoid_layer(shape):
    area = Area(shape)
    data = te.placeholder(area, name="data")
    sigmoid = topi.nn.sigmoid(data)
    return [data], sigmoid

def relu_layer(shape):
    area = Area(shape)
    data = te.placeholder(area, name="data")
    relu = topi.nn.relu(data)
    return [data], relu

def get_elementwise_tvm_source(inputs, output, config):
    """
        inputs: list of te.tensor
        outputs: te.tensor
        config: config.sche
    """
    s = te.create_schedule(output.op)
    cgen = CodeGenerator()
    tensors = [i for i in inputs]
    tensors.append(output)
    cgen.rewrite_schedule(s, config.to_codegen_dict(), do_smem_tiling, do_reg_tiling)
    func = tvm.build(s, tensors, "cuda")
    return func.imported_modules[0].get_source()


def compile_and_run_kernel(configs, configs_idx, device_id, 
    arch, EleOP, inputs, output, SizeA, SizeB=-1):

    log_name = tmp_file + str(device_id)

    best_time = 1e100
    best_idx = 0
    for idx in range(configs_idx[device_id], configs_idx[device_id + 1]):
        config = configs[idx]

        config.dim_size += 1
        config.spatial_axis.append(last_name)
        for l in range(arch.num_level + 1):
            x = [config.get_tile(l)[0][0], 1]
            config.update_tile(l, x, reduction_dict=None)
        block_size = config.subtile_count(0, 1)
        grid_size = EleOP.get_grid_size(config.get_tile(0)[0])
        source = get_elementwise_tvm_source(inputs, output, config)
        file_name = "elmentwise-{}-{}-{}-{}".format(idx, SizeA, grid_size, block_size)
 
        with open('{}.cu'.format(file_name), 'w') as ouf:
            if isinstance(EleOP, UnitaryOp):
                main_source = unitary_main_template(source, SizeA, grid_size, block_size, 10)
            else:
                main_source = binary_main_template(source, SizeA, SizeB, grid_size, block_size, 10)
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
    os.system("export PATH=$PATH:/usr/local/cuda-10.2/bin/")

    op = "relu"
    shape = (128, 64, 112, 112)

    if len(sys.argv) > 2:
        op = sys.argv[1]
        shape = tuple([int(s) for s in sys.argv[2:]])
    print(op, shape)
    arch = V100(True)

    EleOP = None
    layer = None
    if op == "add":
        EleOP = BinaryOp(list(shape))
        SizeA = Area(shape)[0]
        SizeB = Area(shape)[0] 
        inputs, output = add_layer(shape)
    elif op == "biasadd":
        EleOP = BiasAddOp(shape[0], shape[1])
        SizeA = shape[0] * shape[1]
        SizeB = shape[0]
        inputs, output = biasadd_layer(shape[0], shape[1])
    elif op == "mul":
        EleOP = BinaryOp(list(shape))
        SizeA = Area(shape)[0]
        SizeB = Area(shape)[0]
        inputs, output = mul_layer(shape)
    elif op == "tanh":
        EleOP = UnitaryOp(list(shape))
        SizeA = Area(shape)[0]
        inputs, output = tanh_layer(shape)
    elif op == "sigmoid":
        EleOP = UnitaryOp(list(shape))
        SizeA = Area(shape)[0]
        inputs, output = sigmoid_layer(shape)
    elif op == "relu":
        EleOP = UnitaryOp(list(shape))
        SizeA = Area(shape)[0]
        inputs, output = relu_layer(shape)
    else:
        raise ValueError("unrecognized type: " + op)
    
    s = te.create_schedule(output.op)
    saxis_names = [axis.var.name for axis in s[output].op.axis]
    last_name = saxis_names[-1]
    if output.shape[-1] == 1:
        saxis_names = saxis_names[:-1]
    raxis_names = [axis.var.name for axis in s[output].op.reduce_axis]

    Tiling_Policy = ConstructionPolicyPlain(EleOP, arch, saxis_names, raxis_names, smem_tiling=do_smem_tiling, tile_tensor="output")
    
    start_time = time.time()
    configs = Tiling_Policy.emit_config_without_trails(50)[:topk]
    emit_time = time.time() - start_time

    configs_idx = alloc_configs_for_subprocess(GPUs, len(configs))
    threads = []
    for device_id in range(GPUs):
        thread = MyThread(func=compile_and_run_kernel, args=(configs, configs_idx, device_id, 
                                arch, EleOP, inputs, output, SizeA))
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
