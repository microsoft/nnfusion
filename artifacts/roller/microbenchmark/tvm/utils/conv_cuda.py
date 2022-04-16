import sys
import subprocess
from tvm.topi.nn.utils import get_pad_tuple
import os

def generate_conv_cuda(source_code, grid, block, N, C, H, W, F, K, S, P, entry_point):
    P = get_pad_tuple(P, (int(K), int(K)))[0]
    generated = """
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <limits>

#define cudaCheckError() {{                                          \\
    cudaError_t e=cudaGetLastError();                                 \\
    if(e!=cudaSuccess) {{                                              \\
    printf("Cuda failure %s:%d: %s\\n",__FILE__,__LINE__,cudaGetErrorString(e));           \\
    exit(0); \\
    }}                                                           \\
}}

#define checkCudaErrors(func) \\
{{									\\
    cudaError_t e = (func);\\
    if(e != cudaSuccess)			              \\
        printf ("%s %d CUDA: %s\\n", __FILE__,  __LINE__, cudaGetErrorString(e)); \\
}}

{0}

int main() {{
    //GPU time measurement
    float ms_max = std::numeric_limits<float>::min();
    float ms_min = std::numeric_limits<float>::max();
    float ms_i;
    float ms_all = 0;
    cudaEvent_t start_i, stop_i;
    cudaEventCreate(&start_i);
    cudaEventCreate(&stop_i);
    
    dim3 grid({1});
    dim3 block({2});

    int N = {3};
    int C = {4};
    int H = {5};
    int W = {6};
    int F = {7};
    int K = {8};
    int S = {9};
    int P = {10};
    // allocate memory
    int input_size0 = N * C * H * W;
    int input_size1 = F * C * K * K;
    int output_size = N * F * ((H - K + 2 * P) / S + 1) * ((W - K + 2 * P) / S + 1);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(input_size0 * sizeof(float));
    h_B = (float*)malloc(input_size1 * sizeof(float));
    h_C = (float*)malloc(output_size * sizeof(float));


    float* d_A;
    float* d_B;
    float* d_C;
    checkCudaErrors(cudaMalloc(&d_A, input_size0 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, input_size1 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, output_size * sizeof(float)));

    checkCudaErrors(cudaMemcpy( d_A, h_A, input_size0 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, input_size1 * sizeof(float), cudaMemcpyHostToDevice));

    //kernel call
    int steps = 100;

    // warm up
    for (int i_=0; i_<5; i_++)
    {{
        {11}<<<grid, block>>>(d_A, d_B, d_C);
        cudaCheckError();
    }}

    for (int i_=0; i_<steps; i_++)
    {{
        cudaEventRecord(start_i, 0);
        // kernel_entry(Parameter_164_0, &Result_259_0);
        {11}<<<grid, block>>>(d_A, d_B, d_C);
        cudaCheckError();
        cudaEventRecord(stop_i, 0);
        cudaEventSynchronize(stop_i);
        cudaEventElapsedTime(&ms_i, start_i, stop_i);
        // printf("Iteration time %f ms\\n", ms_i);
        if (ms_i > ms_max)  ms_max = ms_i;
        if (ms_i < ms_min) ms_min = ms_i;
        ms_all += ms_i;
    }}
    ms_all /= steps;
    printf("Iteration time max: %f ms\\nmin: %f\\nmean: %f\\n", ms_max, ms_min, ms_all);
}}
    """.format(source_code, ", ".join(grid), ", ".join(block), N, C, H, W, F, K, S, P, entry_point)
    return generated


def execute(path):
    with open(path, "r") as f:
        launch_config = f.readline()[2:].rstrip("\n").split("_")
        param = f.readline()[2:].rstrip("\n").split("_")
        print(launch_config)
        kernel = f.read()
        filename = os.path.basename(path)
        if "ansor" in filename:
            entry_point = "default_function_kernel0"
        else:
            entry_point = "conv_kernel0"
        generated = generate_conv_cuda(kernel, launch_config[0:3], launch_config[3:], *param, entry_point)
    
    cu_file = path[:-3] + ".cu"
    exec = path[:-3] + ".out"
    with open(cu_file, "w") as f:
        f.write(generated)
    
    flag="-gencode arch=compute_70,code=sm_70"
    subprocess.run(["nvcc", cu_file, "-gencode", "arch=compute_70,code=sm_70", "-o", exec])
    subprocess.run([exec])

if __name__ == "__main__":
    execute(sys.argv[1])