import sys
import os

def multiply(shape):
    res = 1
    for s in shape:
        res *= int(s)
    return res

def generate_matmul_cuda(source_code, grid, block, op, *shape):
    print(op)
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

    int shape[{3}] = {{{4}}};
    // allocate memory
    int size = {5};

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size * sizeof(float));
    h_B = (float*)malloc(size * sizeof(float));
    h_C = (float*)malloc(size * sizeof(float));


    float* d_A;
    float* d_B;
    float* d_C;
    checkCudaErrors(cudaMalloc(&d_A, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, size * sizeof(float)));

    checkCudaErrors(cudaMemcpy( d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice));

    //kernel call
    int steps = 100;

    // warm up
    for (int i_=0; i_<5; i_++)
    {{
        default_function_kernel0<<<grid, block>>>({6});
        cudaCheckError();
    }}

    for (int i_=0; i_<steps; i_++)
    {{
        cudaEventRecord(start_i, 0);
        // kernel_entry(Parameter_164_0, &Result_259_0);
        default_function_kernel0<<<grid, block>>>({6});
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
    """.format(source_code, ", ".join(grid), ", ".join(block), str(len(shape)), ", ".join(shape), str(multiply(shape)), "d_A, d_B, d_C" if op == "add" or op == "mul" or op == "biasadd" else "d_A, d_C")
    return generated


def main(path):
    with open(path, "r") as f:
        launch_config = f.readline()[2:].rstrip("\n").split("_")
        param = f.readline()[2:].rstrip("\n").split("_")
        print(launch_config)
        kernel = f.read()
        generated = generate_matmul_cuda(kernel, launch_config[0:3], launch_config[3:], *param)

    with open(path[:-3] + ".cu", "w") as f:
        f.write(generated)

main(sys.argv[1])