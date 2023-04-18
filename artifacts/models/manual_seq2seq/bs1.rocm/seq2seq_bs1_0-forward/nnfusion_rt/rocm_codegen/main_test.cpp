#include "nnfusion_rt.h"

#include <stdio.h>
#include <limits>
#include <stdlib.h>
#include <sstream>
#include <stdexcept>
#include <fstream>
#define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)

int main(int argc, char *argv[]){

    cuda_init();

    //input argument
    int64_t* Parameter_13_0_host, *Parameter_13_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_13_0_host, sizeof(int64_t)* 1));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_13_0, sizeof(int64_t) * 1));
    //input argument
    int64_t* Parameter_14_0_host, *Parameter_14_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_14_0_host, sizeof(int64_t)* 1));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_14_0, sizeof(int64_t) * 1));
    //input argument
    float* Parameter_15_0_host, *Parameter_15_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_15_0_host, sizeof(float)* 256));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_15_0, sizeof(float) * 256));
    //input argument
    float* Parameter_16_0_host, *Parameter_16_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_16_0_host, sizeof(float)* 256));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_16_0, sizeof(float) * 256));
    //input argument
    float* Parameter_17_0_host, *Parameter_17_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_17_0_host, sizeof(float)* 189850));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_17_0, sizeof(float) * 189850));
    //input argument
    int64_t* Parameter_18_0_host, *Parameter_18_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_18_0_host, sizeof(int64_t)* 50));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_18_0, sizeof(int64_t) * 50));

    //output arguments
    char* Result_96_0_host, *Result_96_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_96_0_host, sizeof(char) * 1));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_96_0, sizeof(char) * 1));
    int64_t* Result_97_0_host, *Result_97_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_97_0_host, sizeof(int64_t) * 1));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_97_0, sizeof(int64_t) * 1));
    int64_t* Result_98_0_host, *Result_98_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_98_0_host, sizeof(int64_t) * 50));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_98_0, sizeof(int64_t) * 50));
    float* Result_99_0_host, *Result_99_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_99_0_host, sizeof(float) * 256));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_99_0, sizeof(float) * 256));
    float* Result_100_0_host, *Result_100_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_100_0_host, sizeof(float) * 256));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_100_0, sizeof(float) * 256));
    int64_t* Result_101_0_host, *Result_101_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_101_0_host, sizeof(int64_t) * 1));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_101_0, sizeof(int64_t) * 1));

    // fill input values
    for (int i = 0; i < 1; ++i) Parameter_13_0_host[i] = 1.0f;
    for (int i = 0; i < 1; ++i) Parameter_14_0_host[i] = 1.0f;
    for (int i = 0; i < 256; ++i) Parameter_15_0_host[i] = 1.0f;
    for (int i = 0; i < 256; ++i) Parameter_16_0_host[i] = 1.0f;
    for (int i = 0; i < 189850; ++i) Parameter_17_0_host[i] = 1.0f;
    for (int i = 0; i < 50; ++i) Parameter_18_0_host[i] = 1.0f;


    //warm up for 5 iters:
    for(int i_=0; i_< 100; i_++)
    {
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_13_0, Parameter_13_0_host, sizeof(int64_t) * 1, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_14_0, Parameter_14_0_host, sizeof(int64_t) * 1, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_15_0, Parameter_15_0_host, sizeof(float) * 256, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_16_0, Parameter_16_0_host, sizeof(float) * 256, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_17_0, Parameter_17_0_host, sizeof(float) * 189850, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_18_0, Parameter_18_0_host, sizeof(int64_t) * 50, cudaMemcpyHostToDevice));
        kernel_entry(Parameter_13_0, Parameter_14_0, Parameter_15_0, Parameter_16_0, Parameter_17_0, Parameter_18_0, Result_96_0, Result_97_0, Result_98_0, Result_99_0, Result_100_0, Result_101_0);
        CUDA_SAFE_CALL(cudaMemcpy(Result_96_0_host, Result_96_0,  sizeof(char) * 1, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(Result_97_0_host, Result_97_0,  sizeof(int64_t) * 1, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(Result_98_0_host, Result_98_0,  sizeof(int64_t) * 50, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(Result_99_0_host, Result_99_0,  sizeof(float) * 256, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(Result_100_0_host, Result_100_0,  sizeof(float) * 256, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(Result_101_0_host, Result_101_0,  sizeof(int64_t) * 1, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        printf("%s \n", "Result_96_0:");
        for (int i = 0; i < 1; ++i) printf("%e ", (float)Result_96_0_host[i]); 
        printf(" .. (size = 1, ends with %e);\n", (float)Result_96_0_host[0]);
        printf("%s \n", "Result_97_0:");
        for (int i = 0; i < 1; ++i) printf("%e ", (float)Result_97_0_host[i]); 
        printf(" .. (size = 1, ends with %e);\n", (float)Result_97_0_host[0]);
        printf("%s \n", "Result_98_0:");
        for (int i = 0; i < 10; ++i) printf("%e ", (float)Result_98_0_host[i]); 
        printf(" .. (size = 50, ends with %e);\n", (float)Result_98_0_host[49]);
        printf("%s \n", "Result_99_0:");
        for (int i = 0; i < 10; ++i) printf("%e ", (float)Result_99_0_host[i]); 
        printf(" .. (size = 256, ends with %e);\n", (float)Result_99_0_host[255]);
        printf("%s \n", "Result_100_0:");
        for (int i = 0; i < 10; ++i) printf("%e ", (float)Result_100_0_host[i]); 
        printf(" .. (size = 256, ends with %e);\n", (float)Result_100_0_host[255]);
        printf("%s \n", "Result_101_0:");
        for (int i = 0; i < 1; ++i) printf("%e ", (float)Result_101_0_host[i]); 
        printf(" .. (size = 1, ends with %e);\n", (float)Result_101_0_host[0]);
    }

    //GPU time measurement
    float ms_max = std::numeric_limits<float>::min();
    float ms_min = std::numeric_limits<float>::max();
    float ms_total, ms_i;
    cudaEvent_t start_i, stop_i;
    cudaEventCreate(&start_i);
    cudaEventCreate(&stop_i);

    //time measurement
    ms_total = 0;

    //kernel call
    int steps = 100;
    
    for (int i_=0; i_<steps; i_++)
    {
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_13_0, Parameter_13_0_host, sizeof(int64_t) * 1, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_14_0, Parameter_14_0_host, sizeof(int64_t) * 1, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_15_0, Parameter_15_0_host, sizeof(float) * 256, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_16_0, Parameter_16_0_host, sizeof(float) * 256, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_17_0, Parameter_17_0_host, sizeof(float) * 189850, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_18_0, Parameter_18_0_host, sizeof(int64_t) * 50, cudaMemcpyHostToDevice));
        cudaEventRecord(start_i, 0);
        kernel_entry(Parameter_13_0, Parameter_14_0, Parameter_15_0, Parameter_16_0, Parameter_17_0, Parameter_18_0, Result_96_0, Result_97_0, Result_98_0, Result_99_0, Result_100_0, Result_101_0);
        cudaEventRecord(stop_i, 0);
        cudaEventSynchronize(stop_i);
        cudaEventElapsedTime(&ms_i, start_i, stop_i);
        printf("Iteration time %f ms\n", ms_i);
        ms_total += ms_i;
        if (ms_i > ms_max)  ms_max = ms_i;
        if (ms_i < ms_min) ms_min = ms_i;
    }
    

    //time measurement
    printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total / steps);

    //free context
    CUDA_SAFE_CALL(cudaFree(Parameter_13_0));
    CUDA_SAFE_CALL(cudaFree(Parameter_14_0));
    CUDA_SAFE_CALL(cudaFree(Parameter_15_0));
    CUDA_SAFE_CALL(cudaFree(Parameter_16_0));
    CUDA_SAFE_CALL(cudaFree(Parameter_17_0));
    CUDA_SAFE_CALL(cudaFree(Parameter_18_0));
    CUDA_SAFE_CALL(cudaFree(Result_96_0));
    CUDA_SAFE_CALL(cudaFree(Result_97_0));
    CUDA_SAFE_CALL(cudaFree(Result_98_0));
    CUDA_SAFE_CALL(cudaFree(Result_99_0));
    CUDA_SAFE_CALL(cudaFree(Result_100_0));
    CUDA_SAFE_CALL(cudaFree(Result_101_0));
    cuda_free();

    cudaFreeHost(Parameter_13_0_host);
    cudaFreeHost(Parameter_14_0_host);
    cudaFreeHost(Parameter_15_0_host);
    cudaFreeHost(Parameter_16_0_host);
    cudaFreeHost(Parameter_17_0_host);
    cudaFreeHost(Parameter_18_0_host);
    cudaFreeHost(Result_96_0_host);
    cudaFreeHost(Result_97_0_host);
    cudaFreeHost(Result_98_0_host);
    cudaFreeHost(Result_99_0_host);
    cudaFreeHost(Result_100_0_host);
    cudaFreeHost(Result_101_0_host);
    return 0;
}
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

