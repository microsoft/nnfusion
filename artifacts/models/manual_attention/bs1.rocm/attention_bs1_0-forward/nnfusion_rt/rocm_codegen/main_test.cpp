#include "nnfusion_rt.h"

#include <limits>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
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
    int64_t* Parameter_4_0_host, *Parameter_4_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_4_0_host, sizeof(int64_t)* 1));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_4_0, sizeof(int64_t) * 1));
    //input argument
    float* Parameter_5_0_host, *Parameter_5_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_5_0_host, sizeof(float)* 49152));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_5_0, sizeof(float) * 49152));
    //input argument
    float* Parameter_6_0_host, *Parameter_6_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_6_0_host, sizeof(float)* 768));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_6_0, sizeof(float) * 768));
    //input argument
    float* Parameter_7_0_host, *Parameter_7_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_7_0_host, sizeof(float)* 49152));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_7_0, sizeof(float) * 49152));

    //output arguments
    int64_t* Result_53_0_host, *Result_53_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_53_0_host, sizeof(int64_t) * 1));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_53_0, sizeof(int64_t) * 1));
    float* Result_54_0_host, *Result_54_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_54_0_host, sizeof(float) * 49152));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_54_0, sizeof(float) * 49152));
    float* Result_55_0_host, *Result_55_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_55_0_host, sizeof(float) * 768));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_55_0, sizeof(float) * 768));
    float* Result_56_0_host, *Result_56_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_56_0_host, sizeof(float) * 49152));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Result_56_0, sizeof(float) * 49152));

    // fill input values
    for (int i = 0; i < 1; ++i) Parameter_4_0_host[i] = 1.0f;
    for (int i = 0; i < 49152; ++i) Parameter_5_0_host[i] = 1.0f;
    for (int i = 0; i < 768; ++i) Parameter_6_0_host[i] = 1.0f;
    for (int i = 0; i < 49152; ++i) Parameter_7_0_host[i] = 1.0f;


    //warm up for 5 iters:
    for(int i_=0; i_< 100; i_++)
    {
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_4_0, Parameter_4_0_host, sizeof(int64_t) * 1, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_5_0, Parameter_5_0_host, sizeof(float) * 49152, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_6_0, Parameter_6_0_host, sizeof(float) * 768, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_7_0, Parameter_7_0_host, sizeof(float) * 49152, cudaMemcpyHostToDevice));
        kernel_entry(Parameter_4_0, Parameter_5_0, Parameter_6_0, Parameter_7_0, Result_53_0, Result_54_0, Result_55_0, Result_56_0);
        CUDA_SAFE_CALL(cudaMemcpy(Result_53_0_host, Result_53_0,  sizeof(int64_t) * 1, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(Result_54_0_host, Result_54_0,  sizeof(float) * 49152, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(Result_55_0_host, Result_55_0,  sizeof(float) * 768, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(Result_56_0_host, Result_56_0,  sizeof(float) * 49152, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        printf("%s \n", "Result_53_0:");
        for (int i = 0; i < 1; ++i) printf("%e ", (float)Result_53_0_host[i]); 
        printf(" .. (size = 1, ends with %e);\n", (float)Result_53_0_host[0]);
        printf("%s \n", "Result_54_0:");
        for (int i = 0; i < 10; ++i) printf("%e ", (float)Result_54_0_host[i]); 
        printf(" .. (size = 49152, ends with %e);\n", (float)Result_54_0_host[49151]);
        printf("%s \n", "Result_55_0:");
        for (int i = 0; i < 10; ++i) printf("%e ", (float)Result_55_0_host[i]); 
        printf(" .. (size = 768, ends with %e);\n", (float)Result_55_0_host[767]);
        printf("%s \n", "Result_56_0:");
        for (int i = 0; i < 10; ++i) printf("%e ", (float)Result_56_0_host[i]); 
        printf(" .. (size = 49152, ends with %e);\n", (float)Result_56_0_host[49151]);
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
    // cudaProfilerStart();
    for (int i_=0; i_<steps; i_++)
    {
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_4_0, Parameter_4_0_host, sizeof(int64_t) * 1, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_5_0, Parameter_5_0_host, sizeof(float) * 49152, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_6_0, Parameter_6_0_host, sizeof(float) * 768, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_7_0, Parameter_7_0_host, sizeof(float) * 49152, cudaMemcpyHostToDevice));
        cudaEventRecord(start_i, 0);
        kernel_entry(Parameter_4_0, Parameter_5_0, Parameter_6_0, Parameter_7_0, Result_53_0, Result_54_0, Result_55_0, Result_56_0);
        cudaEventRecord(stop_i, 0);
        cudaEventSynchronize(stop_i);
        cudaEventElapsedTime(&ms_i, start_i, stop_i);
        printf("Iteration time %f ms\n", ms_i);
        ms_total += ms_i;
        if (ms_i > ms_max)  ms_max = ms_i;
        if (ms_i < ms_min) ms_min = ms_i;
    }
    // cudaProfilerStop();

    //time measurement
    printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total / steps);

    //free context
    CUDA_SAFE_CALL(cudaFree(Parameter_4_0));
    CUDA_SAFE_CALL(cudaFree(Parameter_5_0));
    CUDA_SAFE_CALL(cudaFree(Parameter_6_0));
    CUDA_SAFE_CALL(cudaFree(Parameter_7_0));
    CUDA_SAFE_CALL(cudaFree(Result_53_0));
    CUDA_SAFE_CALL(cudaFree(Result_54_0));
    CUDA_SAFE_CALL(cudaFree(Result_55_0));
    CUDA_SAFE_CALL(cudaFree(Result_56_0));
    cuda_free();

    cudaFreeHost(Parameter_4_0_host);
    cudaFreeHost(Parameter_5_0_host);
    cudaFreeHost(Parameter_6_0_host);
    cudaFreeHost(Parameter_7_0_host);
    cudaFreeHost(Result_53_0_host);
    cudaFreeHost(Result_54_0_host);
    cudaFreeHost(Result_55_0_host);
    cudaFreeHost(Result_56_0_host);
    return 0;
}
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

