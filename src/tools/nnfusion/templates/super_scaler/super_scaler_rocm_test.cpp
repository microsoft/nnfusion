// Microsoft (c) 2019, NNFusion Team
#include "super_scaler.h"

#include <cstdio>
#include <iostream>

#include "hip/hip_hcc.h"
#include "hip/hip_runtime.h"

#define NNSCALER_CUDACHECK(cmd)                                                                    \
    do                                                                                             \
    {                                                                                              \
        hipError_t e = cmd;                                                                        \
        if (e != hipSuccess)                                                                       \
        {                                                                                          \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, hipGetErrorString(e));   \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

void callBackFunction(void* context)
{
    std::cout << "Call Back Success!" << std::endl;
}

__global__ void add_one(float* input0)
{
    input0[threadIdx.x] += 1;
}

void allreduce_sync(float* input0, float* output0, size_t size, hipStream_t* applygradient_stream)
{
    super_scaler_all_reduce(input0, output0, size, applygradient_stream, nullptr, nullptr);
}

void allreduce_call(size_t cnt)
{
    int size = 16;
    float* gradients = new float[size * cnt];
    for (int i = 0; i < size * cnt; i++)
    {
        gradients[i] = i;
    }

    std::cout << "Before all reduce" << std::endl;
    for (int i = 0; i < size * cnt; i++)
    {
        std::cout << gradients[i] << " ";
    }
    std::cout << std::endl;

    //initializing GPU memery based on localRank
    float** sendbuff = (float**)malloc(cnt * sizeof(float*));
    float** recvbuff = (float**)malloc(cnt * sizeof(float*));

    // hipSetDevice(localRank * 1 + 0);
    for (size_t i = 0; i < cnt; i++)
    {
        hipMalloc(sendbuff + i, size * sizeof(float));
        hipMalloc(recvbuff + i, size * sizeof(float));
        hipMemset(sendbuff[i], 1, size * sizeof(float));
        hipMemset(recvbuff[i], 1, size * sizeof(float));
        hipMemcpy(sendbuff[i], gradients + size * i, size * sizeof(float), hipMemcpyHostToDevice);
    }

    hipStream_t applygradient_stream;
    hipStreamCreate(&applygradient_stream);

    for (size_t i = 0; i < cnt; i++)
    {
        allreduce_sync(sendbuff[i], recvbuff[i], size, &applygradient_stream);
        // add_one<<<1, size, 0, applygradient_stream>>>(recvbuff[i]);
    }

    hipStreamSynchronize(applygradient_stream);

    //get gradients after allreduce
    for (int i = 0; i < size * cnt; i++)
        gradients[i] = 0;

    for (size_t i = 0; i < cnt; i++)
        hipMemcpy(gradients + i * size, recvbuff[i], sizeof(float) * size, hipMemcpyDeviceToHost);

    //freeing device memory
    for (size_t i = 0; i < cnt; i++)
    {
        hipFree(sendbuff[i]);
        hipFree(recvbuff[i]);
    }

    hipStreamDestroy(applygradient_stream);
    // hipEventDestroy(applygrad_ready);

    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < cnt * size; i++)
    {
        std::cout << gradients[i] << " ";
    }
    delete[] gradients;
    std::cout << std::endl;
}

int main()
{
    super_scaler_initialization();

    std::cout << "======================================================================="
              << std::endl;
    std::cout << "allreduce: " << super_scaler_get_localrank() << std::endl;
    allreduce_call(5);

    super_scaler_finalization();

    return 0;
}
