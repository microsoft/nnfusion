// Microsoft (c) 2019, NNFusion Team
#include "super_scaler.h"

#include <cstdio>
#include <iostream>

#define NNSCALER_CUDACHECK(cmd)                                                                    \
    do                                                                                             \
    {                                                                                              \
        cudaError_t e = cmd;                                                                       \
        if (e != cudaSuccess)                                                                      \
        {                                                                                          \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
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

void allreduce_sync(float* input0, float* output0, size_t size, cudaStream_t* applygradient_stream)
{
    super_scaler_all_reduce(input0, output0, size, applygradient_stream, nullptr, nullptr);
}

void allreduce_call(size_t cnt)
{
    int localRank = super_scaler_get_localrank();
    NNSCALER_CUDACHECK(cudaSetDevice(localRank));

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

    // cudaSetDevice(localRank * 1 + 0);
    for (size_t i = 0; i < cnt; i++)
    {
        cudaMalloc(sendbuff + i, size * sizeof(float));
        cudaMalloc(recvbuff + i, size * sizeof(float));
        cudaMemset(sendbuff[i], 1, size * sizeof(float));
        cudaMemset(recvbuff[i], 1, size * sizeof(float));
        cudaMemcpy(sendbuff[i], gradients + size * i, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaStream_t applygradient_stream;
    cudaStreamCreate(&applygradient_stream);

    for (size_t i = 0; i < cnt; i++)
    {
        allreduce_sync(sendbuff[i], recvbuff[i], size, &applygradient_stream);
        add_one<<<1, size, 0, applygradient_stream>>>(recvbuff[i]);
    }

    cudaStreamSynchronize(applygradient_stream);

    //get gradients after allreduce
    for (int i = 0; i < size * cnt; i++)
        gradients[i] = 0;

    for (size_t i = 0; i < cnt; i++)
        cudaMemcpy(gradients + i * size, recvbuff[i], sizeof(float) * size, cudaMemcpyDeviceToHost);

    //freeing device memory
    for (size_t i = 0; i < cnt; i++)
    {
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
    }

    cudaStreamDestroy(applygradient_stream);
    // cudaEventDestroy(applygrad_ready); 

    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < cnt * size; i++)
    {
        std::cout << gradients[i] << " ";
    }
    delete gradients;
    std::cout << std::endl;
}

int main()
{
    super_scaler_initialization();

    std::cout << "======================================================================="
              << std::endl;
    std::cout << "allreduce: " << super_scaler_get_localrank() <<  std::endl;
    allreduce_call(5);

    super_scaler_finalization();

    return 0;
}