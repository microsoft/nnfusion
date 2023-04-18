#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <limits>
#include <cuda_profiler_api.h>

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

// %%%
extern "C" __global__ void default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ output0)
{
    // 64 128
    int batch_id = blockIdx.x >> 3;
    int in_batch_id = blockIdx.x & 7;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int col_id = in_batch_id * 32 + lane_id;
    float val = 0;
    int k_start = warp_id * 64;
    int k_end = (warp_id + 1) * 64;
    for (int i = k_start; i < k_end; i++)
    {
        val = fma(input0[i], input1[batch_id * 256 * 256 + i * 256 + col_id], val);
    }
    if (warp_id == 0)
    {
        output0[batch_id * 256 + col_id] = 0.0;
    }
    __syncthreads();
    atomicAdd(output0 + batch_id * 256 + col_id, val);
}
// %%%

// +++
dim3 grid(64, 1, 1);
dim3 block(128, 1, 1);
// +++

int main() {
    float host_A[256];
    float host_B[8 * 256 * 256];
    float host_C_ans[8 * 256];
    float host_C_out[8 * 256];
    for (int i = 0; i < 256; i++) host_A[i] = rand() * 1.0 / RAND_MAX;
    for (int i = 0; i < 8 * 256 * 256; i++) host_B[i] = rand() * 1.0 / RAND_MAX; 
    
    float* dev_A;
    float* dev_B;
    float* dev_C;
    CUDA_SAFE_CALL(cudaMalloc(&dev_A, sizeof(float) * 256));
    CUDA_SAFE_CALL(cudaMalloc(&dev_B, sizeof(float) * 8 * 256 * 256));
    CUDA_SAFE_CALL(cudaMalloc(&dev_C, sizeof(float) * 8 * 256));
    
    CUDA_SAFE_CALL(cudaMemcpy(dev_A, host_A, sizeof(float) * 256, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_B, host_B, sizeof(float) * 8 * 256 * 256, cudaMemcpyHostToDevice));

    default_function_kernel0<<<grid, block>>>(dev_A, dev_B, dev_C);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());


    CUDA_SAFE_CALL(cudaMemcpy(host_C_out, dev_C, sizeof(float) * 8 * 256, cudaMemcpyDeviceToHost));

    memset(host_C_ans, 0, sizeof(host_C_ans));
    for (int b = 0; b < 8; b++) {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                host_C_ans[b * 256 + j] += host_A[i] * host_B[b * 256 * 256 + i * 256 + j];
            }
        }
    }

    const float eps = 1e-4;

    for (int b = 0; b < 8; b++) {
        for (int j = 0; j < 256; j++) {
            // printf("%f %f\n", host_C_ans[b * 256 + j], host_C_out[b * 256 + j]);
            if (fabs(host_C_ans[b * 256 + j] - host_C_out[b * 256 + j]) > eps) {
                printf("wa at %d %d: %f %f\n", b, j, host_C_ans[b * 256 + j], host_C_out[b * 256 + j]);
                // return 1;
            }
        }
    }
    printf("ac\n");

    return 0;
}