#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

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
extern "C" __global__ void default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ output0) {
    int tid = threadIdx.x;
    float sum = input0[tid] * input1[blockIdx.x * 1024 + tid] + input0[tid + 256] * input1[blockIdx.x * 1024 + tid + 256] + input0[tid + 512] * input1[blockIdx.x * 1024 + tid + 512] + input0[tid + 768] * input1[blockIdx.x * 1024 + tid + 768];
    __shared__ float share_sum[256];
    share_sum[tid] = sum;
    __syncthreads();
    if (tid < 128) share_sum[tid] += share_sum[tid + 128]; __syncthreads();
    if (tid < 64) share_sum[tid] += share_sum[tid + 64]; __syncthreads();
    if (tid < 32) {
        share_sum[tid] += share_sum[tid + 32]; __syncthreads();
        float s = share_sum[tid];
        s += __shfl_xor_sync(0xffffffff, s, 16);
        s += __shfl_xor_sync(0xffffffff, s, 8);
        s += __shfl_xor_sync(0xffffffff, s, 4);
        s += __shfl_xor_sync(0xffffffff, s, 2);
        s += __shfl_xor_sync(0xffffffff, s, 1);
        if (tid == 0) output0[blockIdx.x] = s;
    }
}
// %%%

// +++
dim3 grid(10, 1, 1);
dim3 block(256, 1, 1);
// +++

int main() {
    float *input0, *input1, *output0;
    cudaMallocManaged(&input0, 1024 * sizeof(float));
    cudaMallocManaged(&input1, 10240 * sizeof(float));
    cudaMallocManaged(&output0, 10 * sizeof(float));
    for (int i = 0; i < 1024; i++) input0[i] = i;
    for (int i = 0; i < 10240; i++) input1[i] = i;
    default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int i = 0; i < 10; i++) printf("%f ", output0[i]);
    return 0;
}

// answer
// s = list(range(0, 1024))
// print([sum([x * (1024 * i + x) for x in s]) for i in range(10)])
