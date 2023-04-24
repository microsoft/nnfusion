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
extern "C" __global__ void default_function_kernel0(float* __restrict__ input0, float* __restrict__ output0) {
    int start_id = blockIdx.x * 28 * 28;
    int end_id = (blockIdx.x + 1) * 28 * 28;
    int tid = threadIdx.x;
    float sum = 0;
    for (int i = start_id + tid; i < end_id; i += blockDim.x) {
        sum += input0[i];
    }
    __shared__ float share_sum[256];
    share_sum[tid] = sum;
    __syncthreads();
    if (tid < 128) share_sum[tid] += share_sum[tid + 128]; __syncthreads();
    if (tid < 64) share_sum[tid] += share_sum[tid + 64]; __syncthreads();
    if (tid < 32) {
        share_sum[tid] += share_sum[tid + 32];
        float s = share_sum[tid];
        s += __shfl_xor_sync(0xffffffff, s, 16);
        s += __shfl_xor_sync(0xffffffff, s, 8);
        s += __shfl_xor_sync(0xffffffff, s, 4);
        s += __shfl_xor_sync(0xffffffff, s, 2);
        s += __shfl_xor_sync(0xffffffff, s, 1);
        if (tid == 0) output0[blockIdx.x] = s * 0.0012755102040816326; // s / 28 / 28
    }
}
// %%%

// +++
dim3 grid(512, 1, 1);
dim3 block(256, 1, 1);
// +++

int main() {
    float *input0, *output0;
    cudaMallocManaged(&input0, 512 * 28 * 28 * sizeof(float));
    cudaMallocManaged(&output0, 512 * sizeof(float));
    for (int i = 0; i < 512 * 28 * 28; i++) input0[i] = i;
    default_function_kernel0<<<grid, block>>>(input0, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int i = 0; i < 512; i++) printf("%f ", output0[i]);
    return 0;
}

// python
// s = list(range(0, 28 * 28)) 
// print([sum([28 * 28 * i + x for x in s]) / 28 / 28 for i in range(512)])
// [44608256, 111586048, 178563840, 245541632, 312519424, 379497216, 446475008, 513452800, 580430592, 647408384]
