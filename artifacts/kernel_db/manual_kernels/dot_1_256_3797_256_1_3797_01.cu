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
    const int num_tasks = 3797;
    int block_start = 16 * 256 * blockIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int task_id_in_block = threadIdx.x >> 4;
    int in_task_id = threadIdx.x & 15;
    __shared__ float share_a[256];
    __shared__ float share_b[768]; // 36 * 64
    share_a[threadIdx.x] = input0[threadIdx.x];
    float s = 0.0;
    for (int k = 0; k < 256; k += 32) {
        #pragma unroll
        for (int i = warp_id; i < 16; i += 8) if (blockIdx.x * 16 + i < num_tasks) share_b[i * 48 + lane_id] = input1[block_start + i * 256 + k + lane_id];
        __syncthreads();
        // if (threadIdx.x == 0) { printf("shareb k=%d:", k); for (int i = 0; i < 32; i++) printf("%f ", share_b[i]); printf("\n");}
        #pragma unroll
        for (int j = in_task_id; j < 32; j += 16) s += share_a[k + j] * share_b[task_id_in_block * 48 + j];
        __syncthreads();
    }
    s += __shfl_xor_sync(0xffffffff, s, 8);
    s += __shfl_xor_sync(0xffffffff, s, 4);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    s += __shfl_xor_sync(0xffffffff, s, 1);
    if (in_task_id == 0 && blockIdx.x * 16 + task_id_in_block < num_tasks) output0[blockIdx.x * 16 + task_id_in_block] = s;
}
// %%%

// +++
dim3 grid(238, 1, 1);
dim3 block(256, 1, 1);
// +++

int main() {
    float *input0, *input1, *output0;
    cudaMallocManaged(&input0, 256 * sizeof(float));
    cudaMallocManaged(&input1, 256 * 3797 * sizeof(float));
    cudaMallocManaged(&output0, 3797 * sizeof(float));
    for (int i = 0; i < 256; i++) input0[i] = i;
    for (int i = 0; i < 256 * 3797; i++) input1[i] = i;
    default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int i = 0; i < 3797; i++) { printf("%f ", output0[i]); if (i % 32 == 31) printf("\n");}
    
    for (int i = 0; i < 1000; i++) default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    return 0;
}

// answer
// s = list(range(0, 256))
// print([sum([x * (256 * i + x) for x in s]) for i in range(3797)])
