#include <hip/hip_runtime.h>
#include <stdexcept>
#include <sstream>

#define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        hipError_t result = (x);                                                                  \
        if (result != hipSuccess)                                                                 \
        {                                                                                          \
            const char* msg = hipGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)

// 240 256
// "Name","Calls","TotalDurationNs","AverageNs","Percentage"
// "default_function_kernel0.kd",1001,10615620,10605,100.0

// %%%
extern "C" __global__ void default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ output0) {
    const int num_tasks = 3797;
    int block_start = 16 * 256 * blockIdx.x;
    int warp_id = threadIdx.x >> 6;
    int lane_id = threadIdx.x & 63;
    int task_id_in_block = threadIdx.x >> 4;
    int in_task_id = threadIdx.x & 15;
    __shared__ float share_a[256];
    __shared__ float share_b[5120]; // 64 * 80
    share_a[threadIdx.x] = input0[threadIdx.x];
    float s = 0.0;
    const int XX = 80;
    for (int k = 0; k < 256; k += 64) {
        if (blockIdx.x * 16 + warp_id < num_tasks)
            share_b[warp_id * XX + lane_id] = input1[block_start + warp_id * 256 + k + lane_id];
        if (blockIdx.x * 16 + warp_id + 4 < num_tasks)
            share_b[(warp_id + 4) * XX + lane_id] = input1[block_start + (warp_id + 4) * 256 + k + lane_id];
        if (blockIdx.x * 16 + warp_id + 8 < num_tasks)
            share_b[(warp_id + 8) * XX + lane_id] = input1[block_start + (warp_id + 8) * 256 + k + lane_id];
        if (blockIdx.x * 16 + warp_id + 12 < num_tasks)
            share_b[(warp_id + 12) * XX + lane_id] = input1[block_start + (warp_id + 12) * 256 + k + lane_id];

        __syncthreads();
        // if (threadIdx.x == 0) { printf("shareb k=%d:", k); for (int i = 0; i < 32; i++) printf("%f ", share_b[i]); printf("\n");}
        s += share_a[k + in_task_id] * share_b[task_id_in_block * XX + in_task_id];
        s += share_a[k + in_task_id + 16] * share_b[task_id_in_block * XX + in_task_id + 16];
        s += share_a[k + in_task_id + 32] * share_b[task_id_in_block * XX + in_task_id + 32];
        s += share_a[k + in_task_id + 48] * share_b[task_id_in_block * XX + in_task_id + 48];
        __syncthreads();
    }
    s += __shfl_xor(s, 8);
    s += __shfl_xor(s, 4);
    s += __shfl_xor(s, 2);
    s += __shfl_xor(s, 1);
    if (in_task_id == 0 && blockIdx.x * 16 + task_id_in_block < num_tasks) output0[blockIdx.x * 16 + task_id_in_block] = s;
}
// %%%

// 128 256
// "Name","Calls","TotalDurationNs","AverageNs","Percentage"
// "default_function_kernel0.kd",1001,15649987,15634,100.0
extern "C" __global__ void default_function_kernel2(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ output0) {
    const int num_tasks = 3797;
    int block_start = 32 * 256 * blockIdx.x;
    int warp_id = threadIdx.x >> 6;
    int lane_id = threadIdx.x & 63;
    int task_id_in_block = threadIdx.x >> 3;
    int in_task_id = threadIdx.x & 7;
    __shared__ float share_a[256];
    __shared__ float share_b[4608]; // 64 * 72
    share_a[threadIdx.x] = input0[threadIdx.x];
    float s = 0.0;
    const int XX = 72;
    for (int k = 0; k < 256; k += 64) {
        #pragma unroll
        for (int i = warp_id; i < 32; i += 4) if (blockIdx.x * 32 + i < num_tasks) share_b[i * XX + lane_id] = input1[block_start + i * 256 + k + lane_id];
        __syncthreads();
        // if (threadIdx.x == 0) { printf("shareb k=%d:", k); for (int i = 0; i < 32; i++) printf("%f ", share_b[i]); printf("\n");}
        #pragma unroll
        for (int j = in_task_id; j < 64; j += 8) s += share_a[k + j] * share_b[task_id_in_block * XX + j];
        __syncthreads();
    }
    s += __shfl_xor(s, 4);
    s += __shfl_xor(s, 2);
    s += __shfl_xor(s, 1);
    if (in_task_id == 0 && blockIdx.x * 32 + task_id_in_block < num_tasks) output0[blockIdx.x * 32 + task_id_in_block] = s;
}
// 64 256
// "Name","Calls","TotalDurationNs","AverageNs","Percentage"
// "default_function_kernel0.kd",1001,26722383,26695,100.0
extern "C" __global__ void default_function_kernel1(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ output0) {
    const int num_tasks = 3797;
    int block_start = 64 * 256 * blockIdx.x;
    int warp_id = threadIdx.x >> 6;
    int lane_id = threadIdx.x & 63;
    int task_id_in_block = threadIdx.x >> 2;
    int in_task_id = threadIdx.x & 3;
    __shared__ float share_a[256];
    __shared__ float share_b[4352]; // 36 * 64
    share_a[threadIdx.x] = input0[threadIdx.x];
    float s = 0.0;
    const int XX = 68;
    for (int k = 0; k < 256; k += 64) {
        #pragma unroll
        for (int i = warp_id; i < 64; i += 4) if (blockIdx.x * 64 + i < num_tasks) share_b[i * XX + lane_id] = input1[block_start + i * 256 + k + lane_id];
        __syncthreads();
        // if (threadIdx.x == 0) { printf("shareb k=%d:", k); for (int i = 0; i < 32; i++) printf("%f ", share_b[i]); printf("\n");}
        #pragma unroll
        for (int j = in_task_id; j < 64; j += 4) s += share_a[k + j] * share_b[task_id_in_block * XX + j];
        __syncthreads();
    }
    s += __shfl_xor(s, 2);
    s += __shfl_xor(s, 1);
    if (in_task_id == 0 && blockIdx.x * 64 + task_id_in_block < num_tasks) output0[blockIdx.x * 64 + task_id_in_block] = s;
}

// +++
dim3 grid(240, 1, 1);
dim3 block(256, 1, 1);
// +++

int main() {
    float *input0, *input1, *output0;
    hipMallocManaged(&input0, 256 * sizeof(float));
    hipMallocManaged(&input1, 256 * 3797 * sizeof(float));
    hipMallocManaged(&output0, 3797 * sizeof(float));
    for (int i = 0; i < 256; i++) input0[i] = i;
    for (int i = 0; i < 256 * 3797; i++) input1[i] = i;
    default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(hipDeviceSynchronize());
    for (int i = 0; i < 3797; i++) { printf("%f ", output0[i]); if (i % 32 == 31) printf("\n");}
    
    for (int i = 0; i < 1000; i++) default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(hipDeviceSynchronize());
    return 0;
}

// answer
// s = list(range(0, 256))
// print([sum([x * (256 * i + x) for x in s]) for i in range(3797)])
