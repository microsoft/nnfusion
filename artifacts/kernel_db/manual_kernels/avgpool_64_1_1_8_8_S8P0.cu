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
    int lane_id = threadIdx.x >> 5;
    int tid = threadIdx.x & 31;
    int task_id = blockIdx.x * 8 + lane_id;
    if (task_id < 64) {
        float s = input0[task_id * 64 + tid] + input0[task_id * 64 + tid + 32];
        s += __shfl_xor_sync(0xffffffff, s, 16);
        s += __shfl_xor_sync(0xffffffff, s, 8);
        s += __shfl_xor_sync(0xffffffff, s, 4);
        s += __shfl_xor_sync(0xffffffff, s, 2);
        s += __shfl_xor_sync(0xffffffff, s, 1);
        if (tid == 0) output0[task_id] = s * 0.015625;
    }
}
// %%%

// +++
dim3 grid(8, 1, 1);
dim3 block(256, 1, 1);
// +++

int main() {
    float *input0, *output0;
    cudaMallocManaged(&input0, 64 * 64 * sizeof(float));
    cudaMallocManaged(&output0, 64 * sizeof(float));
    for (int i = 0; i < 64 * 64; i++) input0[i] = i;
    default_function_kernel0<<<grid, block>>>(input0, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int i = 0; i < 64; i++) printf("%f ", output0[i]);
    return 0;
}

// python
// s = list(range(0, 64)) 
// for i in range(15): print(sum([64 * i + x for x in s]) / 64)