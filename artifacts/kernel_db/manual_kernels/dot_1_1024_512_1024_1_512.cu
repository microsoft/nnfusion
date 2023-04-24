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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int in_warp_id = tid & 31;
    float s = 0;
    #pragma unroll
    for (int i = in_warp_id; i < 1024; i += 32)
        s += input1[warp_id * 1024 + i] * input0[i];
    s += __shfl_xor_sync(0xffffffff, s, 16);
    s += __shfl_xor_sync(0xffffffff, s, 8);
    s += __shfl_xor_sync(0xffffffff, s, 4);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    s += __shfl_xor_sync(0xffffffff, s, 1);
    if (in_warp_id == 0) output0[warp_id] = s;
}
// %%%

// +++
dim3 grid(64, 1, 1);
dim3 block(256, 1, 1);
// +++

// input shape [1, 1024] [512, 1024] output [512]
int main() {
    float *input0, *input1, *output0;
    cudaMallocManaged(&input0, 1024 * sizeof(float));
    cudaMallocManaged(&input1, 512 * 1024 * sizeof(float));
    cudaMallocManaged(&output0, 512 * sizeof(float));
    for (int i = 0; i < 1024; i++) input0[i] = i;
    for (int i = 0; i < 1024 * 512; i++) input1[i] = i;
    default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int i = 0; i < 10; i++) printf("%f ", output0[i]);
    printf("...");
    for (int i = 502; i < 512; i++) printf("%f ", output0[i]);
    printf("\n");
    return 0;
}

// answer
// s = list(range(0, 1024))
// print([sum([x * (1024 * i + x) for x in s]) for i in range(512)])
