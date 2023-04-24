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
extern "C" __global__ void default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
    // 32 256
    __shared__ float ans[256];
    int batch_id = blockIdx.x >> 3;
    int in_batch_id = blockIdx.x & 7;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int col_id = in_batch_id * 32 + lane_id;
    float val = 0;
    int k_start = warp_id * 32;
    int k_end = (warp_id + 1) * 32;
    for (int i = k_start; i < k_end; i++)
    {
        val = fma(A[i], B[batch_id * 256 * 256 + i * 256 + col_id], val);
    }
    ans[threadIdx.x] = val;
    __syncthreads();
    if (warp_id == 0)
    {
        C[batch_id * 256 + col_id] = ans[lane_id] + ans[lane_id + 32] + ans[lane_id + 64] + ans[lane_id + 96] + ans[lane_id + 128] + ans[lane_id + 160] + ans[lane_id + 192] + ans[lane_id + 224];
    }
}
// %%%

// +++
dim3 grid(32, 1, 1);
dim3 block(256, 1, 1);
// +++

extern "C" __global__ void default_function_kernel1(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
    // 32 128
    __shared__ float ans[128];
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
        val = fma(A[i], B[batch_id * 256 * 256 + i * 256 + col_id], val);
    }
    ans[threadIdx.x] = val;
    __syncthreads();
    if (warp_id == 0) {
        C[batch_id * 256 + col_id] = ans[lane_id] + ans[lane_id + 32] + ans[lane_id + 64] + ans[lane_id + 96];
    }
}

extern "C" __global__ void default_function_kernel2(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
    // 32 128
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
        val = fma(A[i], B[batch_id * 256 * 256 + i * 256 + col_id], val);
    }
    if (warp_id == 0)
    {
        C[batch_id * 256 + col_id] = 0.0;
    }
    __syncthreads();
    atomicAdd(C + batch_id * 256 + col_id, val);
}

extern "C" __global__ void default_function_kernel3(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
    // 32 256
    __shared__ float ans[256];
    int batch_id = blockIdx.x >> 3;
    int in_batch_id = blockIdx.x & 7;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int col_id = in_batch_id * 32 + lane_id;
    float val = 0;
    int k_start = warp_id * 32;
    int k_end = (warp_id + 1) * 32;
    for (int i = k_start; i < k_end; i++)
    {
        val = fma(A[i], B[batch_id * 256 * 256 + i * 256 + col_id], val);
    }
    ans[threadIdx.x] = val;
    __syncthreads();
    if (warp_id == 0)
    {
        C[batch_id * 256 + col_id] = ans[lane_id] + ans[lane_id + 32] + ans[lane_id + 64] + ans[lane_id + 96] + ans[lane_id + 128] + ans[lane_id + 160] + ans[lane_id + 192] + ans[lane_id + 224];
    }
}

extern "C" __global__ void default_function_kernel4(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
    // 64 128
    __shared__ float ans[128];
    int batch_id = blockIdx.x >> 4;
    int in_batch_id = blockIdx.x & 15;
    int warp_id = threadIdx.x >> 4;
    int lane_id = threadIdx.x & 15;
    int col_id = in_batch_id * 16 + lane_id;
    float val = 0;
    int k_start = warp_id * 32;
    int k_end = (warp_id + 1) * 32;
    for (int i = k_start; i < k_end; i++)
    {
        val = fma(A[i], B[batch_id * 256 * 256 + i * 256 + col_id], val);
    }
    ans[threadIdx.x] = val;
    __syncthreads();
    if (warp_id == 0) {
        C[batch_id * 256 + col_id] = ans[lane_id] + ans[lane_id + 16] + ans[lane_id + 32] + ans[lane_id + 48] + ans[lane_id + 64] + ans[lane_id + 80] + ans[lane_id + 96] + ans[lane_id + 112];
    }
}

extern "C" __global__ void default_function_kernel5(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
    // bmm 64 256
    __shared__ float ans[256];
    int batch_id = blockIdx.x >> 4;
    int in_batch_id = blockIdx.x & 15;
    int warp_id = threadIdx.x >> 4;
    int lane_id = threadIdx.x & 15;
    int col_id = in_batch_id * 16 + lane_id;
    float val = 0;
    int k_start = warp_id * 16;
    int k_end = (warp_id + 1) * 16;
    for (int i = k_start; i < k_end; i++)
    {
        val = fma(A[i], B[batch_id * 256 * 256 + i * 256 + col_id], val);
    }
    val += __shfl_xor_sync(0xffffffff, val, 16);
    ans[threadIdx.x] = val;
    __syncthreads();
    if (warp_id == 0) {
        C[batch_id * 256 + col_id] = ans[lane_id] + ans[lane_id + 32] + ans[lane_id + 64] + ans[lane_id + 96] + ans[lane_id + 128] + ans[lane_id + 160] + ans[lane_id + 192] + ans[lane_id + 224];
    }
}

int main() {
    float *input0, *input1, *output0, *ref0;
    cudaMallocManaged(&input0, 256 * sizeof(float));
    cudaMallocManaged(&input1, 4 * 256 * 256 * sizeof(float));
    cudaMallocManaged(&output0, 4 * 256 * sizeof(float));
    ref0 = new float[4 * 256];
    for (int i = 0; i < 256; i++) input0[i] = rand() * 1.0 / RAND_MAX;
    for (int i = 0; i < 4 * 256 * 256; i++) input1[i] = rand() * 1.0 / RAND_MAX;
    default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    memset(ref0, 0, sizeof(float) * 4 * 256);
    for (int b = 0; b < 4; b++) {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                ref0[b * 256 + j] += input0[i] * input1[b * 256 * 256 + i * 256 + j];
            }
        }
    }
    const float eps = 1e-4;
    for (int b = 0; b < 4; b++) {
        for (int j = 0; j < 256; j++) {
            // printf("%f %f\n", host_C_ans[b * 256 + j], host_C_out[b * 256 + j]);
            if (fabs(output0[b * 256 + j] - ref0[b * 256 + j]) > eps) {
                printf("wa at %d %d: %f %f\n", b, j, output0[b * 256 + j], ref0[b * 256 + j]);
                // return 1;
            }
        }
    }
    printf("ac\n");
    return 0;
}