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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
    __shared__ float s_C[256];
    int batch_id = blockIdx.x / 4;
    int in_batch_id = blockIdx.x & 3;
    int wrap_id = threadIdx.x / 16;
    int in_wrap_id = threadIdx.x & 15;
    float s = 0;
    for (int k = wrap_id; k < 64; k += 16) {
        s += A[64 * batch_id + k] * B[64 * 64 * batch_id + 64 * k + in_batch_id * 16 + in_wrap_id];
    }
    s_C[threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.x < 128) s_C[threadIdx.x] += s_C[threadIdx.x + 128]; __syncthreads();
    if (threadIdx.x < 64) s_C[threadIdx.x] += s_C[threadIdx.x + 64]; __syncthreads();
    if (threadIdx.x < 32) {
        s = s_C[threadIdx.x];
        s += s_C[threadIdx.x + 32];
        s += __shfl_xor_sync(0xffffffff, s, 16);
        // s += __shfl_xor_sync(0xffffffff, s, 8);
        if (wrap_id == 0) {
            compute[64 * batch_id + in_batch_id * 16 + in_wrap_id] = s;
        }
    }
}
// %%%

// +++
dim3 grid(48, 1, 1);
dim3 block(256, 1, 1);
// +++

int main() {
    float *input0, *input1, *output0, *ref0;
    cudaMallocManaged(&input0, 12 * 64 * sizeof(float));
    cudaMallocManaged(&input1, 12 * 64 * 64 * sizeof(float));
    cudaMallocManaged(&output0, 12 * 64 * sizeof(float));
    ref0 = new float[128 * 256];
    for (int i = 0; i < 12 * 64; i++) input0[i] = rand() * 1.0 / RAND_MAX;
    for (int i = 0; i < 12 * 64 * 64; i++) input1[i] = rand() * 1.0 / RAND_MAX;
    default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    memset(ref0, 0, sizeof(float) * 12 * 64);
    for (int b = 0; b < 12; b++) {
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                ref0[b * 64 + j] += input0[b * 64 + i] * input1[b * 64 * 64 + i * 64 + j];
            }
        }
    }
    const float eps = 1e-4;
    for (int b = 0; b < 12; b++) {
        for (int j = 0; j < 64; j++) {
            // printf("%f %f\n", host_C_ans[b * 256 + j], host_C_out[b * 256 + j]);
            if (fabs(output0[b * 64 + j] - ref0[b * 64 + j]) > eps) {
                printf("wa at %d %d: %f %f\n", b, j, output0[b * 64 + j], ref0[b * 64 + j]);
                // return 1;
            }
        }
    }

    for (int i = 0; i < 1000; i++) {
        default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    return 0;
}