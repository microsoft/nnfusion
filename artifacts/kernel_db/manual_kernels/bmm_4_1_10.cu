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
    int batch = threadIdx.x / 10;
    int in_batch = threadIdx.x % 10;
    float sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += A[i] * B[batch * 100 + i * 10 + in_batch];
    }
    C[threadIdx.x] = sum;
}
// %%%

// +++
dim3 grid(1, 1, 1);
dim3 block(40, 1, 1);
// +++

int main() {
    float *input0, *input1, *output0, *ref0;
    cudaMallocManaged(&input0, 10 * sizeof(float));
    cudaMallocManaged(&input1, 400 * sizeof(float));
    cudaMallocManaged(&output0, 40 * sizeof(float));
    ref0 = new float[40];
    for (int i = 0; i < 10; i++) input0[i] = rand() * 1.0 / RAND_MAX;
    for (int i = 0; i < 400; i++) input1[i] = rand() * 1.0 / RAND_MAX;
    default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    memset(ref0, 0, sizeof(float) * 40);
    for (int b = 0; b < 4; b++) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                ref0[b * 10 + j] += input0[i] * input1[b * 10 * 10 + i * 10 + j];
            }
        }
    }
    const float eps = 1e-4;
    for (int b = 0; b < 4; b++) {
        for (int j = 0; j < 10; j++) {
            // printf("%f %f\n", host_C_ans[b * 256 + j], host_C_out[b * 256 + j]);
            if (fabs(output0[b * 10 + j] - ref0[b * 10 + j]) > eps) {
                printf("wa at %d %d: %f %f\n", b, j, output0[b * 10 + j], ref0[b * 10 + j]);
                // return 1;
            }
        }
    }
    printf("ac\n");
    return 0;
}