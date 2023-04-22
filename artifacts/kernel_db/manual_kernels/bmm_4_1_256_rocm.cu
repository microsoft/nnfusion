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

// "Name","Calls","TotalDurationNs","AverageNs","Percentage"
// "default_function_kernel0.kd",100001,985261885,9852,100.0


// "Name","Calls","TotalDurationNs","AverageNs","Percentage"
// "default_function_kernel0.kd",100001,763293487,7632,100.0
// %%%
extern "C" __global__ void default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
    // 32 256
    __shared__ float ans[256];
    int batch_id = blockIdx.x >> 3;
    int in_batch_id = blockIdx.x & 7;
    int warp_id = threadIdx.x >> 6;
    int lane_id = threadIdx.x & 63;
    int col_id = (in_batch_id & 3) * 64 + lane_id;
    float val = 0;
    int k_start = warp_id * 32 + in_batch_id / 4 * 128;
    int k_end = (warp_id + 1) * 32 + in_batch_id / 4 * 128;
    for (int i = k_start; i < k_end; i++)
    {
        val = fma(A[i], B[batch_id * 256 * 256 + i * 256 + col_id], val);
    }
    ans[threadIdx.x] = val;
    __syncthreads();
    if (warp_id == 0)
    {
        atomicAdd(C + batch_id * 256 + col_id, ans[lane_id] + ans[lane_id + 64] + ans[lane_id + 128] + ans[lane_id + 192]);
    }
}
// %%%

// +++
dim3 grid(32, 1, 1);
dim3 block(256, 1, 1);
// +++

int main() {
    float *input0, *input1, *output0, *ref0;
    hipMallocManaged(&input0, 256 * sizeof(float));
    hipMallocManaged(&input1, 4 * 256 * 256 * sizeof(float));
    hipMallocManaged(&output0, 4 * 256 * sizeof(float));
    ref0 = new float[4 * 256];
    for (int i = 0; i < 256; i++) input0[i] = rand() * 1.0 / RAND_MAX;
    for (int i = 0; i < 4 * 256 * 256; i++) input1[i] = rand() * 1.0 / RAND_MAX;
    default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(hipDeviceSynchronize());
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
    for (int i = 0; i < 100000; i++) default_function_kernel0<<<grid, block>>>(input0, input1, output0);
    CUDA_SAFE_CALL(hipDeviceSynchronize());
    return 0;
}