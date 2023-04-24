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
    __shared__ float shm[65 * 64];
    int base = blockIdx.x * 64 * 64 + threadIdx.x;
    for (int i = 0; i < 64; i++) {
        shm[i * 65 + threadIdx.x] = input0[base + i * 64];
    }
    __syncthreads();
    float s = 0;
    for (int i = 0; i < 64; i++)
        s += shm[65 * threadIdx.x + i];
    output0[blockIdx.x * 64 + threadIdx.x] = s * 0.015625;
}
// %%%

// +++
dim3 grid(64, 1, 1);
dim3 block(64, 1, 1);
// +++

int main() {
    float *input0, *output0;
    cudaMallocManaged(&input0, 64 * 64 * 64 * sizeof(float));
    cudaMallocManaged(&output0, 64 * 64 * sizeof(float));
    for (int i = 0; i < 64 * 64 * 64; i++) input0[i] = i;
    default_function_kernel0<<<grid, block>>>(input0, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int i = 0; i < 64 * 64; i++) {
        printf("%f ", output0[i]);
        if (i % 64 == 63) printf("\n");
    }
    return 0;
}

// python
// s = list(range(0, 64))
// >>> lst = [sum([(64 * i + x) for x in s]) / 64 for i in range(4096)]
// >>> lst[:64]
// [31.5, 95.5, 159.5, 223.5, 287.5, 351.5, 415.5, 479.5, 543.5, 607.5, 671.5, 735.5, 799.5, 863.5, 927.5, 991.5, 1055.5, 1119.5, 1183.5, 1247.5, 1311.5, 1375.5, 1439.5, 1503.5, 1567.5, 1631.5, 1695.5, 1759.5, 1823.5, 1887.5, 1951.5, 2015.5, 2079.5, 2143.5, 2207.5, 2271.5, 2335.5, 2399.5, 2463.5, 2527.5, 2591.5, 2655.5, 2719.5, 2783.5, 2847.5, 2911.5, 2975.5, 3039.5, 3103.5, 3167.5, 3231.5, 3295.5, 3359.5, 3423.5, 3487.5, 3551.5, 3615.5, 3679.5, 3743.5, 3807.5, 3871.5, 3935.5, 3999.5, 4063.5]
// >>> lst[-64:]
// [258079.5, 258143.5, 258207.5, 258271.5, 258335.5, 258399.5, 258463.5, 258527.5, 258591.5, 258655.5, 258719.5, 258783.5, 258847.5, 258911.5, 258975.5, 259039.5, 259103.5, 259167.5, 259231.5, 259295.5, 259359.5, 259423.5, 259487.5, 259551.5, 259615.5, 259679.5, 259743.5, 259807.5, 259871.5, 259935.5, 259999.5, 260063.5, 260127.5, 260191.5, 260255.5, 260319.5, 260383.5, 260447.5, 260511.5, 260575.5, 260639.5, 260703.5, 260767.5, 260831.5, 260895.5, 260959.5, 261023.5, 261087.5, 261151.5, 261215.5, 261279.5, 261343.5, 261407.5, 261471.5, 261535.5, 261599.5, 261663.5, 261727.5, 261791.5, 261855.5, 261919.5, 261983.5, 262047.5, 262111.5]