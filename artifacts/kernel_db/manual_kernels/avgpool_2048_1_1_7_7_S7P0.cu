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
    int batch = blockIdx.x * 8 + (threadIdx.x >> 5);
    int in_batch = threadIdx.x & 31;
    int start_id = batch * 7 * 7;
    float s = input0[start_id + in_batch];
    s += in_batch + 32 < 49 ? input0[start_id + in_batch + 32] : 0;
    s += __shfl_xor_sync(0xffffffff, s, 16);
    s += __shfl_xor_sync(0xffffffff, s, 8);
    s += __shfl_xor_sync(0xffffffff, s, 4);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    s += __shfl_xor_sync(0xffffffff, s, 1);
    if (in_batch == 0) output0[batch] = s * 0.02040816326530612; // s / 7 / 7
}
// %%%

// +++
dim3 grid(256, 1, 1);
dim3 block(256, 1, 1);
// +++

int main() {
    float *input0, *output0;
    cudaMallocManaged(&input0, 2048 * 7 * 7 * sizeof(float));
    cudaMallocManaged(&output0, 2048 * sizeof(float));
    for (int i = 0; i < 2048 * 7 * 7; i++) input0[i] = i;
    default_function_kernel0<<<grid, block>>>(input0, output0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int i = 0; i < 2048; i++) {
        printf("%f ", output0[i]);
        if (i % 64 == 63) printf("\n");
    }
    return 0;
}

// python
// s = list(range(0, 49))
// lst = [sum([(49 * i + x) for x in s]) / 49 for i in range(2048)]
// >>> lst[:64]
// [18.375, 55.890625, 93.40625, 130.921875, 168.4375, 205.953125, 243.46875, 280.984375, 318.5, 356.015625, 393.53125, 431.046875, 468.5625, 506.078125, 543.59375, 581.109375, 618.625, 656.140625, 693.65625, 731.171875, 768.6875, 806.203125, 843.71875, 881.234375, 918.75, 956.265625, 993.78125, 1031.296875, 1068.8125, 1106.328125, 1143.84375, 1181.359375, 1218.875, 1256.390625, 1293.90625, 1331.421875, 1368.9375, 1406.453125, 1443.96875, 1481.484375, 1519.0, 1556.515625, 1594.03125, 1631.546875, 1669.0625, 1706.578125, 1744.09375, 1781.609375, 1819.125, 1856.640625, 1894.15625, 1931.671875, 1969.1875, 2006.703125, 2044.21875, 2081.734375, 2119.25, 2156.765625, 2194.28125, 2231.796875, 2269.3125, 2306.828125, 2344.34375, 2381.859375]
// >>> lst[-64:]
// [74449.375, 74486.890625, 74524.40625, 74561.921875, 74599.4375, 74636.953125, 74674.46875, 74711.984375, 74749.5, 74787.015625, 74824.53125, 74862.046875, 74899.5625, 74937.078125, 74974.59375, 75012.109375, 75049.625, 75087.140625, 75124.65625, 75162.171875, 75199.6875, 75237.203125, 75274.71875, 75312.234375, 75349.75, 75387.265625, 75424.78125, 75462.296875, 75499.8125, 75537.328125, 75574.84375, 75612.359375, 75649.875, 75687.390625, 75724.90625, 75762.421875, 75799.9375, 75837.453125, 75874.96875, 75912.484375, 75950.0, 75987.515625, 76025.03125, 76062.546875, 76100.0625, 76137.578125, 76175.09375, 76212.609375, 76250.125, 76287.640625, 76325.15625, 76362.671875, 76400.1875, 76437.703125, 76475.21875, 76512.734375, 76550.25, 76587.765625, 76625.28125, 76662.796875, 76700.3125, 76737.828125, 76775.34375, 76812.859375]