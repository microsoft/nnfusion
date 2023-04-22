
#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ bias, float* __restrict__ conv_unpad) {
  float conv_local[1];
  __shared__ float data_pad_shared[128];
  __shared__ float kernel_pad_shared[1024];
  float data_pad_shared_local[1];
  float kernel_pad_shared_local[1];
  conv_local[0] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 72; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = (((0 < ((((((((int)blockIdx.x) % 49) * 2) + ((((int)threadIdx.x) & 3) >> 1)) / 7) * 2) + ((((ra_fused0_outer * 5) + (((int)threadIdx.x) >> 2)) % 9) / 3))) && (0 < ((((((((int)blockIdx.x) % 49) * 4) + (((int)threadIdx.x) & 3)) % 14) * 2) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 2)) % 3)))) ? data[(((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 2)) / 9) * 784) + (((((((int)blockIdx.x) % 49) * 2) + ((((int)threadIdx.x) & 3) >> 1)) / 7) * 56)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) >> 2)) % 9) / 3) * 28)) + (((((((int)blockIdx.x) % 49) * 4) + (((int)threadIdx.x) & 3)) % 14) * 2)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 2)) % 3)) - 29)] : 0.000000e+00f);
    kernel_pad_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) / 49) * 73728) + ((((int)threadIdx.x) >> 5) * 2304)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3))];
    kernel_pad_shared[(((int)threadIdx.x) + 128)] = kernel[(((((((((int)blockIdx.x) / 49) * 73728) + ((((int)threadIdx.x) >> 5) * 2304)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 9216)];
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((((int)blockIdx.x) / 49) * 73728) + ((((int)threadIdx.x) >> 5) * 2304)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 18432)];
    kernel_pad_shared[(((int)threadIdx.x) + 384)] = kernel[(((((((((int)blockIdx.x) / 49) * 73728) + ((((int)threadIdx.x) >> 5) * 2304)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 27648)];
    kernel_pad_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((((int)blockIdx.x) / 49) * 73728) + ((((int)threadIdx.x) >> 5) * 2304)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 36864)];
    kernel_pad_shared[(((int)threadIdx.x) + 640)] = kernel[(((((((((int)blockIdx.x) / 49) * 73728) + ((((int)threadIdx.x) >> 5) * 2304)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 46080)];
    kernel_pad_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((((int)blockIdx.x) / 49) * 73728) + ((((int)threadIdx.x) >> 5) * 2304)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 55296)];
    kernel_pad_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((((int)blockIdx.x) / 49) * 73728) + ((((int)threadIdx.x) >> 5) * 2304)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 64512)];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 4) + (((int)threadIdx.x) & 3))];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 2) * 32) + ra_fused0_inner_outer)];
      conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
    }
  }
  conv_unpad[(((((((int)blockIdx.x) / 49) * 6272) + ((((int)threadIdx.x) >> 2) * 196)) + ((((int)blockIdx.x) % 49) * 4)) + (((int)threadIdx.x) & 3))] = max((conv_local[0] + bias[(((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 2))]), 0.000000e+00f);
}

dim3 grid(392, 1, 1);
dim3 block(128, 1, 1);
best_idx 1