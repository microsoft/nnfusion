
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
  __shared__ float data_pad_shared[256];
  __shared__ float kernel_pad_shared[512];
  float data_pad_shared_local[1];
  float kernel_pad_shared_local[1];
  conv_local[0] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 144; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = (((((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) < 49) && (0 < (((((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) % 49) / 7) * 2) + ((((ra_fused0_outer * 5) + (((int)threadIdx.x) >> 3)) % 9) / 3)))) && (0 < (((((((int)threadIdx.x) & 7) + (((int)blockIdx.x) % 7)) % 7) * 2) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 3)) % 3)))) ? data[(((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 3)) / 9) * 196) + (((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) / 7) * 28)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) >> 3)) % 9) / 3) * 14)) + ((((((int)threadIdx.x) & 7) + (((int)blockIdx.x) % 7)) % 7) * 2)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 3)) % 3)) - 15)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 128)] = (((((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) < 49) && (0 < (((((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) % 49) / 7) * 2) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) >> 3)) + 7) % 9) / 3)))) && (0 < (((((((int)threadIdx.x) & 7) + (((int)blockIdx.x) % 7)) % 7) * 2) + ((((ra_fused0_outer * 5) + (((int)threadIdx.x) >> 3)) + 1) % 3)))) ? data[((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 3)) + 16) / 9) * 196) + (((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) / 7) * 28)) + ((((((ra_fused0_outer * 5) + (((int)threadIdx.x) >> 3)) + 7) % 9) / 3) * 14)) + ((((((int)threadIdx.x) & 7) + (((int)blockIdx.x) % 7)) % 7) * 2)) + ((((ra_fused0_outer * 5) + (((int)threadIdx.x) >> 3)) + 1) % 3)) - 15)] : 0.000000e+00f);
    kernel_pad_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) >> 5) * 4608)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3))];
    kernel_pad_shared[(((int)threadIdx.x) + 128)] = kernel[(((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) >> 5) * 4608)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 18432)];
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) >> 5) * 4608)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 36864)];
    kernel_pad_shared[(((int)threadIdx.x) + 384)] = kernel[(((((((((int)blockIdx.x) / 7) * 73728) + ((((int)threadIdx.x) >> 5) * 4608)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) & 31)) / 9) * 9)) + (((((ra_fused0_outer * 5) + (((int)threadIdx.x) & 31)) % 9) / 3) * 3)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 31)) % 3)) + 55296)];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 8) + (((int)threadIdx.x) & 7))];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 3) * 32) + ra_fused0_inner_outer)];
      conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
    }
  }
  if ((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) < 49) {
    conv_unpad[(((((((int)blockIdx.x) / 7) * 784) + ((((int)threadIdx.x) >> 3) * 49)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = max((conv_local[0] + bias[(((((int)blockIdx.x) / 7) * 16) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
  }
}

dim3 grid(224, 1, 1);
dim3 block(128, 1, 1);
best_idx 3