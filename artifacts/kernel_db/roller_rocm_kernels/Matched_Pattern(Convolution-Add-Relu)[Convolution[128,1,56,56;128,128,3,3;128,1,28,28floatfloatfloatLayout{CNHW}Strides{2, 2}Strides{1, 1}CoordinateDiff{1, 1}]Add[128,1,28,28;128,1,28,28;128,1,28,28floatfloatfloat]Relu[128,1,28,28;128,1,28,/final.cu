
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ bias, float* __restrict__ conv_unpad) {
  float conv_local[2];
  __shared__ float data_pad_shared[256];
  __shared__ float kernel_pad_shared[512];
  float data_pad_shared_local[1];
  float kernel_pad_shared_local[2];
  conv_local[0] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 72; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = (((0 < ((((((((int)blockIdx.x) % 49) * 4) + ((((int)threadIdx.x) & 15) >> 2)) / 7) * 2) + ((((ra_fused0_outer * 7) + (((int)threadIdx.x) >> 4)) % 9) / 3))) && (0 < ((((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) % 28) * 2) + (((((int)threadIdx.x) >> 4) + ra_fused0_outer) % 3)))) ? data[(((((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 4)) / 9) * 3136) + (((((((int)blockIdx.x) % 49) * 4) + ((((int)threadIdx.x) & 15) >> 2)) / 7) * 112)) + (((((ra_fused0_outer * 7) + (((int)threadIdx.x) >> 4)) % 9) / 3) * 56)) + (((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) % 28) * 2)) + (((((int)threadIdx.x) >> 4) + ra_fused0_outer) % 3)) - 57)] : 0.000000e+00f);
    kernel_pad_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) / 49) * 36864) + ((((int)threadIdx.x) >> 4) * 1152)) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) & 15)) / 9) * 9)) + (((((ra_fused0_outer * 7) + (((int)threadIdx.x) & 15)) % 9) / 3) * 3)) + ((ra_fused0_outer + (((int)threadIdx.x) & 15)) % 3))];
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((((int)blockIdx.x) / 49) * 36864) + ((((int)threadIdx.x) >> 4) * 1152)) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) & 15)) / 9) * 9)) + (((((ra_fused0_outer * 7) + (((int)threadIdx.x) & 15)) % 9) / 3) * 3)) + ((ra_fused0_outer + (((int)threadIdx.x) & 15)) % 3)) + 18432)];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 16; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 15))];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 4) * 16) + ra_fused0_inner_outer)];
      kernel_pad_shared_local[1] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 16) + ra_fused0_inner_outer) + 256)];
      conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
      conv_local[1] = (conv_local[1] + (data_pad_shared_local[0] * kernel_pad_shared_local[1]));
    }
  }
  conv_unpad[(((((((int)blockIdx.x) / 49) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 49) * 16)) + (((int)threadIdx.x) & 15))] = max((conv_local[0] + bias[(((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 4))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 49) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 49) * 16)) + (((int)threadIdx.x) & 15)) + 12544)] = max((conv_local[1] + bias[((((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 4)) + 16)]), 0.000000e+00f);
}

dim3 grid(196, 1, 1);
dim3 block(256, 1, 1);
best_idx 0