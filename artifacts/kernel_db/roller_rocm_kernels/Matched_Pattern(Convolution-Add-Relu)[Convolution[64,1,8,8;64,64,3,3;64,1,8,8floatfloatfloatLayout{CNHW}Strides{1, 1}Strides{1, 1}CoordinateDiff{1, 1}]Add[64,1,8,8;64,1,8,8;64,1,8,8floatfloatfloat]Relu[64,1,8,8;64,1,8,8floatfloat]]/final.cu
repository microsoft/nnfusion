
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
extern "C" __global__ void __launch_bounds__(512) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ bias, float* __restrict__ conv_unpad) {
  float conv_local[2];
  __shared__ float data_pad_shared[4096];
  __shared__ float kernel_pad_shared[1024];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[1];
  conv_local[0] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 9; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = (((((0 < (((((int)threadIdx.x) & 63) >> 3) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 9) / 3))) && ((((((int)threadIdx.x) & 63) >> 3) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + (((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 3)))) && (((((int)threadIdx.x) & 7) + (((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 3)) < 9)) ? data[((((((((ra_fused0_outer * 64) + (((int)threadIdx.x) >> 6)) / 9) * 64) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 9) / 3) * 8)) + (((int)threadIdx.x) & 63)) + (((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 3)) - 9)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 512)] = (((((0 < (((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 8) % 9) / 3))) && ((((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 8) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 3)))) && (((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 3)) < 9)) ? data[(((((((((ra_fused0_outer * 64) + (((int)threadIdx.x) >> 6)) + 8) / 9) * 64) + ((((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 8) % 9) / 3) * 8)) + (((int)threadIdx.x) & 63)) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 3)) - 9)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 1024)] = (((((0 < (((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 7) % 9) / 3))) && ((((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 7) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 1) % 3)))) && (((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 1) % 3)) < 9)) ? data[(((((((((ra_fused0_outer * 64) + (((int)threadIdx.x) >> 6)) + 16) / 9) * 64) + ((((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 7) % 9) / 3) * 8)) + (((int)threadIdx.x) & 63)) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 1) % 3)) - 9)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 1536)] = (((((0 < (((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) / 3) + 2) % 3))) && ((((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) / 3) + 2) % 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + (((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 3)))) && (((((int)threadIdx.x) & 7) + (((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 3)) < 9)) ? data[(((((((((ra_fused0_outer * 64) + (((int)threadIdx.x) >> 6)) + 24) / 9) * 64) + ((((((((int)threadIdx.x) >> 6) + ra_fused0_outer) / 3) + 2) % 3) * 8)) + (((int)threadIdx.x) & 63)) + (((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 3)) - 9)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 2048)] = (((((0 < (((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 5) % 9) / 3))) && ((((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 5) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 3)))) && (((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 3)) < 9)) ? data[(((((((((ra_fused0_outer * 64) + (((int)threadIdx.x) >> 6)) + 32) / 9) * 64) + ((((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 5) % 9) / 3) * 8)) + (((int)threadIdx.x) & 63)) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 3)) - 9)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 2560)] = (((((0 < (((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 4) % 9) / 3))) && ((((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 4) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 1) % 3)))) && (((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 1) % 3)) < 9)) ? data[(((((((((ra_fused0_outer * 64) + (((int)threadIdx.x) >> 6)) + 40) / 9) * 64) + ((((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 4) % 9) / 3) * 8)) + (((int)threadIdx.x) & 63)) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 1) % 3)) - 9)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 3072)] = (((((0 < (((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) / 3) + 1) % 3))) && ((((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) / 3) + 1) % 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + (((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 3)))) && (((((int)threadIdx.x) & 7) + (((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 3)) < 9)) ? data[(((((((((ra_fused0_outer * 64) + (((int)threadIdx.x) >> 6)) + 48) / 9) * 64) + ((((((((int)threadIdx.x) >> 6) + ra_fused0_outer) / 3) + 1) % 3) * 8)) + (((int)threadIdx.x) & 63)) + (((((int)threadIdx.x) >> 6) + ra_fused0_outer) % 3)) - 9)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 3584)] = (((((0 < (((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 9) / 3))) && ((((((int)threadIdx.x) & 63) >> 3) + (((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 3)))) && (((((int)threadIdx.x) & 7) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 3)) < 9)) ? data[(((((((((ra_fused0_outer * 64) + (((int)threadIdx.x) >> 6)) + 56) / 9) * 64) + ((((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 9) / 3) * 8)) + (((int)threadIdx.x) & 63)) + ((((((int)threadIdx.x) >> 6) + ra_fused0_outer) + 2) % 3)) - 9)] : 0.000000e+00f);
    kernel_pad_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) >> 6) * 576)) + ((((ra_fused0_outer * 64) + (((int)threadIdx.x) & 63)) / 9) * 9)) + ((ra_fused0_outer + (((int)threadIdx.x) & 63)) % 9))];
    kernel_pad_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) >> 6) * 576)) + ((((ra_fused0_outer * 64) + (((int)threadIdx.x) & 63)) / 9) * 9)) + ((ra_fused0_outer + (((int)threadIdx.x) & 63)) % 9)) + 4608)];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 64; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 31))];
      data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 31)) + 32)];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 5) * 64) + ra_fused0_inner_outer)];
      conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
      conv_local[1] = (conv_local[1] + (data_pad_shared_local[1] * kernel_pad_shared_local[0]));
    }
  }
  conv_unpad[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))] = max((conv_local[0] + bias[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 5))]), 0.000000e+00f);
  conv_unpad[((((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = max((conv_local[1] + bias[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 5))]), 0.000000e+00f);
}

dim3 grid(4, 1, 1);
dim3 block(512, 1, 1);
best_idx 15