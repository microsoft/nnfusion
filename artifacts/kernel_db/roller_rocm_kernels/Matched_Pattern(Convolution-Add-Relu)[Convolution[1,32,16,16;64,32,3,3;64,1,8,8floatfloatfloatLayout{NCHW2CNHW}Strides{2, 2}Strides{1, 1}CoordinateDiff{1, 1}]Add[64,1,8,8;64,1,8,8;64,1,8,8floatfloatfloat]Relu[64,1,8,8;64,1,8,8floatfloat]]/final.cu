
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
  __shared__ float data_pad_shared[2048];
  __shared__ float kernel_pad_shared[8192];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[1];
  conv_local[0] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 3; ++ra_fused0_outer) {
    bool cse_var_1 = (ra_fused0_outer < 2);
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = (((0 < (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) & 15) >> 3) * 2)) + ((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) % 9) / 3))) && (0 < (((((int)threadIdx.x) & 7) * 2) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) % 3)))) ? data[((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) / 9) * 256) + (((int)blockIdx.x) * 64)) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + (((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) % 3)) - 17)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 512)] = (((cse_var_1 && (0 < (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) & 15) >> 3) * 2)) + (((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) + 5) % 9) / 3)))) && (0 < (((((int)threadIdx.x) & 7) * 2) + ((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) + 2) % 3)))) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 32) / 9) * 256) + (((int)blockIdx.x) * 64)) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + ((((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) + 5) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) + 2) % 3)) - 17)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 1024)] = (((cse_var_1 && (0 < (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) & 15) >> 3) * 2)) + (((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) + 1) % 9) / 3)))) && (0 < (((((int)threadIdx.x) & 7) * 2) + ((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) + 1) % 3)))) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 64) / 9) * 256) + (((int)blockIdx.x) * 64)) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + ((((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) + 1) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) + 1) % 3)) - 17)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 1536)] = (((cse_var_1 && (0 < (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) & 15) >> 3) * 2)) + (((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) / 3) + 2) % 3)))) && (0 < (((((int)threadIdx.x) & 7) * 2) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) % 3)))) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 96) / 9) * 256) + (((int)blockIdx.x) * 64)) + (((((int)threadIdx.x) & 15) >> 3) * 32)) + ((((((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) / 3) + 2) % 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 4)) % 3)) - 17)] : 0.000000e+00f);
    kernel_pad_shared[((int)threadIdx.x)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 512)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 1152)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 1024)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 2304)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 1536)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 3456)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 2048)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 4608)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 2560)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 5760)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 3072)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 6912)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 3584)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 8064)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 4096)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 9216)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 4608)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 10368)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 5120)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 11520)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 5632)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 12672)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 6144)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 13824)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 6656)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 14976)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 7168)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 16128)] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 7680)] = ((((ra_fused0_outer * 4) + ((((int)threadIdx.x) & 127) >> 5)) < 9) ? kernel[(((((((int)threadIdx.x) >> 7) * 288) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) & 127)) / 9) * 9)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) & 127)) % 9)) + 17280)] : 0.000000e+00f);
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 128; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 7))];
      data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 7)) + 8)];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 3) * 128) + ra_fused0_inner_outer)];
      if (((ra_fused0_outer * 4) + (ra_fused0_inner_outer >> 5)) < 9) {
        conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
        conv_local[1] = (conv_local[1] + (data_pad_shared_local[1] * kernel_pad_shared_local[0]));
      }
    }
  }
  conv_unpad[((((((int)threadIdx.x) >> 3) * 64) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 7))] = max((conv_local[0] + bias[(((int)threadIdx.x) >> 3)]), 0.000000e+00f);
  conv_unpad[(((((((int)threadIdx.x) >> 3) * 64) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = max((conv_local[1] + bias[(((int)threadIdx.x) >> 3)]), 0.000000e+00f);
}

dim3 grid(4, 1, 1);
dim3 block(512, 1, 1);
best_idx 11