
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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv_unpad) {
  float conv_local[4];
  __shared__ float data_pad_shared[72];
  __shared__ float kernel_pad_shared[576];
  float data_pad_shared_local[1];
  float kernel_pad_shared_local[4];
  conv_local[0] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  conv_local[2] = 0.000000e+00f;
  conv_local[3] = 0.000000e+00f;
  if (((int)threadIdx.x) < 72) {
    data_pad_shared[((int)threadIdx.x)] = (((((((((((int)blockIdx.x) % 25) * 2) + ((((int)threadIdx.x) & 7) >> 2)) < 49) && (0 < ((((int)threadIdx.x) / 24) + (((((((int)blockIdx.x) % 25) * 4) + ((((int)threadIdx.x) & 7) >> 1)) % 98) / 7)))) && (((((int)threadIdx.x) / 24) + (((((((int)blockIdx.x) % 25) * 4) + ((((int)threadIdx.x) & 7) >> 1)) % 98) / 7)) < 15)) && (0 < (((((int)threadIdx.x) % 24) >> 3) + ((((((int)blockIdx.x) % 25) * 8) + (((int)threadIdx.x) & 7)) % 14)))) && ((((((int)threadIdx.x) % 24) >> 3) + ((((((int)blockIdx.x) % 25) * 8) + (((int)threadIdx.x) & 7)) % 14)) < 15)) ? data[((((((((int)threadIdx.x) / 24) * 14) + ((((int)blockIdx.x) % 25) * 8)) + ((((int)threadIdx.x) % 24) >> 3)) + (((int)threadIdx.x) & 7)) - 15)] : 0.000000e+00f);
  }
  kernel_pad_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) / 25) * 576) + ((int)threadIdx.x))];
  kernel_pad_shared[(((int)threadIdx.x) + 128)] = kernel[((((((int)blockIdx.x) / 25) * 576) + (((((int)threadIdx.x) + 128) / 9) * 9)) + ((((int)threadIdx.x) + 2) % 9))];
  kernel_pad_shared[(((int)threadIdx.x) + 256)] = kernel[((((((int)blockIdx.x) / 25) * 576) + (((((int)threadIdx.x) + 256) / 9) * 9)) + ((((int)threadIdx.x) + 4) % 9))];
  kernel_pad_shared[(((int)threadIdx.x) + 384)] = kernel[((((((int)blockIdx.x) / 25) * 576) + (((((int)threadIdx.x) + 384) / 9) * 9)) + ((((int)threadIdx.x) + 6) % 9))];
  if (((int)threadIdx.x) < 64) {
    kernel_pad_shared[(((int)threadIdx.x) + 512)] = kernel[((((((int)blockIdx.x) / 25) * 576) + (((((int)threadIdx.x) + 512) / 9) * 9)) + ((((int)threadIdx.x) + 8) % 9))];
  }
  __syncthreads();
  for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 9; ++ra_fused0_inner_outer) {
    data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 8) + (((int)threadIdx.x) & 7))];
    kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 3) * 9) + ra_fused0_inner_outer)];
    kernel_pad_shared_local[1] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 9) + ra_fused0_inner_outer) + 144)];
    kernel_pad_shared_local[2] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 9) + ra_fused0_inner_outer) + 288)];
    kernel_pad_shared_local[3] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 9) + ra_fused0_inner_outer) + 432)];
    conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
    conv_local[1] = (conv_local[1] + (data_pad_shared_local[0] * kernel_pad_shared_local[1]));
    conv_local[2] = (conv_local[2] + (data_pad_shared_local[0] * kernel_pad_shared_local[2]));
    conv_local[3] = (conv_local[3] + (data_pad_shared_local[0] * kernel_pad_shared_local[3]));
  }
  if ((((((int)blockIdx.x) % 25) * 2) + ((((int)threadIdx.x) & 7) >> 2)) < 49) {
    conv_unpad[(((((((int)blockIdx.x) / 25) * 12544) + ((((int)threadIdx.x) >> 3) * 196)) + ((((int)blockIdx.x) % 25) * 8)) + (((int)threadIdx.x) & 7))] = conv_local[0];
    conv_unpad[((((((((int)blockIdx.x) / 25) * 12544) + ((((int)threadIdx.x) >> 3) * 196)) + ((((int)blockIdx.x) % 25) * 8)) + (((int)threadIdx.x) & 7)) + 3136)] = conv_local[1];
    conv_unpad[((((((((int)blockIdx.x) / 25) * 12544) + ((((int)threadIdx.x) >> 3) * 196)) + ((((int)blockIdx.x) % 25) * 8)) + (((int)threadIdx.x) & 7)) + 6272)] = conv_local[2];
    conv_unpad[((((((((int)blockIdx.x) / 25) * 12544) + ((((int)threadIdx.x) >> 3) * 196)) + ((((int)blockIdx.x) % 25) * 8)) + (((int)threadIdx.x) & 7)) + 9408)] = conv_local[3];
  }
}

dim3 grid(400, 1, 1);
dim3 block(128, 1, 1);
best_idx 1