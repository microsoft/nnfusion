
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
  float conv_local[8];
  __shared__ float data_pad_shared[2048];
  __shared__ float kernel_pad_shared[1024];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[4];
  conv_local[0] = 0.000000e+00f;
  conv_local[2] = 0.000000e+00f;
  conv_local[4] = 0.000000e+00f;
  conv_local[6] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  conv_local[3] = 0.000000e+00f;
  conv_local[5] = 0.000000e+00f;
  conv_local[7] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 8; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = data[((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 63))];
    data_pad_shared[(((int)threadIdx.x) + 256)] = data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 63)) + 12544)];
    data_pad_shared[(((int)threadIdx.x) + 512)] = data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 63)) + 25088)];
    data_pad_shared[(((int)threadIdx.x) + 768)] = data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 63)) + 37632)];
    data_pad_shared[(((int)threadIdx.x) + 1024)] = data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 63)) + 50176)];
    data_pad_shared[(((int)threadIdx.x) + 1280)] = data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 63)) + 62720)];
    data_pad_shared[(((int)threadIdx.x) + 1536)] = data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 63)) + 75264)];
    data_pad_shared[(((int)threadIdx.x) + 1792)] = data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 6) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 63)) + 87808)];
    kernel_pad_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048)];
    kernel_pad_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
    kernel_pad_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((int)blockIdx.x) / 49) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144)];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 31))];
      data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 31)) + 32)];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 5) * 32) + ra_fused0_inner_outer)];
      kernel_pad_shared_local[1] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 32) + ra_fused0_inner_outer) + 256)];
      kernel_pad_shared_local[2] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 32) + ra_fused0_inner_outer) + 512)];
      kernel_pad_shared_local[3] = kernel_pad_shared[((((((int)threadIdx.x) >> 5) * 32) + ra_fused0_inner_outer) + 768)];
      conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
      conv_local[2] = (conv_local[2] + (data_pad_shared_local[0] * kernel_pad_shared_local[1]));
      conv_local[4] = (conv_local[4] + (data_pad_shared_local[0] * kernel_pad_shared_local[2]));
      conv_local[6] = (conv_local[6] + (data_pad_shared_local[0] * kernel_pad_shared_local[3]));
      conv_local[1] = (conv_local[1] + (data_pad_shared_local[1] * kernel_pad_shared_local[0]));
      conv_local[3] = (conv_local[3] + (data_pad_shared_local[1] * kernel_pad_shared_local[1]));
      conv_local[5] = (conv_local[5] + (data_pad_shared_local[1] * kernel_pad_shared_local[2]));
      conv_local[7] = (conv_local[7] + (data_pad_shared_local[1] * kernel_pad_shared_local[3]));
    }
  }
  conv_unpad[(((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 31))] = max((conv_local[0] + bias[(((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 5))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 31)) + 25088)] = max((conv_local[2] + bias[((((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 5)) + 8)]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 31)) + 50176)] = max((conv_local[4] + bias[((((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 5)) + 16)]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 31)) + 75264)] = max((conv_local[6] + bias[((((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 5)) + 24)]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = max((conv_local[1] + bias[(((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 5))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 31)) + 25120)] = max((conv_local[3] + bias[((((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 5)) + 8)]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 31)) + 50208)] = max((conv_local[5] + bias[((((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 5)) + 16)]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 49) * 64)) + (((int)threadIdx.x) & 31)) + 75296)] = max((conv_local[7] + bias[((((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 5)) + 24)]), 0.000000e+00f);
}

dim3 grid(196, 1, 1);
dim3 block(256, 1, 1);
best_idx 4