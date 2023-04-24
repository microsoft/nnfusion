
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
  __shared__ float data_pad_shared[512];
  __shared__ float kernel_pad_shared[1024];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[1];
  conv_local[0] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 8; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = data[((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 196) * 16)) + (((int)threadIdx.x) & 15))];
    data_pad_shared[(((int)threadIdx.x) + 256)] = data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 196) * 16)) + (((int)threadIdx.x) & 15)) + 50176)];
    kernel_pad_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 196) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) / 196) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048)];
    kernel_pad_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) / 196) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
    kernel_pad_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((int)blockIdx.x) / 196) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144)];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 7))];
      data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 7)) + 8)];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 3) * 32) + ra_fused0_inner_outer)];
      conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
      conv_local[1] = (conv_local[1] + (data_pad_shared_local[1] * kernel_pad_shared_local[0]));
    }
  }
  conv_unpad[(((((((int)blockIdx.x) / 196) * 100352) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 196) * 16)) + (((int)threadIdx.x) & 7))] = max((conv_local[0] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 196) * 100352) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 196) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = max((conv_local[1] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
}

dim3 grid(392, 1, 1);
dim3 block(256, 1, 1);
best_idx 6