
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv_unpad) {
  float conv_local[8];
  __shared__ float data_pad_shared[256];
  __shared__ float kernel_pad_shared[512];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[4];
  conv_local[(0)] = 0.000000e+00f;
  conv_local[(2)] = 0.000000e+00f;
  conv_local[(4)] = 0.000000e+00f;
  conv_local[(6)] = 0.000000e+00f;
  conv_local[(1)] = 0.000000e+00f;
  conv_local[(3)] = 0.000000e+00f;
  conv_local[(5)] = 0.000000e+00f;
  conv_local[(7)] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 64; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = (((((((int)blockIdx.x) % 7) * 32) + (((int)threadIdx.x) & 31)) < 196) ? data[(((((ra_fused0_outer * 6272) + ((((int)threadIdx.x) >> 5) * 784)) + (((((((int)blockIdx.x) % 7) * 32) + (((int)threadIdx.x) & 31)) / 14) * 56)) + (((((((int)blockIdx.x) % 7) * 32) + (((int)threadIdx.x) & 31)) % 14) * 2)))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + (ra_fused0_outer * 8)) + (((int)threadIdx.x) & 7)))];
    kernel_pad_shared[((((int)threadIdx.x) + 256))] = kernel[(((((((((int)blockIdx.x) / 7) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + (ra_fused0_outer * 8)) + (((int)threadIdx.x) & 7)) + 16384))];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 8; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15)))];
      data_pad_shared_local[(1)] = data_pad_shared[((((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 16))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer))];
      kernel_pad_shared_local[(1)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 128))];
      kernel_pad_shared_local[(2)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 256))];
      kernel_pad_shared_local[(3)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 384))];
      conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
      conv_local[(2)] = (conv_local[(2)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(1)]));
      conv_local[(4)] = (conv_local[(4)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(2)]));
      conv_local[(6)] = (conv_local[(6)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(3)]));
      conv_local[(1)] = (conv_local[(1)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(0)]));
      conv_local[(3)] = (conv_local[(3)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(1)]));
      conv_local[(5)] = (conv_local[(5)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(2)]));
      conv_local[(7)] = (conv_local[(7)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(3)]));
    }
  }
  if ((((((int)blockIdx.x) % 7) * 32) + (((int)threadIdx.x) & 15)) < 196) {
    conv_unpad[((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) >> 4) * 196)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 15)))] = conv_local[(0)];
    conv_unpad[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) >> 4) * 196)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 15)) + 3136))] = conv_local[(2)];
    conv_unpad[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) >> 4) * 196)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 15)) + 6272))] = conv_local[(4)];
    conv_unpad[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) >> 4) * 196)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 15)) + 9408))] = conv_local[(6)];
  }
  if ((((((int)blockIdx.x) % 7) * 32) + (((int)threadIdx.x) & 15)) < 180) {
    conv_unpad[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) >> 4) * 196)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 15)) + 16))] = conv_local[(1)];
    conv_unpad[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) >> 4) * 196)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 15)) + 3152))] = conv_local[(3)];
    conv_unpad[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) >> 4) * 196)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 15)) + 6288))] = conv_local[(5)];
    conv_unpad[(((((((((int)blockIdx.x) / 7) * 12544) + ((((int)threadIdx.x) >> 4) * 196)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 15)) + 9424))] = conv_local[(7)];
  }
}

dim3 grid(112, 1, 1);
dim3 block(256, 1, 1);
best_idx 11