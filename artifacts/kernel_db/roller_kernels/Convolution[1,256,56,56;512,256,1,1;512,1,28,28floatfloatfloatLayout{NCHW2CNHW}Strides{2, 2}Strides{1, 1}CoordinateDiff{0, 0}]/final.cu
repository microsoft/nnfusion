
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
extern "C" __global__ void __launch_bounds__(384) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv_unpad) {
  float conv_local[8];
  __shared__ float data_pad_shared[1024];
  __shared__ float kernel_pad_shared[3072];
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
  for (int ra_fused0_outer = 0; ra_fused0_outer < 8; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = (((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 31)) < 784) ? data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + (((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 31)) / 28) * 112)) + (((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 31)) % 28) * 2)))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 384))] = (((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 31)) < 784) ? data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + (((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 31)) / 28) * 112)) + (((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 31)) % 28) * 2)) + 37632))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 256) {
      data_pad_shared[((((int)threadIdx.x) + 768))] = (((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 31)) < 784) ? data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + (((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 31)) / 28) * 112)) + (((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 31)) % 28) * 2)) + 75264))] : 0.000000e+00f);
    }
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 25) * 24576) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)))];
    kernel_pad_shared[((((int)threadIdx.x) + 384))] = kernel[(((((((((int)blockIdx.x) / 25) * 24576) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 3072))];
    kernel_pad_shared[((((int)threadIdx.x) + 768))] = (((((((int)blockIdx.x) / 25) * 96) + (((int)threadIdx.x) >> 5)) < 488) ? kernel[(((((((((int)blockIdx.x) / 25) * 24576) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1152))] = (((((((int)blockIdx.x) / 25) * 96) + (((int)threadIdx.x) >> 5)) < 476) ? kernel[(((((((((int)blockIdx.x) / 25) * 24576) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 9216))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1536))] = (((((((int)blockIdx.x) / 25) * 96) + (((int)threadIdx.x) >> 5)) < 464) ? kernel[(((((((((int)blockIdx.x) / 25) * 24576) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1920))] = (((((((int)blockIdx.x) / 25) * 96) + (((int)threadIdx.x) >> 5)) < 452) ? kernel[(((((((((int)blockIdx.x) / 25) * 24576) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 15360))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 2304))] = (((((((int)blockIdx.x) / 25) * 96) + (((int)threadIdx.x) >> 5)) < 440) ? kernel[(((((((((int)blockIdx.x) / 25) * 24576) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 18432))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 2688))] = (((((((int)blockIdx.x) / 25) * 96) + (((int)threadIdx.x) >> 5)) < 428) ? kernel[(((((((((int)blockIdx.x) / 25) * 24576) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 21504))] : 0.000000e+00f);
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15)))];
      data_pad_shared_local[(1)] = data_pad_shared[((((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 16))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer))];
      kernel_pad_shared_local[(1)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 768))];
      kernel_pad_shared_local[(2)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 1536))];
      kernel_pad_shared_local[(3)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 2304))];
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
  conv_unpad[((((((((int)blockIdx.x) / 25) * 75264) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 25) * 32)) + (((int)threadIdx.x) & 15)))] = conv_local[(0)];
  if ((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 15)) < 768) {
    conv_unpad[(((((((((int)blockIdx.x) / 25) * 75264) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 25) * 32)) + (((int)threadIdx.x) & 15)) + 16))] = conv_local[(1)];
  }
  if ((((((int)blockIdx.x) / 25) * 96) + (((int)threadIdx.x) >> 4)) < 488) {
    conv_unpad[(((((((((int)blockIdx.x) / 25) * 75264) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 25) * 32)) + (((int)threadIdx.x) & 15)) + 18816))] = conv_local[(2)];
    if ((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 15)) < 768) {
      conv_unpad[(((((((((int)blockIdx.x) / 25) * 75264) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 25) * 32)) + (((int)threadIdx.x) & 15)) + 18832))] = conv_local[(3)];
    }
  }
  if ((((((int)blockIdx.x) / 25) * 96) + (((int)threadIdx.x) >> 4)) < 464) {
    conv_unpad[(((((((((int)blockIdx.x) / 25) * 75264) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 25) * 32)) + (((int)threadIdx.x) & 15)) + 37632))] = conv_local[(4)];
    if ((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 15)) < 768) {
      conv_unpad[(((((((((int)blockIdx.x) / 25) * 75264) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 25) * 32)) + (((int)threadIdx.x) & 15)) + 37648))] = conv_local[(5)];
    }
  }
  if ((((((int)blockIdx.x) / 25) * 96) + (((int)threadIdx.x) >> 4)) < 440) {
    conv_unpad[(((((((((int)blockIdx.x) / 25) * 75264) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 25) * 32)) + (((int)threadIdx.x) & 15)) + 56448))] = conv_local[(6)];
    if ((((((int)blockIdx.x) % 25) * 32) + (((int)threadIdx.x) & 15)) < 768) {
      conv_unpad[(((((((((int)blockIdx.x) / 25) * 75264) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 25) * 32)) + (((int)threadIdx.x) & 15)) + 56464))] = conv_local[(7)];
    }
  }
}

dim3 grid(150, 1, 1);
dim3 block(384, 1, 1);
best_idx 8