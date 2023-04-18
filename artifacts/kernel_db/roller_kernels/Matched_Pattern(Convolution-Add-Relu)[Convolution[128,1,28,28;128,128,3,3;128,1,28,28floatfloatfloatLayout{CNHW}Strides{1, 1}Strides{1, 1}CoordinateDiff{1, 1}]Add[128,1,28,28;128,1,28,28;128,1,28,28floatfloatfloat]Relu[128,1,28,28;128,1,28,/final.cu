
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
  float data_pad_shared_local[1];
  float kernel_pad_shared_local[2];
  conv_local[(0)] = 0.000000e+00f;
  conv_local[(1)] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 36; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = (((((0 < (((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) / 28) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) % 9) / 3))) && ((((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) / 28) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) % 9) / 3)) < 29)) && (0 < (((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) % 28) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) % 3)))) && ((((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) % 28) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) % 3)) < 29)) ? data[((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) / 9) * 784) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) % 9) / 3) * 28)) + ((((int)blockIdx.x) % 49) * 16)) + (((int)threadIdx.x) & 15)) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) % 3)) - 29))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 256))] = (((((0 < (((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) / 28) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) + 7) % 9) / 3))) && ((((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) / 28) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) + 7) % 9) / 3)) < 29)) && (0 < (((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) % 28) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) + 1) % 3)))) && ((((((((int)blockIdx.x) % 49) * 16) + (((int)threadIdx.x) & 15)) % 28) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) + 1) % 3)) < 29)) ? data[(((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) + 16) / 9) * 784) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) + 7) % 9) / 3) * 28)) + ((((int)blockIdx.x) % 49) * 16)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 4)) + 1) % 3)) - 29))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 49) * 36864) + ((((int)threadIdx.x) >> 5) * 1152)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)))];
    kernel_pad_shared[((((int)threadIdx.x) + 256))] = kernel[(((((((((int)blockIdx.x) / 49) * 36864) + ((((int)threadIdx.x) >> 5) * 1152)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 9216))];
    kernel_pad_shared[((((int)threadIdx.x) + 512))] = kernel[(((((((((int)blockIdx.x) / 49) * 36864) + ((((int)threadIdx.x) >> 5) * 1152)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 18432))];
    kernel_pad_shared[((((int)threadIdx.x) + 768))] = kernel[(((((((((int)blockIdx.x) / 49) * 36864) + ((((int)threadIdx.x) >> 5) * 1152)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 27648))];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 15)))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer))];
      kernel_pad_shared_local[(1)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 512))];
      conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
      conv_local[(1)] = (conv_local[(1)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(1)]));
    }
  }
  conv_unpad[((((((((int)blockIdx.x) / 49) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 49) * 16)) + (((int)threadIdx.x) & 15)))] = max((conv_local[(0)] + bias[((((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 49) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 49) * 16)) + (((int)threadIdx.x) & 15)) + 12544))] = max((conv_local[(1)] + bias[(((((((int)blockIdx.x) / 49) * 32) + (((int)threadIdx.x) >> 4)) + 16))]), 0.000000e+00f);
}

dim3 grid(196, 1, 1);
dim3 block(256, 1, 1);
best_idx 5