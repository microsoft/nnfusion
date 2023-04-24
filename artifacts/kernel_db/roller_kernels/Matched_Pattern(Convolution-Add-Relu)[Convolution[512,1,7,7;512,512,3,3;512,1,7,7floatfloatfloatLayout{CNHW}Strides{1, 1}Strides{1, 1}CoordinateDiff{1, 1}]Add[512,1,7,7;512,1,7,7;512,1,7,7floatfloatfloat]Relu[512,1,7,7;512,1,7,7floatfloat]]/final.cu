
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
  float conv_local[1];
  __shared__ float data_pad_shared[256];
  __shared__ float kernel_pad_shared[1024];
  float data_pad_shared_local[1];
  float kernel_pad_shared_local[1];
  conv_local[(0)] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 144; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = (((((((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) < 49) && (0 < ((((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) % 49) / 7) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 3)) % 9) / 3)))) && (((((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) % 49) / 7) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 3)) % 9) / 3)) < 8)) && (0 < (((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) % 7) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 3)) % 3)))) && ((((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) % 7) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 3)) % 3)) < 8)) ? data[((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 3)) / 9) * 49) + ((((int)blockIdx.x) % 7) * 8)) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 3)) % 9) / 3) * 7)) + (((int)threadIdx.x) & 7)) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 3)) % 3)) - 8))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) >> 5) * 4608)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)))];
    kernel_pad_shared[((((int)threadIdx.x) + 256))] = kernel[(((((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) >> 5) * 4608)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 36864))];
    kernel_pad_shared[((((int)threadIdx.x) + 512))] = kernel[(((((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) >> 5) * 4608)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 73728))];
    kernel_pad_shared[((((int)threadIdx.x) + 768))] = kernel[(((((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) >> 5) * 4608)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 110592))];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 8) + (((int)threadIdx.x) & 7)))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 32) + ra_fused0_inner_outer))];
      conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
    }
  }
  if ((((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) & 7)) < 49) {
    conv_unpad[((((((((int)blockIdx.x) / 7) * 1568) + ((((int)threadIdx.x) >> 3) * 49)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)))] = max((conv_local[(0)] + bias[((((((int)blockIdx.x) / 7) * 32) + (((int)threadIdx.x) >> 3)))]), 0.000000e+00f);
  }
}

dim3 grid(112, 1, 1);
dim3 block(256, 1, 1);
best_idx 4