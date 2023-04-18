
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
  __shared__ float data_pad_shared[512];
  __shared__ float kernel_pad_shared[1024];
  float data_pad_shared_local[1];
  float kernel_pad_shared_local[4];
  conv_local[(0)] = 0.000000e+00f;
  conv_local[(1)] = 0.000000e+00f;
  conv_local[(2)] = 0.000000e+00f;
  conv_local[(3)] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 16; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) < 49) ? data[(((((ra_fused0_outer * 1568) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 128))] = (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) < 49) ? data[((((((ra_fused0_outer * 1568) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 392))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 256))] = (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) < 49) ? data[((((((ra_fused0_outer * 1568) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 784))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 384))] = (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) < 49) ? data[((((((ra_fused0_outer * 1568) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 1176))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)))];
    kernel_pad_shared[((((int)threadIdx.x) + 128))] = kernel[(((((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048))];
    kernel_pad_shared[((((int)threadIdx.x) + 256))] = kernel[(((((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096))];
    kernel_pad_shared[((((int)threadIdx.x) + 384))] = kernel[(((((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144))];
    kernel_pad_shared[((((int)threadIdx.x) + 512))] = kernel[(((((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192))];
    kernel_pad_shared[((((int)threadIdx.x) + 640))] = kernel[(((((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 10240))];
    kernel_pad_shared[((((int)threadIdx.x) + 768))] = kernel[(((((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288))];
    kernel_pad_shared[((((int)threadIdx.x) + 896))] = kernel[(((((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336))];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 15)))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer))];
      kernel_pad_shared_local[(1)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 256))];
      kernel_pad_shared_local[(2)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 512))];
      kernel_pad_shared_local[(3)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 768))];
      conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
      conv_local[(1)] = (conv_local[(1)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(1)]));
      conv_local[(2)] = (conv_local[(2)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(2)]));
      conv_local[(3)] = (conv_local[(3)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(3)]));
    }
  }
  if ((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) < 49) {
    conv_unpad[((((((((int)blockIdx.x) >> 2) * 1568) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)))] = conv_local[(0)];
    conv_unpad[(((((((((int)blockIdx.x) >> 2) * 1568) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 392))] = conv_local[(1)];
    conv_unpad[(((((((((int)blockIdx.x) >> 2) * 1568) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 784))] = conv_local[(2)];
    conv_unpad[(((((((((int)blockIdx.x) >> 2) * 1568) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 1176))] = conv_local[(3)];
  }
}

dim3 grid(256, 1, 1);
dim3 block(128, 1, 1);
best_idx 15