
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
  float conv_local[4];
  __shared__ float data_pad_shared[512];
  __shared__ float kernel_pad_shared[2048];
  float data_pad_shared_local[1];
  float kernel_pad_shared_local[4];
  conv_local[(0)] = 0.000000e+00f;
  conv_local[(1)] = 0.000000e+00f;
  conv_local[(2)] = 0.000000e+00f;
  conv_local[(3)] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 32; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) < 49) ? data[(((((ra_fused0_outer * 6272) + ((((int)threadIdx.x) >> 4) * 196)) + (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) / 7) * 28)) + (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) % 7) * 2)))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 256))] = (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) < 49) ? data[((((((ra_fused0_outer * 6272) + ((((int)threadIdx.x) >> 4) * 196)) + (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) / 7) * 28)) + (((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) % 7) * 2)) + 3136))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.x) >> 5) * 1024)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)))];
    kernel_pad_shared[((((int)threadIdx.x) + 256))] = kernel[(((((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.x) >> 5) * 1024)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192))];
    kernel_pad_shared[((((int)threadIdx.x) + 512))] = kernel[(((((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.x) >> 5) * 1024)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 16384))];
    kernel_pad_shared[((((int)threadIdx.x) + 768))] = kernel[(((((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.x) >> 5) * 1024)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 24576))];
    kernel_pad_shared[((((int)threadIdx.x) + 1024))] = kernel[(((((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.x) >> 5) * 1024)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 32768))];
    kernel_pad_shared[((((int)threadIdx.x) + 1280))] = kernel[(((((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.x) >> 5) * 1024)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 40960))];
    kernel_pad_shared[((((int)threadIdx.x) + 1536))] = kernel[(((((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.x) >> 5) * 1024)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 49152))];
    kernel_pad_shared[((((int)threadIdx.x) + 1792))] = kernel[(((((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.x) >> 5) * 1024)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 57344))];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 15)))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer))];
      kernel_pad_shared_local[(1)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 512))];
      kernel_pad_shared_local[(2)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 1024))];
      kernel_pad_shared_local[(3)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 1536))];
      conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
      conv_local[(1)] = (conv_local[(1)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(1)]));
      conv_local[(2)] = (conv_local[(2)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(2)]));
      conv_local[(3)] = (conv_local[(3)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(3)]));
    }
  }
  if ((((((int)blockIdx.x) & 3) * 16) + (((int)threadIdx.x) & 15)) < 49) {
    conv_unpad[((((((((int)blockIdx.x) >> 2) * 3136) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)))] = conv_local[(0)];
    conv_unpad[(((((((((int)blockIdx.x) >> 2) * 3136) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 784))] = conv_local[(1)];
    conv_unpad[(((((((((int)blockIdx.x) >> 2) * 3136) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 1568))] = conv_local[(2)];
    conv_unpad[(((((((((int)blockIdx.x) >> 2) * 3136) + ((((int)threadIdx.x) >> 4) * 49)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 2352))] = conv_local[(3)];
  }
}

dim3 grid(128, 1, 1);
dim3 block(256, 1, 1);
best_idx 14