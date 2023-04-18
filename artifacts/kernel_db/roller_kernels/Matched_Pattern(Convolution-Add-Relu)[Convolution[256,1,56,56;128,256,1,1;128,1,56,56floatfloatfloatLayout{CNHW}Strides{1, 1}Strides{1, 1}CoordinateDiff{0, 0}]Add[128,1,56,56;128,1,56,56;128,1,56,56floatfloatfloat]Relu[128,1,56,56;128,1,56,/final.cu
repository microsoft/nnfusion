
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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ bias, float* __restrict__ conv_unpad) {
  float conv_local[16];
  __shared__ float data_pad_shared[1024];
  __shared__ float kernel_pad_shared[2048];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[8];
  conv_local[(0)] = 0.000000e+00f;
  conv_local[(2)] = 0.000000e+00f;
  conv_local[(4)] = 0.000000e+00f;
  conv_local[(6)] = 0.000000e+00f;
  conv_local[(8)] = 0.000000e+00f;
  conv_local[(10)] = 0.000000e+00f;
  conv_local[(12)] = 0.000000e+00f;
  conv_local[(14)] = 0.000000e+00f;
  conv_local[(1)] = 0.000000e+00f;
  conv_local[(3)] = 0.000000e+00f;
  conv_local[(5)] = 0.000000e+00f;
  conv_local[(7)] = 0.000000e+00f;
  conv_local[(9)] = 0.000000e+00f;
  conv_local[(11)] = 0.000000e+00f;
  conv_local[(13)] = 0.000000e+00f;
  conv_local[(15)] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 8; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 31)))];
    data_pad_shared[((((int)threadIdx.x) + 128))] = data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 31)) + 12544))];
    data_pad_shared[((((int)threadIdx.x) + 256))] = data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 31)) + 25088))];
    data_pad_shared[((((int)threadIdx.x) + 384))] = data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 31)) + 37632))];
    data_pad_shared[((((int)threadIdx.x) + 512))] = data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 31)) + 50176))];
    data_pad_shared[((((int)threadIdx.x) + 640))] = data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 31)) + 62720))];
    data_pad_shared[((((int)threadIdx.x) + 768))] = data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 31)) + 75264))];
    data_pad_shared[((((int)threadIdx.x) + 896))] = data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) >> 5) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 31)) + 87808))];
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)))];
    kernel_pad_shared[((((int)threadIdx.x) + 128))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 1024))];
    kernel_pad_shared[((((int)threadIdx.x) + 256))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048))];
    kernel_pad_shared[((((int)threadIdx.x) + 384))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 3072))];
    kernel_pad_shared[((((int)threadIdx.x) + 512))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096))];
    kernel_pad_shared[((((int)threadIdx.x) + 640))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 5120))];
    kernel_pad_shared[((((int)threadIdx.x) + 768))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144))];
    kernel_pad_shared[((((int)threadIdx.x) + 896))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168))];
    kernel_pad_shared[((((int)threadIdx.x) + 1024))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192))];
    kernel_pad_shared[((((int)threadIdx.x) + 1152))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 9216))];
    kernel_pad_shared[((((int)threadIdx.x) + 1280))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 10240))];
    kernel_pad_shared[((((int)threadIdx.x) + 1408))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 11264))];
    kernel_pad_shared[((((int)threadIdx.x) + 1536))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288))];
    kernel_pad_shared[((((int)threadIdx.x) + 1664))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 13312))];
    kernel_pad_shared[((((int)threadIdx.x) + 1792))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336))];
    kernel_pad_shared[((((int)threadIdx.x) + 1920))] = kernel[(((((((((int)blockIdx.x) / 98) * 16384) + ((((int)threadIdx.x) >> 5) * 256)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 15360))];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15)))];
      data_pad_shared_local[(1)] = data_pad_shared[((((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 16))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer))];
      kernel_pad_shared_local[(1)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 256))];
      kernel_pad_shared_local[(2)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 512))];
      kernel_pad_shared_local[(3)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 768))];
      kernel_pad_shared_local[(4)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 1024))];
      kernel_pad_shared_local[(5)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 1280))];
      kernel_pad_shared_local[(6)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 1536))];
      kernel_pad_shared_local[(7)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 1792))];
      conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
      conv_local[(2)] = (conv_local[(2)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(1)]));
      conv_local[(4)] = (conv_local[(4)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(2)]));
      conv_local[(6)] = (conv_local[(6)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(3)]));
      conv_local[(8)] = (conv_local[(8)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(4)]));
      conv_local[(10)] = (conv_local[(10)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(5)]));
      conv_local[(12)] = (conv_local[(12)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(6)]));
      conv_local[(14)] = (conv_local[(14)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(7)]));
      conv_local[(1)] = (conv_local[(1)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(0)]));
      conv_local[(3)] = (conv_local[(3)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(1)]));
      conv_local[(5)] = (conv_local[(5)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(2)]));
      conv_local[(7)] = (conv_local[(7)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(3)]));
      conv_local[(9)] = (conv_local[(9)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(4)]));
      conv_local[(11)] = (conv_local[(11)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(5)]));
      conv_local[(13)] = (conv_local[(13)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(6)]));
      conv_local[(15)] = (conv_local[(15)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(7)]));
    }
  }
  conv_unpad[((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)))] = max((conv_local[(0)] + bias[((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 25088))] = max((conv_local[(2)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 8))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 50176))] = max((conv_local[(4)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 16))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 75264))] = max((conv_local[(6)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 24))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 100352))] = max((conv_local[(8)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 32))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 125440))] = max((conv_local[(10)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 40))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 150528))] = max((conv_local[(12)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 48))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 175616))] = max((conv_local[(14)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 56))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 16))] = max((conv_local[(1)] + bias[((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 25104))] = max((conv_local[(3)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 8))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 50192))] = max((conv_local[(5)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 16))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 75280))] = max((conv_local[(7)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 24))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 100368))] = max((conv_local[(9)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 32))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 125456))] = max((conv_local[(11)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 40))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 150544))] = max((conv_local[(13)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 48))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 4) * 3136)) + ((((int)blockIdx.x) % 98) * 32)) + (((int)threadIdx.x) & 15)) + 175632))] = max((conv_local[(15)] + bias[(((((((int)blockIdx.x) / 98) * 64) + (((int)threadIdx.x) >> 4)) + 56))]), 0.000000e+00f);
}

dim3 grid(196, 1, 1);
dim3 block(128, 1, 1);
best_idx 5