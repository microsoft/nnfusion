
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
  float conv_local[16];
  __shared__ float data_pad_shared[3072];
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
  for (int ra_fused0_outer = 0; ra_fused0_outer < 2; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = (((((((int)blockIdx.x) % 33) * 96) + (((int)threadIdx.x) % 96)) < 3136) ? data[(((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) / 96) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 96)))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 384))] = (((((((int)blockIdx.x) % 33) * 96) + (((int)threadIdx.x) % 96)) < 3136) ? data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) / 96) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 96)) + 12544))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 768))] = (((((((int)blockIdx.x) % 33) * 96) + (((int)threadIdx.x) % 96)) < 3136) ? data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) / 96) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 96)) + 25088))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1152))] = (((((((int)blockIdx.x) % 33) * 96) + (((int)threadIdx.x) % 96)) < 3136) ? data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) / 96) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 96)) + 37632))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1536))] = (((((((int)blockIdx.x) % 33) * 96) + (((int)threadIdx.x) % 96)) < 3136) ? data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) / 96) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 96)) + 50176))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1920))] = (((((((int)blockIdx.x) % 33) * 96) + (((int)threadIdx.x) % 96)) < 3136) ? data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) / 96) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 96)) + 62720))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 2304))] = (((((((int)blockIdx.x) % 33) * 96) + (((int)threadIdx.x) % 96)) < 3136) ? data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) / 96) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 96)) + 75264))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 2688))] = (((((((int)blockIdx.x) % 33) * 96) + (((int)threadIdx.x) % 96)) < 3136) ? data[((((((ra_fused0_outer * 100352) + ((((int)threadIdx.x) / 96) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 96)) + 87808))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 33) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)))];
    kernel_pad_shared[((((int)threadIdx.x) + 384))] = kernel[(((((((((int)blockIdx.x) / 33) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 768))];
    kernel_pad_shared[((((int)threadIdx.x) + 768))] = kernel[(((((((((int)blockIdx.x) / 33) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 1536))];
    kernel_pad_shared[((((int)threadIdx.x) + 1152))] = kernel[(((((((((int)blockIdx.x) / 33) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 2304))];
    kernel_pad_shared[((((int)threadIdx.x) + 1536))] = kernel[(((((((((int)blockIdx.x) / 33) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 3072))];
    if (((int)threadIdx.x) < 128) {
      kernel_pad_shared[((((int)threadIdx.x) + 1920))] = kernel[(((((((((int)blockIdx.x) / 33) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 3840))];
    }
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 96) + (((int)threadIdx.x) % 48)))];
      data_pad_shared_local[(1)] = data_pad_shared[((((ra_fused0_inner_outer * 96) + (((int)threadIdx.x) % 48)) + 48))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) / 48) * 32) + ra_fused0_inner_outer))];
      kernel_pad_shared_local[(1)] = kernel_pad_shared[(((((((int)threadIdx.x) / 48) * 32) + ra_fused0_inner_outer) + 256))];
      kernel_pad_shared_local[(2)] = kernel_pad_shared[(((((((int)threadIdx.x) / 48) * 32) + ra_fused0_inner_outer) + 512))];
      kernel_pad_shared_local[(3)] = kernel_pad_shared[(((((((int)threadIdx.x) / 48) * 32) + ra_fused0_inner_outer) + 768))];
      kernel_pad_shared_local[(4)] = kernel_pad_shared[(((((((int)threadIdx.x) / 48) * 32) + ra_fused0_inner_outer) + 1024))];
      kernel_pad_shared_local[(5)] = kernel_pad_shared[(((((((int)threadIdx.x) / 48) * 32) + ra_fused0_inner_outer) + 1280))];
      kernel_pad_shared_local[(6)] = kernel_pad_shared[(((((((int)threadIdx.x) / 48) * 32) + ra_fused0_inner_outer) + 1536))];
      kernel_pad_shared_local[(7)] = kernel_pad_shared[(((((((int)threadIdx.x) / 48) * 32) + ra_fused0_inner_outer) + 1792))];
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
  conv_unpad[((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)))] = conv_local[(0)];
  conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 25088))] = conv_local[(2)];
  conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 50176))] = conv_local[(4)];
  conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 75264))] = conv_local[(6)];
  conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 100352))] = conv_local[(8)];
  conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 125440))] = conv_local[(10)];
  conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 150528))] = conv_local[(12)];
  conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 175616))] = conv_local[(14)];
  if ((((((int)blockIdx.x) % 33) * 96) + (((int)threadIdx.x) % 48)) < 3088) {
    conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 48))] = conv_local[(1)];
    conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 25136))] = conv_local[(3)];
    conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 50224))] = conv_local[(5)];
    conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 75312))] = conv_local[(7)];
    conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 100400))] = conv_local[(9)];
    conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 125488))] = conv_local[(11)];
    conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 150576))] = conv_local[(13)];
    conv_unpad[(((((((((int)blockIdx.x) / 33) * 200704) + ((((int)threadIdx.x) / 48) * 3136)) + ((((int)blockIdx.x) % 33) * 96)) + (((int)threadIdx.x) % 48)) + 175664))] = conv_local[(15)];
  }
}

dim3 grid(132, 1, 1);
dim3 block(384, 1, 1);
best_idx 6