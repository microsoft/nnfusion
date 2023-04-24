
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
  float conv_local[2];
  __shared__ float data_pad_shared[3456];
  __shared__ float kernel_pad_shared[54];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[1];
  conv_local[(0)] = 0.000000e+00f;
  conv_local[(1)] = 0.000000e+00f;
  data_pad_shared[(((int)threadIdx.x))] = (((0 < (((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5))) && (0 < (((int)threadIdx.x) & 31))) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) - 33))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 128))] = ((0 < (((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5))) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) - 32))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 256))] = (((0 < (((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5))) && ((((int)threadIdx.x) & 31) < 31)) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) - 31))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 384))] = ((0 < (((int)threadIdx.x) & 31)) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) - 1))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 512))] = data[((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)))];
  data_pad_shared[((((int)threadIdx.x) + 640))] = (((((int)threadIdx.x) & 31) < 31) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 1))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 768))] = ((((((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5)) < 31) && (0 < (((int)threadIdx.x) & 31))) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 31))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 896))] = (((((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5)) < 31) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 32))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 1024))] = ((((((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5)) < 31) && ((((int)threadIdx.x) & 31) < 31)) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 33))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 1152))] = (((0 < (((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5))) && (0 < (((int)threadIdx.x) & 31))) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 991))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 1280))] = ((0 < (((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5))) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 992))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 1408))] = (((0 < (((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5))) && ((((int)threadIdx.x) & 31) < 31)) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 993))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 1536))] = ((0 < (((int)threadIdx.x) & 31)) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 1023))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 1664))] = data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 1024))];
  data_pad_shared[((((int)threadIdx.x) + 1792))] = (((((int)threadIdx.x) & 31) < 31) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 1025))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 1920))] = ((((((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5)) < 31) && (0 < (((int)threadIdx.x) & 31))) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 1055))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 2048))] = (((((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5)) < 31) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 1056))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 2176))] = ((((((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5)) < 31) && ((((int)threadIdx.x) & 31) < 31)) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 1057))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 2304))] = (((0 < (((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5))) && (0 < (((int)threadIdx.x) & 31))) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 2015))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 2432))] = ((0 < (((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5))) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 2016))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 2560))] = (((0 < (((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5))) && ((((int)threadIdx.x) & 31) < 31)) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 2017))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 2688))] = ((0 < (((int)threadIdx.x) & 31)) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 2047))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 2816))] = data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 2048))];
  data_pad_shared[((((int)threadIdx.x) + 2944))] = (((((int)threadIdx.x) & 31) < 31) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 2049))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 3072))] = ((((((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5)) < 31) && (0 < (((int)threadIdx.x) & 31))) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 2079))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 3200))] = (((((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5)) < 31) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 2080))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 3328))] = ((((((((int)blockIdx.x) & 7) * 4) + (((int)threadIdx.x) >> 5)) < 31) && ((((int)threadIdx.x) & 31) < 31)) ? data[(((((((int)blockIdx.x) & 7) * 128) + ((int)threadIdx.x)) + 2081))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 54) {
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((int)blockIdx.x) >> 3) * 54) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 27; ++ra_fused0_inner_outer) {
    data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 128) + (((int)threadIdx.x) & 63)))];
    data_pad_shared_local[(1)] = data_pad_shared[((((ra_fused0_inner_outer * 128) + (((int)threadIdx.x) & 63)) + 64))];
    kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 6) * 27) + ra_fused0_inner_outer))];
    conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
    conv_local[(1)] = (conv_local[(1)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(0)]));
  }
  conv_unpad[((((((((int)blockIdx.x) >> 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + (((int)threadIdx.x) & 63)))] = max((conv_local[(0)] + bias[((((((int)blockIdx.x) >> 3) * 2) + (((int)threadIdx.x) >> 6)))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) >> 3) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + (((int)threadIdx.x) & 63)) + 64))] = max((conv_local[(1)] + bias[((((((int)blockIdx.x) >> 3) * 2) + (((int)threadIdx.x) >> 6)))]), 0.000000e+00f);
}

dim3 grid(64, 1, 1);
dim3 block(128, 1, 1);
best_idx 5