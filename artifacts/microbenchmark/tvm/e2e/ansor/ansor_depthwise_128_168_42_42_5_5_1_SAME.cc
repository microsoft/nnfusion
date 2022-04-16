//21504_1_1_84_1_1
//128_168_42_42_5_1_SAME
//dim3 grid(21504, 1, 1);
//dim3 block(84, 1, 1);

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
extern "C" __global__ void __launch_bounds__(84) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[21];
  __shared__ float PaddedInput_shared[2116];
  __shared__ float kernel_shared[25];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  DepthwiseConv2d_local[(9)] = 0.000000e+00f;
  DepthwiseConv2d_local[(12)] = 0.000000e+00f;
  DepthwiseConv2d_local[(15)] = 0.000000e+00f;
  DepthwiseConv2d_local[(18)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(7)] = 0.000000e+00f;
  DepthwiseConv2d_local[(10)] = 0.000000e+00f;
  DepthwiseConv2d_local[(13)] = 0.000000e+00f;
  DepthwiseConv2d_local[(16)] = 0.000000e+00f;
  DepthwiseConv2d_local[(19)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(8)] = 0.000000e+00f;
  DepthwiseConv2d_local[(11)] = 0.000000e+00f;
  DepthwiseConv2d_local[(14)] = 0.000000e+00f;
  DepthwiseConv2d_local[(17)] = 0.000000e+00f;
  DepthwiseConv2d_local[(20)] = 0.000000e+00f;
  PaddedInput_shared[((((int)threadIdx.x) * 3))] = ((((31 <= ((int)threadIdx.x)) && (2 <= ((((int)threadIdx.x) * 3) % 46))) && (((((int)threadIdx.x) * 3) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + (((((int)threadIdx.x) * 3) / 46) * 42)) + ((((int)threadIdx.x) * 3) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1))] = ((((31 <= ((int)threadIdx.x)) && (2 <= (((((int)threadIdx.x) * 3) + 1) % 46))) && ((((((int)threadIdx.x) * 3) + 1) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 1) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 2))] = ((((30 <= ((int)threadIdx.x)) && (2 <= (((((int)threadIdx.x) * 3) + 2) % 46))) && ((((((int)threadIdx.x) * 3) + 2) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 2) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 2) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 252))] = (((2 <= (((((int)threadIdx.x) * 3) + 22) % 46)) && ((((((int)threadIdx.x) * 3) + 22) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 252) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 22) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 253))] = (((2 <= (((((int)threadIdx.x) * 3) + 23) % 46)) && ((((((int)threadIdx.x) * 3) + 23) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 253) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 23) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 254))] = (((2 <= (((((int)threadIdx.x) * 3) + 24) % 46)) && ((((((int)threadIdx.x) * 3) + 24) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 254) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 24) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 504))] = (((2 <= (((((int)threadIdx.x) * 3) + 44) % 46)) && ((((((int)threadIdx.x) * 3) + 44) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 504) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 44) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 505))] = (((2 <= (((((int)threadIdx.x) * 3) + 45) % 46)) && ((((((int)threadIdx.x) * 3) + 45) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 505) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 45) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 506))] = (((2 <= ((((int)threadIdx.x) * 3) % 46)) && (((((int)threadIdx.x) * 3) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + (((((int)threadIdx.x) * 3) / 46) * 42)) + ((((int)threadIdx.x) * 3) % 46)) + 376))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 756))] = (((2 <= (((((int)threadIdx.x) * 3) + 20) % 46)) && ((((((int)threadIdx.x) * 3) + 20) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 756) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 20) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 757))] = (((2 <= (((((int)threadIdx.x) * 3) + 21) % 46)) && ((((((int)threadIdx.x) * 3) + 21) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 757) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 21) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 758))] = (((2 <= (((((int)threadIdx.x) * 3) + 22) % 46)) && ((((((int)threadIdx.x) * 3) + 22) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 758) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 22) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1008))] = (((2 <= (((((int)threadIdx.x) * 3) + 42) % 46)) && ((((((int)threadIdx.x) * 3) + 42) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1008) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 42) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1009))] = (((2 <= (((((int)threadIdx.x) * 3) + 43) % 46)) && ((((((int)threadIdx.x) * 3) + 43) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1009) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 43) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1010))] = (((2 <= (((((int)threadIdx.x) * 3) + 44) % 46)) && ((((((int)threadIdx.x) * 3) + 44) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1010) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 44) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1260))] = (((2 <= (((((int)threadIdx.x) * 3) + 18) % 46)) && ((((((int)threadIdx.x) * 3) + 18) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1260) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 18) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1261))] = (((2 <= (((((int)threadIdx.x) * 3) + 19) % 46)) && ((((((int)threadIdx.x) * 3) + 19) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1261) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 19) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1262))] = (((2 <= (((((int)threadIdx.x) * 3) + 20) % 46)) && ((((((int)threadIdx.x) * 3) + 20) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1262) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 20) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1512))] = (((2 <= (((((int)threadIdx.x) * 3) + 40) % 46)) && ((((((int)threadIdx.x) * 3) + 40) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1512) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 40) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1513))] = (((2 <= (((((int)threadIdx.x) * 3) + 41) % 46)) && ((((((int)threadIdx.x) * 3) + 41) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1513) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 41) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1514))] = (((2 <= (((((int)threadIdx.x) * 3) + 42) % 46)) && ((((((int)threadIdx.x) * 3) + 42) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1514) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 42) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1764))] = (((2 <= (((((int)threadIdx.x) * 3) + 16) % 46)) && ((((((int)threadIdx.x) * 3) + 16) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1764) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 16) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1765))] = (((2 <= (((((int)threadIdx.x) * 3) + 17) % 46)) && ((((((int)threadIdx.x) * 3) + 17) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1765) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 17) % 46)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 3) + 1766))] = (((2 <= (((((int)threadIdx.x) * 3) + 18) % 46)) && ((((((int)threadIdx.x) * 3) + 18) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 1766) / 46) * 42)) + (((((int)threadIdx.x) * 3) + 18) % 46)) - 86))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 34) {
    PaddedInput_shared[(((((int)threadIdx.x) * 3) + 2016))] = ((((((int)threadIdx.x) < 3) && (2 <= (((((int)threadIdx.x) * 3) + 38) % 46))) && ((((((int)threadIdx.x) * 3) + 38) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 2016) / 46) * 42)) + ((((int)threadIdx.x) * 3) + 38)) - 86))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 33) {
    PaddedInput_shared[(((((int)threadIdx.x) * 3) + 2017))] = ((((((int)threadIdx.x) < 3) && (2 <= (((((int)threadIdx.x) * 3) + 39) % 46))) && ((((((int)threadIdx.x) * 3) + 39) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 2017) / 46) * 42)) + ((((int)threadIdx.x) * 3) + 39)) - 86))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 33) {
    PaddedInput_shared[(((((int)threadIdx.x) * 3) + 2018))] = ((((((int)threadIdx.x) < 2) && (2 <= (((((int)threadIdx.x) * 3) + 40) % 46))) && ((((((int)threadIdx.x) * 3) + 40) % 46) < 44)) ? data[(((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 3) + 2018) / 46) * 42)) + ((((int)threadIdx.x) * 3) + 40)) - 86))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 25) {
    kernel_shared[(((int)threadIdx.x))] = kernel[((((((int)blockIdx.x) % 168) * 25) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int dj_outer_inner = 0; dj_outer_inner < 5; ++dj_outer_inner) {
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 6))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 12))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(9)] = (DepthwiseConv2d_local[(9)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 18))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(12)] = (DepthwiseConv2d_local[(12)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 24))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(15)] = (DepthwiseConv2d_local[(15)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 30))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(18)] = (DepthwiseConv2d_local[(18)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 36))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 46))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 52))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(7)] = (DepthwiseConv2d_local[(7)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 58))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(10)] = (DepthwiseConv2d_local[(10)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 64))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(13)] = (DepthwiseConv2d_local[(13)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 70))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(16)] = (DepthwiseConv2d_local[(16)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 76))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(19)] = (DepthwiseConv2d_local[(19)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 82))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 92))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 98))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(8)] = (DepthwiseConv2d_local[(8)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 104))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(11)] = (DepthwiseConv2d_local[(11)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 110))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(14)] = (DepthwiseConv2d_local[(14)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 116))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(17)] = (DepthwiseConv2d_local[(17)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 122))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(20)] = (DepthwiseConv2d_local[(20)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 128))] * kernel_shared[(dj_outer_inner)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 46))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 52))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 58))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(9)] = (DepthwiseConv2d_local[(9)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 64))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(12)] = (DepthwiseConv2d_local[(12)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 70))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(15)] = (DepthwiseConv2d_local[(15)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 76))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(18)] = (DepthwiseConv2d_local[(18)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 82))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 92))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 98))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(7)] = (DepthwiseConv2d_local[(7)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 104))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(10)] = (DepthwiseConv2d_local[(10)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 110))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(13)] = (DepthwiseConv2d_local[(13)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 116))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(16)] = (DepthwiseConv2d_local[(16)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 122))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(19)] = (DepthwiseConv2d_local[(19)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 128))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 138))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 144))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(8)] = (DepthwiseConv2d_local[(8)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 150))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(11)] = (DepthwiseConv2d_local[(11)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 156))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(14)] = (DepthwiseConv2d_local[(14)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 162))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(17)] = (DepthwiseConv2d_local[(17)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 168))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(20)] = (DepthwiseConv2d_local[(20)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 174))] * kernel_shared[((dj_outer_inner + 5))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 92))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 98))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 104))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(9)] = (DepthwiseConv2d_local[(9)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 110))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(12)] = (DepthwiseConv2d_local[(12)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 116))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(15)] = (DepthwiseConv2d_local[(15)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 122))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(18)] = (DepthwiseConv2d_local[(18)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 128))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 138))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 144))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(7)] = (DepthwiseConv2d_local[(7)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 150))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(10)] = (DepthwiseConv2d_local[(10)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 156))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(13)] = (DepthwiseConv2d_local[(13)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 162))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(16)] = (DepthwiseConv2d_local[(16)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 168))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(19)] = (DepthwiseConv2d_local[(19)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 174))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 184))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 190))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(8)] = (DepthwiseConv2d_local[(8)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 196))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(11)] = (DepthwiseConv2d_local[(11)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 202))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(14)] = (DepthwiseConv2d_local[(14)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 208))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(17)] = (DepthwiseConv2d_local[(17)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 214))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(20)] = (DepthwiseConv2d_local[(20)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 220))] * kernel_shared[((dj_outer_inner + 10))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 138))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 144))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 150))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(9)] = (DepthwiseConv2d_local[(9)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 156))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(12)] = (DepthwiseConv2d_local[(12)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 162))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(15)] = (DepthwiseConv2d_local[(15)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 168))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(18)] = (DepthwiseConv2d_local[(18)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 174))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 184))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 190))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(7)] = (DepthwiseConv2d_local[(7)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 196))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(10)] = (DepthwiseConv2d_local[(10)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 202))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(13)] = (DepthwiseConv2d_local[(13)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 208))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(16)] = (DepthwiseConv2d_local[(16)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 214))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(19)] = (DepthwiseConv2d_local[(19)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 220))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 230))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 236))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(8)] = (DepthwiseConv2d_local[(8)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 242))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(11)] = (DepthwiseConv2d_local[(11)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 248))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(14)] = (DepthwiseConv2d_local[(14)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 254))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(17)] = (DepthwiseConv2d_local[(17)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 260))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(20)] = (DepthwiseConv2d_local[(20)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 266))] * kernel_shared[((dj_outer_inner + 15))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 184))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 190))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 196))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(9)] = (DepthwiseConv2d_local[(9)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 202))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(12)] = (DepthwiseConv2d_local[(12)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 208))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(15)] = (DepthwiseConv2d_local[(15)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 214))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(18)] = (DepthwiseConv2d_local[(18)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 220))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 230))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 236))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(7)] = (DepthwiseConv2d_local[(7)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 242))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(10)] = (DepthwiseConv2d_local[(10)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 248))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(13)] = (DepthwiseConv2d_local[(13)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 254))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(16)] = (DepthwiseConv2d_local[(16)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 260))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(19)] = (DepthwiseConv2d_local[(19)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 266))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 276))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 282))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(8)] = (DepthwiseConv2d_local[(8)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 288))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(11)] = (DepthwiseConv2d_local[(11)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 294))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(14)] = (DepthwiseConv2d_local[(14)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 300))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(17)] = (DepthwiseConv2d_local[(17)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 306))] * kernel_shared[((dj_outer_inner + 20))]));
    DepthwiseConv2d_local[(20)] = (DepthwiseConv2d_local[(20)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 6) * 138) + dj_outer_inner) + (((int)threadIdx.x) % 6)) + 312))] * kernel_shared[((dj_outer_inner + 20))]));
  }
  for (int i_inner = 0; i_inner < 3; ++i_inner) {
    DepthwiseConv2d[(((((((int)blockIdx.x) * 1764) + ((((int)threadIdx.x) / 6) * 126)) + (i_inner * 42)) + (((int)threadIdx.x) % 6)))] = DepthwiseConv2d_local[(i_inner)];
    DepthwiseConv2d[((((((((int)blockIdx.x) * 1764) + ((((int)threadIdx.x) / 6) * 126)) + (i_inner * 42)) + (((int)threadIdx.x) % 6)) + 6))] = DepthwiseConv2d_local[((i_inner + 3))];
    DepthwiseConv2d[((((((((int)blockIdx.x) * 1764) + ((((int)threadIdx.x) / 6) * 126)) + (i_inner * 42)) + (((int)threadIdx.x) % 6)) + 12))] = DepthwiseConv2d_local[((i_inner + 6))];
    DepthwiseConv2d[((((((((int)blockIdx.x) * 1764) + ((((int)threadIdx.x) / 6) * 126)) + (i_inner * 42)) + (((int)threadIdx.x) % 6)) + 18))] = DepthwiseConv2d_local[((i_inner + 9))];
    DepthwiseConv2d[((((((((int)blockIdx.x) * 1764) + ((((int)threadIdx.x) / 6) * 126)) + (i_inner * 42)) + (((int)threadIdx.x) % 6)) + 24))] = DepthwiseConv2d_local[((i_inner + 12))];
    DepthwiseConv2d[((((((((int)blockIdx.x) * 1764) + ((((int)threadIdx.x) / 6) * 126)) + (i_inner * 42)) + (((int)threadIdx.x) % 6)) + 30))] = DepthwiseConv2d_local[((i_inner + 15))];
    DepthwiseConv2d[((((((((int)blockIdx.x) * 1764) + ((((int)threadIdx.x) / 6) * 126)) + (i_inner * 42)) + (((int)threadIdx.x) % 6)) + 36))] = DepthwiseConv2d_local[((i_inner + 18))];
  }
}

