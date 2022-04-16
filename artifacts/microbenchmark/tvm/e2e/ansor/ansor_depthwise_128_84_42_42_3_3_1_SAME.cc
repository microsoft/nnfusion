//37632_1_1_84_1_1
//128_84_42_42_3_1_SAME
//dim3 grid(37632, 1, 1);
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
  float DepthwiseConv2d_local[6];
  __shared__ float PaddedInput_shared[704];
  __shared__ float kernel_shared[18];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  PaddedInput_shared[((((int)threadIdx.x) * 2))] = (((1 <= (((((int)blockIdx.x) % 7) * 6) + (((int)threadIdx.x) / 22))) && (1 <= (((int)threadIdx.x) % 22))) ? data[(((((((((int)blockIdx.x) / 7) * 3528) + ((((int)blockIdx.x) % 7) * 252)) + ((((int)threadIdx.x) / 22) * 42)) + ((((int)threadIdx.x) % 22) * 2)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1))] = ((((1 <= (((((int)blockIdx.x) % 7) * 6) + (((((int)threadIdx.x) * 2) + 1) / 44))) && (1 <= (((((int)threadIdx.x) * 2) + 1) % 44))) && ((((((int)threadIdx.x) * 2) + 1) % 44) < 43)) ? data[(((((((((int)blockIdx.x) / 7) * 3528) + ((((int)blockIdx.x) % 7) * 252)) + ((((((int)threadIdx.x) * 2) + 1) / 44) * 42)) + (((((int)threadIdx.x) * 2) + 1) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 168))] = (((((((((int)blockIdx.x) % 7) * 6) + (((((int)threadIdx.x) * 2) + 168) / 44)) < 43) && (1 <= (((((int)threadIdx.x) * 2) + 36) % 44))) && ((((((int)threadIdx.x) * 2) + 36) % 44) < 43)) ? data[(((((((((int)blockIdx.x) / 7) * 3528) + ((((int)blockIdx.x) % 7) * 252)) + ((((((int)threadIdx.x) * 2) + 168) / 44) * 42)) + (((((int)threadIdx.x) * 2) + 36) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 169))] = (((((((((int)blockIdx.x) % 7) * 6) + (((((int)threadIdx.x) * 2) + 169) / 44)) < 43) && (1 <= (((((int)threadIdx.x) * 2) + 37) % 44))) && ((((((int)threadIdx.x) * 2) + 37) % 44) < 43)) ? data[(((((((((int)blockIdx.x) / 7) * 3528) + ((((int)blockIdx.x) % 7) * 252)) + ((((((int)threadIdx.x) * 2) + 169) / 44) * 42)) + (((((int)threadIdx.x) * 2) + 37) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 336))] = (((((1 <= (((((int)blockIdx.x) % 7) * 6) + ((((((int)threadIdx.x) * 2) + 336) % 352) / 44))) && ((((((int)blockIdx.x) % 7) * 6) + ((((((int)threadIdx.x) * 2) + 336) % 352) / 44)) < 43)) && (1 <= (((((int)threadIdx.x) * 2) + 28) % 44))) && ((((((int)threadIdx.x) * 2) + 28) % 44) < 43)) ? data[((((((((((int)blockIdx.x) / 7) * 3528) + ((((((int)threadIdx.x) * 2) + 336) / 352) * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + (((((((int)threadIdx.x) * 2) + 336) % 352) / 44) * 42)) + (((((int)threadIdx.x) * 2) + 28) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 337))] = (((((1 <= (((((int)blockIdx.x) % 7) * 6) + ((((((int)threadIdx.x) * 2) + 337) % 352) / 44))) && ((((((int)blockIdx.x) % 7) * 6) + ((((((int)threadIdx.x) * 2) + 337) % 352) / 44)) < 43)) && (1 <= (((((int)threadIdx.x) * 2) + 29) % 44))) && ((((((int)threadIdx.x) * 2) + 29) % 44) < 43)) ? data[((((((((((int)blockIdx.x) / 7) * 3528) + ((((((int)threadIdx.x) * 2) + 337) / 352) * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + (((((((int)threadIdx.x) * 2) + 337) % 352) / 44) * 42)) + (((((int)threadIdx.x) * 2) + 29) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 504))] = (((((((((int)blockIdx.x) % 7) * 6) + (((((int)threadIdx.x) * 2) + 152) / 44)) < 43) && (1 <= (((((int)threadIdx.x) * 2) + 20) % 44))) && ((((((int)threadIdx.x) * 2) + 20) % 44) < 43)) ? data[((((((((((int)blockIdx.x) / 7) * 3528) + ((((((int)threadIdx.x) * 2) + 504) / 352) * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + ((((((int)threadIdx.x) * 2) + 152) / 44) * 42)) + (((((int)threadIdx.x) * 2) + 20) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 505))] = (((((((((int)blockIdx.x) % 7) * 6) + (((((int)threadIdx.x) * 2) + 153) / 44)) < 43) && (1 <= (((((int)threadIdx.x) * 2) + 21) % 44))) && ((((((int)threadIdx.x) * 2) + 21) % 44) < 43)) ? data[((((((((((int)blockIdx.x) / 7) * 3528) + ((((((int)threadIdx.x) * 2) + 505) / 352) * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + ((((((int)threadIdx.x) * 2) + 153) / 44) * 42)) + (((((int)threadIdx.x) * 2) + 21) % 44)) - 43))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 16) {
    PaddedInput_shared[(((((int)threadIdx.x) * 2) + 672))] = (((((((int)blockIdx.x) % 7) * 6) + (((((int)threadIdx.x) * 2) + 320) / 44)) < 43) ? data[((((((((((int)blockIdx.x) / 7) * 3528) + ((((((int)threadIdx.x) * 2) + 672) / 352) * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + ((((((int)threadIdx.x) * 2) + 320) / 44) * 42)) + ((((int)threadIdx.x) * 2) + 12)) - 43))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 16) {
    PaddedInput_shared[(((((int)threadIdx.x) * 2) + 673))] = ((((((((int)blockIdx.x) % 7) * 6) + (((((int)threadIdx.x) * 2) + 321) / 44)) < 43) && (((int)threadIdx.x) < 15)) ? data[((((((((((int)blockIdx.x) / 7) * 3528) + ((((((int)threadIdx.x) * 2) + 673) / 352) * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + ((((((int)threadIdx.x) * 2) + 321) / 44) * 42)) + ((((int)threadIdx.x) * 2) + 13)) - 43))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 18) {
    kernel_shared[(((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) % 294) / 7) * 18) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int dj_outer_inner = 0; dj_outer_inner < 3; ++dj_outer_inner) {
    for (int di_inner = 0; di_inner < 3; ++di_inner) {
      DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 42) * 352) + (((((int)threadIdx.x) % 42) / 14) * 88)) + (di_inner * 44)) + dj_outer_inner) + (((int)threadIdx.x) % 14)))] * kernel_shared[(((((((int)threadIdx.x) / 42) * 9) + (di_inner * 3)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 42) * 352) + (((((int)threadIdx.x) % 42) / 14) * 88)) + (di_inner * 44)) + dj_outer_inner) + (((int)threadIdx.x) % 14)) + 14))] * kernel_shared[(((((((int)threadIdx.x) / 42) * 9) + (di_inner * 3)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 42) * 352) + (((((int)threadIdx.x) % 42) / 14) * 88)) + (di_inner * 44)) + dj_outer_inner) + (((int)threadIdx.x) % 14)) + 28))] * kernel_shared[(((((((int)threadIdx.x) / 42) * 9) + (di_inner * 3)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 42) * 352) + (((((int)threadIdx.x) % 42) / 14) * 88)) + (di_inner * 44)) + dj_outer_inner) + (((int)threadIdx.x) % 14)) + 44))] * kernel_shared[(((((((int)threadIdx.x) / 42) * 9) + (di_inner * 3)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 42) * 352) + (((((int)threadIdx.x) % 42) / 14) * 88)) + (di_inner * 44)) + dj_outer_inner) + (((int)threadIdx.x) % 14)) + 58))] * kernel_shared[(((((((int)threadIdx.x) / 42) * 9) + (di_inner * 3)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 42) * 352) + (((((int)threadIdx.x) % 42) / 14) * 88)) + (di_inner * 44)) + dj_outer_inner) + (((int)threadIdx.x) % 14)) + 72))] * kernel_shared[(((((((int)threadIdx.x) / 42) * 9) + (di_inner * 3)) + dj_outer_inner))]));
    }
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    DepthwiseConv2d[((((((((((int)blockIdx.x) / 7) * 3528) + ((((int)threadIdx.x) / 42) * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + (((((int)threadIdx.x) % 42) / 14) * 84)) + (i_inner * 42)) + (((int)threadIdx.x) % 14)))] = DepthwiseConv2d_local[(i_inner)];
    DepthwiseConv2d[(((((((((((int)blockIdx.x) / 7) * 3528) + ((((int)threadIdx.x) / 42) * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + (((((int)threadIdx.x) % 42) / 14) * 84)) + (i_inner * 42)) + (((int)threadIdx.x) % 14)) + 14))] = DepthwiseConv2d_local[((i_inner + 2))];
    DepthwiseConv2d[(((((((((((int)blockIdx.x) / 7) * 3528) + ((((int)threadIdx.x) / 42) * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + (((((int)threadIdx.x) % 42) / 14) * 84)) + (i_inner * 42)) + (((int)threadIdx.x) % 14)) + 28))] = DepthwiseConv2d_local[((i_inner + 4))];
  }
}

