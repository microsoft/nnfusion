//21504_1_1_147_1_1
//128_84_83_83_7_2_SAME
//dim3 grid(21504, 1, 1);
//dim3 block(147, 1, 1);

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
extern "C" __global__ void __launch_bounds__(147) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[6];
  __shared__ float PaddedInput_shared[4183];
  __shared__ float kernel_shared[49];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  PaddedInput_shared[(((int)threadIdx.x))] = ((((141 <= ((int)threadIdx.x)) && (3 <= (((((int)blockIdx.x) & 1) * 42) + (((int)threadIdx.x) % 47)))) && ((((((int)blockIdx.x) & 1) * 42) + (((int)threadIdx.x) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + ((((int)threadIdx.x) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + (((int)threadIdx.x) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 147))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 6) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 6) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 147) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 6) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 294))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 12) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 12) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 294) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 12) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 441))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 18) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 18) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 441) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 18) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 588))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 24) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 24) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 588) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 24) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 735))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 30) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 30) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 735) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 30) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 882))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 36) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 36) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 882) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 36) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1029))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 42) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 42) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 1029) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 42) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1176))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 1) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 1) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 1176) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 1) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1323))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 7) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 7) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 1323) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 7) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1470))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 13) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 13) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 1470) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 13) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1617))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 19) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 19) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 1617) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 19) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1764))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 25) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 25) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 1764) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 25) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1911))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 31) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 31) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 1911) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 31) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 2058))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 37) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 37) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 2058) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 37) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 2205))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 43) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 43) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 2205) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 43) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 2352))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 2) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 2) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 2352) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 2) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 2499))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 8) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 8) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 2499) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 8) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 2646))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 14) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 14) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 2646) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 14) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 2793))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 20) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 20) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 2793) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 20) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 2940))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 26) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 26) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 2940) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 26) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 3087))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 32) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 32) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 3087) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 32) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 3234))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 38) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 38) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 3234) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 38) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 3381))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 44) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 44) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 3381) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 44) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 3528))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 3) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 3) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 3528) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 3) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 3675))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 9) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 9) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 3675) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 9) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 3822))] = (((3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 15) % 47))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 15) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 3822) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 15) % 47)) - 252))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 3969))] = ((((((int)threadIdx.x) < 73) && (3 <= (((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 21) % 47)))) && ((((((int)blockIdx.x) & 1) * 42) + ((((int)threadIdx.x) + 21) % 47)) < 86)) ? data[(((((((((int)blockIdx.x) >> 1) * 6889) + (((((int)threadIdx.x) + 3969) / 47) * 83)) + ((((int)blockIdx.x) & 1) * 42)) + ((((int)threadIdx.x) + 21) % 47)) - 252))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 67) {
    PaddedInput_shared[((((int)threadIdx.x) + 4116))] = 0.000000e+00f;
  }
  if (((int)threadIdx.x) < 49) {
    kernel_shared[(((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) % 168) >> 1) * 49) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int di_outer_inner = 0; di_outer_inner < 7; ++di_outer_inner) {
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)))] * kernel_shared[((di_outer_inner * 7))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 2))] * kernel_shared[((di_outer_inner * 7))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 4))] * kernel_shared[((di_outer_inner * 7))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 1))] * kernel_shared[(((di_outer_inner * 7) + 1))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 3))] * kernel_shared[(((di_outer_inner * 7) + 1))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 5))] * kernel_shared[(((di_outer_inner * 7) + 1))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 2))] * kernel_shared[(((di_outer_inner * 7) + 2))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 4))] * kernel_shared[(((di_outer_inner * 7) + 2))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 6))] * kernel_shared[(((di_outer_inner * 7) + 2))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 3))] * kernel_shared[(((di_outer_inner * 7) + 3))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 5))] * kernel_shared[(((di_outer_inner * 7) + 3))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 7))] * kernel_shared[(((di_outer_inner * 7) + 3))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 4))] * kernel_shared[(((di_outer_inner * 7) + 4))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 6))] * kernel_shared[(((di_outer_inner * 7) + 4))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 8))] * kernel_shared[(((di_outer_inner * 7) + 4))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 5))] * kernel_shared[(((di_outer_inner * 7) + 5))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 7))] * kernel_shared[(((di_outer_inner * 7) + 5))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 9))] * kernel_shared[(((di_outer_inner * 7) + 5))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 6))] * kernel_shared[(((di_outer_inner * 7) + 6))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 8))] * kernel_shared[(((di_outer_inner * 7) + 6))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 10))] * kernel_shared[(((di_outer_inner * 7) + 6))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 94))] * kernel_shared[((di_outer_inner * 7))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 96))] * kernel_shared[((di_outer_inner * 7))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 98))] * kernel_shared[((di_outer_inner * 7))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 95))] * kernel_shared[(((di_outer_inner * 7) + 1))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 97))] * kernel_shared[(((di_outer_inner * 7) + 1))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 99))] * kernel_shared[(((di_outer_inner * 7) + 1))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 96))] * kernel_shared[(((di_outer_inner * 7) + 2))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 98))] * kernel_shared[(((di_outer_inner * 7) + 2))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 100))] * kernel_shared[(((di_outer_inner * 7) + 2))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 97))] * kernel_shared[(((di_outer_inner * 7) + 3))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 99))] * kernel_shared[(((di_outer_inner * 7) + 3))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 101))] * kernel_shared[(((di_outer_inner * 7) + 3))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 98))] * kernel_shared[(((di_outer_inner * 7) + 4))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 100))] * kernel_shared[(((di_outer_inner * 7) + 4))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 102))] * kernel_shared[(((di_outer_inner * 7) + 4))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 99))] * kernel_shared[(((di_outer_inner * 7) + 5))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 101))] * kernel_shared[(((di_outer_inner * 7) + 5))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 103))] * kernel_shared[(((di_outer_inner * 7) + 5))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 100))] * kernel_shared[(((di_outer_inner * 7) + 6))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 102))] * kernel_shared[(((di_outer_inner * 7) + 6))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 7) * 188) + (di_outer_inner * 47)) + ((((int)threadIdx.x) % 7) * 6)) + 104))] * kernel_shared[(((di_outer_inner * 7) + 6))]));
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    for (int j_inner = 0; j_inner < 3; ++j_inner) {
      DepthwiseConv2d[((((((((((int)blockIdx.x) >> 1) * 1764) + ((((int)threadIdx.x) / 7) * 84)) + (i_inner * 42)) + ((((int)blockIdx.x) & 1) * 21)) + ((((int)threadIdx.x) % 7) * 3)) + j_inner))] = DepthwiseConv2d_local[(((i_inner * 3) + j_inner))];
    }
  }
}

