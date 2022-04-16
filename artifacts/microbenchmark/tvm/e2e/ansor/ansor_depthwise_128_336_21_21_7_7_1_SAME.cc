//21504_1_1_126_1_1
//128_336_21_21_7_1_SAME
//dim3 grid(21504, 1, 1);
//dim3 block(126, 1, 1);

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
extern "C" __global__ void __launch_bounds__(126) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[7];
  __shared__ float PaddedInput_shared[1458];
  __shared__ float kernel_shared[98];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  PaddedInput_shared[(((int)threadIdx.x))] = ((((81 <= ((int)threadIdx.x)) && (3 <= (((int)threadIdx.x) % 27))) && ((((int)threadIdx.x) % 27) < 24)) ? data[(((((((int)blockIdx.x) * 882) + ((((int)threadIdx.x) / 27) * 21)) + (((int)threadIdx.x) % 27)) - 66))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 126))] = (((3 <= ((((int)threadIdx.x) + 18) % 27)) && (((((int)threadIdx.x) + 18) % 27) < 24)) ? data[(((((((int)blockIdx.x) * 882) + (((((int)threadIdx.x) + 126) / 27) * 21)) + ((((int)threadIdx.x) + 18) % 27)) - 66))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 252))] = (((3 <= ((((int)threadIdx.x) + 9) % 27)) && (((((int)threadIdx.x) + 9) % 27) < 24)) ? data[(((((((int)blockIdx.x) * 882) + (((((int)threadIdx.x) + 252) / 27) * 21)) + ((((int)threadIdx.x) + 9) % 27)) - 66))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 378))] = (((3 <= (((int)threadIdx.x) % 27)) && ((((int)threadIdx.x) % 27) < 24)) ? data[(((((((int)blockIdx.x) * 882) + ((((int)threadIdx.x) / 27) * 21)) + (((int)threadIdx.x) % 27)) + 228))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 504))] = (((3 <= ((((int)threadIdx.x) + 18) % 27)) && (((((int)threadIdx.x) + 18) % 27) < 24)) ? data[(((((((int)blockIdx.x) * 882) + (((((int)threadIdx.x) + 504) / 27) * 21)) + ((((int)threadIdx.x) + 18) % 27)) - 66))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 630))] = (((((81 <= ((((int)threadIdx.x) + 630) % 729)) && (((((int)threadIdx.x) + 630) % 729) < 648)) && (3 <= ((((int)threadIdx.x) + 9) % 27))) && (((((int)threadIdx.x) + 9) % 27) < 24)) ? data[((((((((int)blockIdx.x) * 882) + (((((int)threadIdx.x) + 630) / 729) * 441)) + ((((((int)threadIdx.x) + 630) % 729) / 27) * 21)) + ((((int)threadIdx.x) + 9) % 27)) - 66))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 756))] = ((((54 <= ((int)threadIdx.x)) && (3 <= (((int)threadIdx.x) % 27))) && ((((int)threadIdx.x) % 27) < 24)) ? data[((((((((int)blockIdx.x) * 882) + (((((int)threadIdx.x) + 756) / 729) * 441)) + (((((int)threadIdx.x) + 27) / 27) * 21)) + (((int)threadIdx.x) % 27)) - 66))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 882))] = (((3 <= ((((int)threadIdx.x) + 18) % 27)) && (((((int)threadIdx.x) + 18) % 27) < 24)) ? data[((((((((int)blockIdx.x) * 882) + (((((int)threadIdx.x) + 882) / 729) * 441)) + (((((int)threadIdx.x) + 153) / 27) * 21)) + ((((int)threadIdx.x) + 18) % 27)) - 66))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1008))] = (((3 <= ((((int)threadIdx.x) + 9) % 27)) && (((((int)threadIdx.x) + 9) % 27) < 24)) ? data[((((((((int)blockIdx.x) * 882) + (((((int)threadIdx.x) + 1008) / 729) * 441)) + (((((int)threadIdx.x) + 279) / 27) * 21)) + ((((int)threadIdx.x) + 9) % 27)) - 66))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1134))] = (((3 <= (((int)threadIdx.x) % 27)) && ((((int)threadIdx.x) % 27) < 24)) ? data[((((((((int)blockIdx.x) * 882) + (((((int)threadIdx.x) + 1134) / 729) * 441)) + (((((int)threadIdx.x) + 405) / 27) * 21)) + (((int)threadIdx.x) % 27)) - 66))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1260))] = ((((((int)threadIdx.x) < 117) && (3 <= ((((int)threadIdx.x) + 18) % 27))) && (((((int)threadIdx.x) + 18) % 27) < 24)) ? data[((((((((int)blockIdx.x) * 882) + (((((int)threadIdx.x) + 1260) / 729) * 441)) + (((((int)threadIdx.x) + 531) / 27) * 21)) + ((((int)threadIdx.x) + 18) % 27)) - 66))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 72) {
    PaddedInput_shared[((((int)threadIdx.x) + 1386))] = 0.000000e+00f;
  }
  if (((int)threadIdx.x) < 98) {
    kernel_shared[(((int)threadIdx.x))] = kernel[((((((int)blockIdx.x) % 168) * 98) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int dj_outer_inner = 0; dj_outer_inner < 7; ++dj_outer_inner) {
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)))] * kernel_shared[((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 27))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 7))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 54))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 14))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 81))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 21))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 108))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 28))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 135))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 35))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 162))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 42))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 27))] * kernel_shared[((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 54))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 7))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 81))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 14))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 108))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 21))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 135))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 28))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 162))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 35))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 189))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 42))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 54))] * kernel_shared[((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 81))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 7))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 108))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 14))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 135))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 21))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 162))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 28))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 189))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 35))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 216))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 42))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 81))] * kernel_shared[((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 108))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 7))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 135))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 14))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 162))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 21))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 189))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 28))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 216))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 35))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 243))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 42))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 108))] * kernel_shared[((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 135))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 7))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 162))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 14))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 189))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 21))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 216))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 28))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 243))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 35))]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 270))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 42))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 135))] * kernel_shared[((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 162))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 7))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 189))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 14))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 216))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 21))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 243))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 28))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 270))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 35))]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 297))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 42))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 162))] * kernel_shared[((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 189))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 7))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 216))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 14))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 243))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 21))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 270))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 28))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 297))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 35))]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 63) * 729) + (((((int)threadIdx.x) % 63) / 21) * 189)) + dj_outer_inner) + (((int)threadIdx.x) % 21)) + 324))] * kernel_shared[(((((((int)threadIdx.x) / 63) * 49) + dj_outer_inner) + 42))]));
  }
  for (int i_inner = 0; i_inner < 7; ++i_inner) {
    DepthwiseConv2d[(((((((int)blockIdx.x) * 882) + ((((int)threadIdx.x) / 21) * 147)) + (i_inner * 21)) + (((int)threadIdx.x) % 21)))] = DepthwiseConv2d_local[(i_inner)];
  }
}

