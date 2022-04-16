//16128_1_1_196_1_1
//128_84_42_42_5_1_SAME
//dim3 grid(16128, 1, 1);
//dim3 block(196, 1, 1);

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
extern "C" __global__ void __launch_bounds__(196) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[6];
  __shared__ float PaddedInput_shared[1656];
  __shared__ float kernel_shared[50];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  PaddedInput_shared[(((int)threadIdx.x))] = ((((36 <= ((int)threadIdx.x)) && (2 <= (((((int)blockIdx.x) % 3) * 14) + (((int)threadIdx.x) % 18)))) && ((((((int)blockIdx.x) % 3) * 14) + (((int)threadIdx.x) % 18)) < 44)) ? data[(((((((((int)blockIdx.x) / 3) * 3528) + ((((int)threadIdx.x) / 18) * 42)) + ((((int)blockIdx.x) % 3) * 14)) + (((int)threadIdx.x) % 18)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 196))] = (((2 <= (((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 16) % 18))) && ((((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 16) % 18)) < 44)) ? data[(((((((((int)blockIdx.x) / 3) * 3528) + (((((int)threadIdx.x) + 196) / 18) * 42)) + ((((int)blockIdx.x) % 3) * 14)) + ((((int)threadIdx.x) + 16) % 18)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 392))] = (((2 <= (((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 14) % 18))) && ((((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 14) % 18)) < 44)) ? data[(((((((((int)blockIdx.x) / 3) * 3528) + (((((int)threadIdx.x) + 392) / 18) * 42)) + ((((int)blockIdx.x) % 3) * 14)) + ((((int)threadIdx.x) + 14) % 18)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 588))] = (((2 <= (((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 12) % 18))) && ((((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 12) % 18)) < 44)) ? data[(((((((((int)blockIdx.x) / 3) * 3528) + (((((int)threadIdx.x) + 588) / 18) * 42)) + ((((int)blockIdx.x) % 3) * 14)) + ((((int)threadIdx.x) + 12) % 18)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 784))] = (((((36 <= ((((int)threadIdx.x) + 784) % 828)) && (((((int)threadIdx.x) + 784) % 828) < 792)) && (2 <= (((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 10) % 18)))) && ((((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 10) % 18)) < 44)) ? data[((((((((((int)blockIdx.x) / 3) * 3528) + (((((int)threadIdx.x) + 784) / 828) * 1764)) + ((((((int)threadIdx.x) + 784) % 828) / 18) * 42)) + ((((int)blockIdx.x) % 3) * 14)) + ((((int)threadIdx.x) + 10) % 18)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 980))] = (((2 <= (((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 8) % 18))) && ((((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 8) % 18)) < 44)) ? data[((((((((((int)blockIdx.x) / 3) * 3528) + (((((int)threadIdx.x) + 980) / 828) * 1764)) + (((((int)threadIdx.x) + 152) / 18) * 42)) + ((((int)blockIdx.x) % 3) * 14)) + ((((int)threadIdx.x) + 8) % 18)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1176))] = (((2 <= (((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 6) % 18))) && ((((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 6) % 18)) < 44)) ? data[((((((((((int)blockIdx.x) / 3) * 3528) + (((((int)threadIdx.x) + 1176) / 828) * 1764)) + (((((int)threadIdx.x) + 348) / 18) * 42)) + ((((int)blockIdx.x) % 3) * 14)) + ((((int)threadIdx.x) + 6) % 18)) - 86))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1372))] = (((2 <= (((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 4) % 18))) && ((((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 4) % 18)) < 44)) ? data[((((((((((int)blockIdx.x) / 3) * 3528) + (((((int)threadIdx.x) + 1372) / 828) * 1764)) + (((((int)threadIdx.x) + 544) / 18) * 42)) + ((((int)blockIdx.x) % 3) * 14)) + ((((int)threadIdx.x) + 4) % 18)) - 86))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 88) {
    PaddedInput_shared[((((int)threadIdx.x) + 1568))] = ((((((int)threadIdx.x) < 52) && (2 <= (((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 2) % 18)))) && ((((((int)blockIdx.x) % 3) * 14) + ((((int)threadIdx.x) + 2) % 18)) < 44)) ? data[((((((((((int)blockIdx.x) / 3) * 3528) + (((((int)threadIdx.x) + 1568) / 828) * 1764)) + (((((int)threadIdx.x) + 740) / 18) * 42)) + ((((int)blockIdx.x) % 3) * 14)) + ((((int)threadIdx.x) + 2) % 18)) - 86))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 50) {
    kernel_shared[(((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) % 126) / 3) * 50) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 18))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 36))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 54))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 72))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 18))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 36))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 54))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 72))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 90))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 36))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 54))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 72))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 90))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 108))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 828))] * kernel_shared[(25)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 846))] * kernel_shared[(30)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 864))] * kernel_shared[(35)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 882))] * kernel_shared[(40)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 900))] * kernel_shared[(45)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 846))] * kernel_shared[(25)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 864))] * kernel_shared[(30)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 882))] * kernel_shared[(35)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 900))] * kernel_shared[(40)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 918))] * kernel_shared[(45)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 864))] * kernel_shared[(25)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 882))] * kernel_shared[(30)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 900))] * kernel_shared[(35)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 918))] * kernel_shared[(40)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 936))] * kernel_shared[(45)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 1))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 19))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 37))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 55))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 73))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 19))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 37))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 55))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 73))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 91))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 37))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 55))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 73))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 91))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 109))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 829))] * kernel_shared[(26)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 847))] * kernel_shared[(31)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 865))] * kernel_shared[(36)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 883))] * kernel_shared[(41)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 901))] * kernel_shared[(46)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 847))] * kernel_shared[(26)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 865))] * kernel_shared[(31)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 883))] * kernel_shared[(36)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 901))] * kernel_shared[(41)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 919))] * kernel_shared[(46)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 865))] * kernel_shared[(26)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 883))] * kernel_shared[(31)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 901))] * kernel_shared[(36)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 919))] * kernel_shared[(41)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 937))] * kernel_shared[(46)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 2))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 20))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 38))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 56))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 74))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 20))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 38))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 56))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 74))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 92))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 38))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 56))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 74))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 92))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 110))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 830))] * kernel_shared[(27)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 848))] * kernel_shared[(32)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 866))] * kernel_shared[(37)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 884))] * kernel_shared[(42)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 902))] * kernel_shared[(47)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 848))] * kernel_shared[(27)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 866))] * kernel_shared[(32)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 884))] * kernel_shared[(37)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 902))] * kernel_shared[(42)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 920))] * kernel_shared[(47)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 866))] * kernel_shared[(27)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 884))] * kernel_shared[(32)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 902))] * kernel_shared[(37)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 920))] * kernel_shared[(42)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 938))] * kernel_shared[(47)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 3))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 21))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 39))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 57))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 75))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 21))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 39))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 57))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 75))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 93))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 39))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 57))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 75))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 93))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 111))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 831))] * kernel_shared[(28)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 849))] * kernel_shared[(33)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 867))] * kernel_shared[(38)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 885))] * kernel_shared[(43)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 903))] * kernel_shared[(48)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 849))] * kernel_shared[(28)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 867))] * kernel_shared[(33)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 885))] * kernel_shared[(38)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 903))] * kernel_shared[(43)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 921))] * kernel_shared[(48)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 867))] * kernel_shared[(28)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 885))] * kernel_shared[(33)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 903))] * kernel_shared[(38)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 921))] * kernel_shared[(43)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 939))] * kernel_shared[(48)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 4))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 22))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 40))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 58))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 76))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 22))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 40))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 58))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 76))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 94))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 40))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 58))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 76))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 94))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 112))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 832))] * kernel_shared[(29)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 850))] * kernel_shared[(34)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 868))] * kernel_shared[(39)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 886))] * kernel_shared[(44)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 904))] * kernel_shared[(49)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 850))] * kernel_shared[(29)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 868))] * kernel_shared[(34)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 886))] * kernel_shared[(39)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 904))] * kernel_shared[(44)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 922))] * kernel_shared[(49)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 868))] * kernel_shared[(29)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 886))] * kernel_shared[(34)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 904))] * kernel_shared[(39)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 922))] * kernel_shared[(44)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 14) * 54) + (((int)threadIdx.x) % 14)) + 940))] * kernel_shared[(49)]));
  for (int c_inner = 0; c_inner < 2; ++c_inner) {
    for (int i_inner = 0; i_inner < 3; ++i_inner) {
      DepthwiseConv2d[((((((((((int)blockIdx.x) / 3) * 3528) + (c_inner * 1764)) + ((((int)threadIdx.x) / 14) * 126)) + (i_inner * 42)) + ((((int)blockIdx.x) % 3) * 14)) + (((int)threadIdx.x) % 14)))] = DepthwiseConv2d_local[(((c_inner * 3) + i_inner))];
    }
  }
}

