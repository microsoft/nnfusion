//43008_1_1_63_1_1
//128_336_21_21_5_1_SAME
//dim3 grid(43008, 1, 1);
//dim3 block(63, 1, 1);

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
extern "C" __global__ void __launch_bounds__(63) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[7];
  __shared__ float PaddedInput_shared[625];
  __shared__ float kernel_shared[25];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  PaddedInput_shared[((((int)threadIdx.x) * 5))] = (((10 <= ((int)threadIdx.x)) && (1 <= (((int)threadIdx.x) % 5))) ? data[(((((((int)blockIdx.x) * 441) + ((((int)threadIdx.x) / 5) * 21)) + ((((int)threadIdx.x) % 5) * 5)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 5) + 1))] = ((((10 <= ((int)threadIdx.x)) && (2 <= (((((int)threadIdx.x) * 5) + 1) % 25))) && ((((((int)threadIdx.x) * 5) + 1) % 25) < 23)) ? data[(((((((int)blockIdx.x) * 441) + ((((((int)threadIdx.x) * 5) + 1) / 25) * 21)) + (((((int)threadIdx.x) * 5) + 1) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 5) + 2))] = ((((10 <= ((int)threadIdx.x)) && (2 <= (((((int)threadIdx.x) * 5) + 2) % 25))) && ((((((int)threadIdx.x) * 5) + 2) % 25) < 23)) ? data[(((((((int)blockIdx.x) * 441) + ((((((int)threadIdx.x) * 5) + 2) / 25) * 21)) + (((((int)threadIdx.x) * 5) + 2) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 5) + 3))] = ((((10 <= ((int)threadIdx.x)) && (2 <= (((((int)threadIdx.x) * 5) + 3) % 25))) && ((((((int)threadIdx.x) * 5) + 3) % 25) < 23)) ? data[(((((((int)blockIdx.x) * 441) + ((((((int)threadIdx.x) * 5) + 3) / 25) * 21)) + (((((int)threadIdx.x) * 5) + 3) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 5) + 4))] = ((((10 <= ((int)threadIdx.x)) && (2 <= (((((int)threadIdx.x) * 5) + 4) % 25))) && ((((((int)threadIdx.x) * 5) + 4) % 25) < 23)) ? data[(((((((int)blockIdx.x) * 441) + ((((((int)threadIdx.x) * 5) + 4) / 25) * 21)) + (((((int)threadIdx.x) * 5) + 4) % 25)) - 44))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 62) {
    PaddedInput_shared[(((((int)threadIdx.x) * 5) + 315))] = ((((((int)threadIdx.x) < 52) && (2 <= (((((int)threadIdx.x) * 5) + 15) % 25))) && ((((((int)threadIdx.x) * 5) + 15) % 25) < 23)) ? data[(((((((int)blockIdx.x) * 441) + ((((((int)threadIdx.x) * 5) + 315) / 25) * 21)) + (((((int)threadIdx.x) * 5) + 15) % 25)) - 44))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 62) {
    PaddedInput_shared[(((((int)threadIdx.x) * 5) + 316))] = ((((((int)threadIdx.x) < 52) && (2 <= (((((int)threadIdx.x) * 5) + 16) % 25))) && ((((((int)threadIdx.x) * 5) + 16) % 25) < 23)) ? data[(((((((int)blockIdx.x) * 441) + ((((((int)threadIdx.x) * 5) + 316) / 25) * 21)) + (((((int)threadIdx.x) * 5) + 16) % 25)) - 44))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 62) {
    PaddedInput_shared[(((((int)threadIdx.x) * 5) + 317))] = ((((((int)threadIdx.x) < 52) && (2 <= (((((int)threadIdx.x) * 5) + 17) % 25))) && ((((((int)threadIdx.x) * 5) + 17) % 25) < 23)) ? data[(((((((int)blockIdx.x) * 441) + ((((((int)threadIdx.x) * 5) + 317) / 25) * 21)) + (((((int)threadIdx.x) * 5) + 17) % 25)) - 44))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 62) {
    PaddedInput_shared[(((((int)threadIdx.x) * 5) + 318))] = ((((((int)threadIdx.x) < 52) && (2 <= (((((int)threadIdx.x) * 5) + 18) % 25))) && ((((((int)threadIdx.x) * 5) + 18) % 25) < 23)) ? data[(((((((int)blockIdx.x) * 441) + ((((((int)threadIdx.x) * 5) + 318) / 25) * 21)) + (((((int)threadIdx.x) * 5) + 18) % 25)) - 44))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 62) {
    PaddedInput_shared[(((((int)threadIdx.x) * 5) + 319))] = ((((((int)threadIdx.x) < 52) && (2 <= (((((int)threadIdx.x) * 5) + 19) % 25))) && ((((((int)threadIdx.x) * 5) + 19) % 25) < 23)) ? data[(((((((int)blockIdx.x) * 441) + ((((((int)threadIdx.x) * 5) + 319) / 25) * 21)) + (((((int)threadIdx.x) * 5) + 19) % 25)) - 44))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 25) {
    kernel_shared[(((int)threadIdx.x))] = kernel[((((((int)blockIdx.x) % 336) * 25) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 25))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 50))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 75))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 100))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 125))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 150))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 1))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 26))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 51))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 76))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 101))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 126))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 151))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 2))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 27))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 52))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 77))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 102))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 127))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 152))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 3))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 28))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 53))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 78))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 103))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 128))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 153))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 4))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 29))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 54))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 79))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 104))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 129))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 154))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 25))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 50))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 75))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 100))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 125))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 150))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 175))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 26))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 51))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 76))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 101))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 126))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 151))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 176))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 27))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 52))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 77))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 102))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 127))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 152))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 177))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 28))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 53))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 78))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 103))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 128))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 153))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 178))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 29))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 54))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 79))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 104))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 129))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 154))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 179))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 50))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 75))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 100))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 125))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 150))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 175))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 200))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 51))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 76))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 101))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 126))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 151))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 176))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 201))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 52))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 77))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 102))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 127))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 152))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 177))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 202))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 53))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 78))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 103))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 128))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 153))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 178))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 203))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 54))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 79))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 104))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 129))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 154))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 179))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 204))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 75))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 100))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 125))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 150))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 175))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 200))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 225))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 76))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 101))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 126))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 151))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 176))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 201))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 226))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 77))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 102))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 127))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 152))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 177))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 202))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 227))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 78))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 103))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 128))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 153))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 178))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 203))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 228))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 79))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 104))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 129))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 154))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 179))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 204))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 229))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 100))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 125))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 150))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 175))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 200))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 225))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 250))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 101))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 126))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 151))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 176))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 201))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 226))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 251))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 102))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 127))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 152))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 177))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 202))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 227))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 252))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 103))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 128))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 153))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 178))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 203))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 228))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 253))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 104))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 129))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 154))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 179))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 204))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 229))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 21) * 175) + (((int)threadIdx.x) % 21)) + 254))] * kernel_shared[(24)]));
  for (int i_inner = 0; i_inner < 7; ++i_inner) {
    DepthwiseConv2d[(((((((int)blockIdx.x) * 441) + ((((int)threadIdx.x) / 21) * 147)) + (i_inner * 21)) + (((int)threadIdx.x) % 21)))] = DepthwiseConv2d_local[(i_inner)];
  }
}

