//27888_1_1_332_1_1
//128_42_83_83_5_1_SAME
//dim3 grid(27888, 1, 1);
//dim3 block(332, 1, 1);

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
extern "C" __global__ void __launch_bounds__(332) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[4];
  __shared__ float PaddedInput_shared[1392];
  __shared__ float kernel_shared[10];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  for (int di_outer_outer = 0; di_outer_outer < 5; ++di_outer_outer) {
    __syncthreads();
    PaddedInput_shared[((((int)threadIdx.x) * 2))] = (((((2 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 85)) && (2 <= ((((int)threadIdx.x) * 2) % 87))) && (((((int)threadIdx.x) * 2) % 87) < 85)) ? data[((((((((((((int)blockIdx.x) / 1743) * 2314704) + ((((int)threadIdx.x) / 87) * 289338)) + (((((int)blockIdx.x) % 1743) / 83) * 13778)) + ((((((int)threadIdx.x) % 87) * 2) / 87) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + ((((int)threadIdx.x) * 2) % 87)) - 168))] : 0.000000e+00f);
    PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1))] = (((((2 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 85)) && (2 <= (((((int)threadIdx.x) * 2) + 1) % 87))) && ((((((int)threadIdx.x) * 2) + 1) % 87) < 85)) ? data[((((((((((((int)blockIdx.x) / 1743) * 2314704) + ((((((int)threadIdx.x) * 2) + 1) / 174) * 289338)) + (((((int)blockIdx.x) % 1743) / 83) * 13778)) + (((((((int)threadIdx.x) * 2) + 1) % 174) / 87) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + (((((int)threadIdx.x) * 2) + 1) % 87)) - 168))] : 0.000000e+00f);
    PaddedInput_shared[(((((int)threadIdx.x) * 2) + 664))] = (((((2 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 85)) && (2 <= (((((int)threadIdx.x) * 2) + 55) % 87))) && ((((((int)threadIdx.x) * 2) + 55) % 87) < 85)) ? data[((((((((((((int)blockIdx.x) / 1743) * 2314704) + ((((((int)threadIdx.x) * 2) + 664) / 174) * 289338)) + (((((int)blockIdx.x) % 1743) / 83) * 13778)) + (((((((int)threadIdx.x) * 2) + 142) % 174) / 87) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + (((((int)threadIdx.x) * 2) + 55) % 87)) - 168))] : 0.000000e+00f);
    PaddedInput_shared[(((((int)threadIdx.x) * 2) + 665))] = (((((2 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 85)) && (2 <= (((((int)threadIdx.x) * 2) + 56) % 87))) && ((((((int)threadIdx.x) * 2) + 56) % 87) < 85)) ? data[((((((((((((int)blockIdx.x) / 1743) * 2314704) + ((((((int)threadIdx.x) * 2) + 665) / 174) * 289338)) + (((((int)blockIdx.x) % 1743) / 83) * 13778)) + (((((((int)threadIdx.x) * 2) + 143) % 174) / 87) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + (((((int)threadIdx.x) * 2) + 56) % 87)) - 168))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 32) {
      PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1328))] = ((((2 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 85)) && (((int)threadIdx.x) < 31)) ? data[((((((((((((int)blockIdx.x) / 1743) * 2314704) + ((((((int)threadIdx.x) * 2) + 1328) / 174) * 289338)) + (((((int)blockIdx.x) % 1743) / 83) * 13778)) + ((((((int)threadIdx.x) * 2) + 110) / 87) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + ((((int)threadIdx.x) * 2) + 23)) - 168))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 32) {
      PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1329))] = ((((2 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 85)) && (((int)threadIdx.x) < 31)) ? data[((((((((((((int)blockIdx.x) / 1743) * 2314704) + ((((((int)threadIdx.x) * 2) + 1329) / 174) * 289338)) + (((((int)blockIdx.x) % 1743) / 83) * 13778)) + ((((((int)threadIdx.x) * 2) + 111) / 87) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + ((((int)threadIdx.x) * 2) + 24)) - 168))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 10) {
      kernel_shared[(((int)threadIdx.x))] = kernel[(((((((((int)blockIdx.x) % 1743) / 83) * 50) + ((((int)threadIdx.x) / 5) * 25)) + (di_outer_outer * 5)) + (((int)threadIdx.x) % 5)))];
    }
    __syncthreads();
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 174))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 1))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 175))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 2))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 176))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 3))] * kernel_shared[(3)]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 177))] * kernel_shared[(3)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 4))] * kernel_shared[(4)]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 178))] * kernel_shared[(4)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 87))] * kernel_shared[(5)]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 261))] * kernel_shared[(5)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 88))] * kernel_shared[(6)]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 262))] * kernel_shared[(6)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 89))] * kernel_shared[(7)]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 263))] * kernel_shared[(7)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 90))] * kernel_shared[(8)]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 264))] * kernel_shared[(8)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 91))] * kernel_shared[(9)]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 83) * 348) + (((int)threadIdx.x) % 83)) + 265))] * kernel_shared[(9)]));
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    for (int c_inner = 0; c_inner < 2; ++c_inner) {
      DepthwiseConv2d[(((((((((((int)blockIdx.x) / 1743) * 2314704) + ((((int)threadIdx.x) / 83) * 578676)) + (b_inner * 289338)) + (((((int)blockIdx.x) % 1743) / 83) * 13778)) + (c_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))] = DepthwiseConv2d_local[(((b_inner * 2) + c_inner))];
    }
  }
}

