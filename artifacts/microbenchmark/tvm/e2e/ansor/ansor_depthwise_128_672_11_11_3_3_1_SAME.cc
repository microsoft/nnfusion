//21504_1_1_44_1_1
//128_672_11_11_3_1_SAME
//dim3 grid(21504, 1, 1);
//dim3 block(44, 1, 1);

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
extern "C" __global__ void __launch_bounds__(44) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[11];
  __shared__ float PaddedInput_shared[572];
  __shared__ float kernel_shared[3];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  DepthwiseConv2d_local[(7)] = 0.000000e+00f;
  DepthwiseConv2d_local[(8)] = 0.000000e+00f;
  DepthwiseConv2d_local[(9)] = 0.000000e+00f;
  DepthwiseConv2d_local[(10)] = 0.000000e+00f;
  for (int dj_outer_outer = 0; dj_outer_outer < 3; ++dj_outer_outer) {
    __syncthreads();
    PaddedInput_shared[(((int)threadIdx.x))] = ((((11 <= ((int)threadIdx.x)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 11)))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((int)blockIdx.x) / 672) * 325248) + ((((int)blockIdx.x) % 672) * 121)) + ((int)threadIdx.x)) + dj_outer_outer) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 44))] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 11))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((int)blockIdx.x) / 672) * 325248) + ((((int)blockIdx.x) % 672) * 121)) + ((int)threadIdx.x)) + dj_outer_outer) + 32))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 88))] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 11))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((int)blockIdx.x) / 672) * 325248) + ((((int)blockIdx.x) % 672) * 121)) + ((int)threadIdx.x)) + dj_outer_outer) + 76))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 132))] = (((((1 <= (((((int)threadIdx.x) / 11) + 12) % 13)) && ((((((int)threadIdx.x) / 11) + 12) % 13) < 12)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 11)))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 132) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + ((((((int)threadIdx.x) / 11) + 12) % 13) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 176))] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 11))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 176) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + (((((int)threadIdx.x) / 11) + 3) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 220))] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 11))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 220) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + (((((int)threadIdx.x) / 11) + 7) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 264))] = (((((1 <= (((((int)threadIdx.x) / 11) + 11) % 13)) && ((((((int)threadIdx.x) / 11) + 11) % 13) < 12)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 11)))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 264) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + ((((((int)threadIdx.x) / 11) + 11) % 13) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 308))] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 11))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 308) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + (((((int)threadIdx.x) / 11) + 2) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 352))] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 11))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 352) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + (((((int)threadIdx.x) / 11) + 6) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 396))] = (((((1 <= (((((int)threadIdx.x) / 11) + 10) % 13)) && ((((((int)threadIdx.x) / 11) + 10) % 13) < 12)) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 11)))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 396) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + ((((((int)threadIdx.x) / 11) + 10) % 13) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 440))] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 11))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 440) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + (((((int)threadIdx.x) / 11) + 1) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 484))] = (((1 <= (dj_outer_outer + (((int)threadIdx.x) % 11))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 484) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + (((((int)threadIdx.x) / 11) + 5) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 528))] = ((((((int)threadIdx.x) < 33) && (1 <= (dj_outer_outer + (((int)threadIdx.x) % 11)))) && ((dj_outer_outer + (((int)threadIdx.x) % 11)) < 12)) ? data[(((((((((((int)blockIdx.x) / 672) * 325248) + (((((int)threadIdx.x) + 528) / 143) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + (((((int)threadIdx.x) / 11) + 9) * 11)) + dj_outer_outer) + (((int)threadIdx.x) % 11)) - 12))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 3) {
      kernel_shared[(((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) % 672) * 9) + (((int)threadIdx.x) * 3)) + dj_outer_outer))];
    }
    __syncthreads();
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 11))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 22))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 11))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 22))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 33))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 22))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 33))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 44))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 33))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 44))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 55))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 44))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 55))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 66))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 55))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 66))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 77))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 66))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 77))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 88))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(7)] = (DepthwiseConv2d_local[(7)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 77))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(7)] = (DepthwiseConv2d_local[(7)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 88))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(7)] = (DepthwiseConv2d_local[(7)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 99))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(8)] = (DepthwiseConv2d_local[(8)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 88))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(8)] = (DepthwiseConv2d_local[(8)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 99))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(8)] = (DepthwiseConv2d_local[(8)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 110))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(9)] = (DepthwiseConv2d_local[(9)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 99))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(9)] = (DepthwiseConv2d_local[(9)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 110))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(9)] = (DepthwiseConv2d_local[(9)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 121))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(10)] = (DepthwiseConv2d_local[(10)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 110))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(10)] = (DepthwiseConv2d_local[(10)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 121))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(10)] = (DepthwiseConv2d_local[(10)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 143) + (((int)threadIdx.x) % 11)) + 132))] * kernel_shared[(2)]));
  }
  for (int i_inner = 0; i_inner < 11; ++i_inner) {
    DepthwiseConv2d[(((((((((int)blockIdx.x) / 672) * 325248) + ((((int)threadIdx.x) / 11) * 81312)) + ((((int)blockIdx.x) % 672) * 121)) + (i_inner * 11)) + (((int)threadIdx.x) % 11)))] = DepthwiseConv2d_local[(i_inner)];
  }
}

