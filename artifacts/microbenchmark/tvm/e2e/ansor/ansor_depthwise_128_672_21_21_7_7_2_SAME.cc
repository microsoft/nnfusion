//16896_1_1_154_1_1
//128_672_21_21_7_2_SAME
//dim3 grid(16896, 1, 1);
//dim3 block(154, 1, 1);

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
extern "C" __global__ void __launch_bounds__(154) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[4];
  __shared__ float PaddedInput_shared[1512];
  __shared__ float kernel_shared[196];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  for (int di_outer_outer = 0; di_outer_outer < 7; ++di_outer_outer) {
    __syncthreads();
    PaddedInput_shared[(((int)threadIdx.x))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= (((int)threadIdx.x) % 27))) && ((((int)threadIdx.x) % 27) < 24)) ? data[(((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + ((((int)threadIdx.x) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + (((int)threadIdx.x) % 27)) - 66))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 154))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= ((((int)threadIdx.x) + 19) % 27))) && (((((int)threadIdx.x) + 19) % 27) < 24)) ? data[(((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + (((((int)threadIdx.x) + 154) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 19) % 27)) - 66))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 308))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= ((((int)threadIdx.x) + 11) % 27))) && (((((int)threadIdx.x) + 11) % 27) < 24)) ? data[(((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + (((((int)threadIdx.x) + 308) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 11) % 27)) - 66))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 462))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= ((((int)threadIdx.x) + 3) % 27))) && (((((int)threadIdx.x) + 3) % 27) < 24)) ? data[(((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + (((((int)threadIdx.x) + 462) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 3) % 27)) - 66))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 616))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= ((((int)threadIdx.x) + 22) % 27))) && (((((int)threadIdx.x) + 22) % 27) < 24)) ? data[((((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)threadIdx.x) + 616) / 756) * 296352)) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + ((((((int)threadIdx.x) + 616) % 756) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 22) % 27)) - 66))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 770))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= ((((int)threadIdx.x) + 14) % 27))) && (((((int)threadIdx.x) + 14) % 27) < 24)) ? data[((((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)threadIdx.x) + 770) / 756) * 296352)) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + (((((int)threadIdx.x) + 14) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 14) % 27)) - 66))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 924))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= ((((int)threadIdx.x) + 6) % 27))) && (((((int)threadIdx.x) + 6) % 27) < 24)) ? data[((((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)threadIdx.x) + 924) / 756) * 296352)) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + (((((int)threadIdx.x) + 168) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 6) % 27)) - 66))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 1078))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= ((((int)threadIdx.x) + 25) % 27))) && (((((int)threadIdx.x) + 25) % 27) < 24)) ? data[((((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)threadIdx.x) + 1078) / 756) * 296352)) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + (((((int)threadIdx.x) + 322) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 25) % 27)) - 66))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 1232))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= ((((int)threadIdx.x) + 17) % 27))) && (((((int)threadIdx.x) + 17) % 27) < 24)) ? data[((((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)threadIdx.x) + 1232) / 756) * 296352)) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + (((((int)threadIdx.x) + 476) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 17) % 27)) - 66))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 126) {
      PaddedInput_shared[((((int)threadIdx.x) + 1386))] = (((((3 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 24)) && (3 <= ((((int)threadIdx.x) + 9) % 27))) && (((((int)threadIdx.x) + 9) % 27) < 24)) ? data[((((((((((((int)blockIdx.x) / 264) * 592704) + (((((int)threadIdx.x) + 1386) / 756) * 296352)) + (((((int)blockIdx.x) % 264) / 11) * 12348)) + (((((int)threadIdx.x) + 630) / 27) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 9) % 27)) - 66))] : 0.000000e+00f);
    }
    kernel_shared[(((int)threadIdx.x))] = kernel[(((((((((int)blockIdx.x) % 264) / 11) * 1372) + ((((int)threadIdx.x) / 7) * 49)) + (di_outer_outer * 7)) + (((int)threadIdx.x) % 7)))];
    if (((int)threadIdx.x) < 42) {
      kernel_shared[((((int)threadIdx.x) + 154))] = kernel[((((((((((int)blockIdx.x) % 264) / 11) * 1372) + ((((int)threadIdx.x) / 7) * 49)) + (di_outer_outer * 7)) + (((int)threadIdx.x) % 7)) + 1078))];
    }
    __syncthreads();
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)))] * kernel_shared[(((((int)threadIdx.x) / 11) * 14))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 27))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 7))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 1))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 1))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 28))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 8))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 2))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 2))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 29))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 9))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 3))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 3))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 30))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 10))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 4))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 4))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 31))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 11))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 5))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 5))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 32))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 12))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 6))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 6))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 33))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 13))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 756))] * kernel_shared[(((((int)threadIdx.x) / 11) * 14))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 783))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 7))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 757))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 1))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 784))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 8))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 758))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 2))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 785))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 9))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 759))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 3))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 786))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 10))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 760))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 4))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 787))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 11))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 761))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 5))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 788))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 12))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 762))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 6))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 54) + ((((int)threadIdx.x) % 11) * 2)) + 789))] * kernel_shared[((((((int)threadIdx.x) / 11) * 14) + 13))]));
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    for (int c_inner = 0; c_inner < 2; ++c_inner) {
      DepthwiseConv2d[(((((((((((int)blockIdx.x) / 264) * 162624) + (b_inner * 81312)) + (((((int)blockIdx.x) % 264) / 11) * 3388)) + ((((int)threadIdx.x) / 11) * 242)) + (c_inner * 121)) + ((((int)blockIdx.x) % 11) * 11)) + (((int)threadIdx.x) % 11)))] = DepthwiseConv2d_local[(((b_inner * 2) + c_inner))];
    }
  }
}

