//118272_1_1_22_1_1
//128_672_21_21_5_2_SAME
//dim3 grid(118272, 1, 1);
//dim3 block(22, 1, 1);

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
extern "C" __global__ void __launch_bounds__(22) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[4];
  __shared__ float PaddedInput_shared[200];
  __shared__ float kernel_shared[10];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  for (int di_outer_outer = 0; di_outer_outer < 5; ++di_outer_outer) {
    __syncthreads();
    PaddedInput_shared[(((int)threadIdx.x))] = ((((2 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 23)) && (2 <= ((int)threadIdx.x))) ? data[((((((((((int)blockIdx.x) / 3696) * 1185408) + (((((int)blockIdx.x) % 3696) / 11) * 882)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((int)threadIdx.x)) - 44))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 22))] = (((((2 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 23)) && (2 <= ((((int)threadIdx.x) + 22) % 25))) && (((((int)threadIdx.x) + 22) % 25) < 23)) ? data[(((((((((((int)blockIdx.x) / 3696) * 1185408) + (((((int)blockIdx.x) % 3696) / 11) * 882)) + (((((int)threadIdx.x) + 22) / 25) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 22) % 25)) - 44))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 44))] = (((((2 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 23)) && (2 <= ((((int)threadIdx.x) + 19) % 25))) && (((((int)threadIdx.x) + 19) % 25) < 23)) ? data[((((((((((((int)blockIdx.x) / 3696) * 1185408) + (((((int)threadIdx.x) + 44) / 50) * 296352)) + (((((int)blockIdx.x) % 3696) / 11) * 882)) + ((((((int)threadIdx.x) + 44) % 50) / 25) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 19) % 25)) - 44))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 66))] = (((((2 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 23)) && (2 <= ((((int)threadIdx.x) + 16) % 25))) && (((((int)threadIdx.x) + 16) % 25) < 23)) ? data[((((((((((((int)blockIdx.x) / 3696) * 1185408) + (((((int)threadIdx.x) + 66) / 50) * 296352)) + (((((int)blockIdx.x) % 3696) / 11) * 882)) + (((((int)threadIdx.x) + 16) / 25) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 16) % 25)) - 44))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 88))] = (((((2 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 23)) && (2 <= ((((int)threadIdx.x) + 13) % 25))) && (((((int)threadIdx.x) + 13) % 25) < 23)) ? data[((((((((((((int)blockIdx.x) / 3696) * 1185408) + (((((int)threadIdx.x) + 88) / 50) * 296352)) + (((((int)blockIdx.x) % 3696) / 11) * 882)) + ((((((int)threadIdx.x) + 38) % 50) / 25) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 13) % 25)) - 44))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 110))] = (((((2 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 23)) && (2 <= ((((int)threadIdx.x) + 10) % 25))) && (((((int)threadIdx.x) + 10) % 25) < 23)) ? data[((((((((((((int)blockIdx.x) / 3696) * 1185408) + (((((int)threadIdx.x) + 110) / 50) * 296352)) + (((((int)blockIdx.x) % 3696) / 11) * 882)) + (((((int)threadIdx.x) + 10) / 25) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 10) % 25)) - 44))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 132))] = (((((2 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 23)) && (2 <= ((((int)threadIdx.x) + 7) % 25))) && (((((int)threadIdx.x) + 7) % 25) < 23)) ? data[((((((((((((int)blockIdx.x) / 3696) * 1185408) + (((((int)threadIdx.x) + 132) / 50) * 296352)) + (((((int)blockIdx.x) % 3696) / 11) * 882)) + ((((((int)threadIdx.x) + 32) % 50) / 25) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 7) % 25)) - 44))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 154))] = (((((2 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 23)) && (2 <= ((((int)threadIdx.x) + 4) % 25))) && (((((int)threadIdx.x) + 4) % 25) < 23)) ? data[((((((((((((int)blockIdx.x) / 3696) * 1185408) + (((((int)threadIdx.x) + 154) / 50) * 296352)) + (((((int)blockIdx.x) % 3696) / 11) * 882)) + (((((int)threadIdx.x) + 4) / 25) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + ((((int)threadIdx.x) + 4) % 25)) - 44))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 176))] = ((((2 <= (((((int)blockIdx.x) % 11) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 11) * 2) + di_outer_outer) < 23)) && (1 <= ((int)threadIdx.x))) ? data[((((((((((((int)blockIdx.x) / 3696) * 1185408) + (((((int)threadIdx.x) + 176) / 50) * 296352)) + (((((int)blockIdx.x) % 3696) / 11) * 882)) + (((((int)threadIdx.x) + 26) / 25) * 441)) + ((((int)blockIdx.x) % 11) * 42)) + (di_outer_outer * 21)) + (((int)threadIdx.x) + 1)) - 44))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 2) {
      PaddedInput_shared[((((int)threadIdx.x) + 198))] = 0.000000e+00f;
    }
    if (((int)threadIdx.x) < 10) {
      kernel_shared[(((int)threadIdx.x))] = kernel[(((((((((int)blockIdx.x) % 3696) / 11) * 50) + ((((int)threadIdx.x) / 5) * 25)) + (di_outer_outer * 5)) + (((int)threadIdx.x) % 5)))];
    }
    __syncthreads();
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)))] * kernel_shared[(((((int)threadIdx.x) / 11) * 5))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 100))] * kernel_shared[(((((int)threadIdx.x) / 11) * 5))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 50))] * kernel_shared[(((((int)threadIdx.x) / 11) * 5))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 150))] * kernel_shared[(((((int)threadIdx.x) / 11) * 5))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 1))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 1))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 101))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 1))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 51))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 1))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 151))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 1))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 2))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 2))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 102))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 2))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 52))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 2))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 152))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 2))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 3))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 3))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 103))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 3))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 53))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 3))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 153))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 3))]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 4))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 4))]));
    DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 104))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 4))]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 54))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 4))]));
    DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 11) * 25) + ((((int)threadIdx.x) % 11) * 2)) + 154))] * kernel_shared[((((((int)threadIdx.x) / 11) * 5) + 4))]));
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    DepthwiseConv2d[((((((((((int)blockIdx.x) / 3696) * 325248) + (b_inner * 81312)) + (((((int)blockIdx.x) % 3696) / 11) * 242)) + ((((int)threadIdx.x) / 11) * 121)) + ((((int)blockIdx.x) % 11) * 11)) + (((int)threadIdx.x) % 11)))] = DepthwiseConv2d_local[(b_inner)];
    DepthwiseConv2d[(((((((((((int)blockIdx.x) / 3696) * 325248) + (b_inner * 81312)) + (((((int)blockIdx.x) % 3696) / 11) * 242)) + ((((int)threadIdx.x) / 11) * 121)) + ((((int)blockIdx.x) % 11) * 11)) + (((int)threadIdx.x) % 11)) + 162624))] = DepthwiseConv2d_local[((b_inner + 2))];
  }
}

