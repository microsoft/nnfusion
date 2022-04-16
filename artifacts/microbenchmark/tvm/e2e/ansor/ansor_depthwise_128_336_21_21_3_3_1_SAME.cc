//10752_1_1_588_1_1
//128_336_21_21_3_1_SAME
//dim3 grid(10752, 1, 1);
//dim3 block(588, 1, 1);

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
extern "C" __global__ void __launch_bounds__(588) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[3];
  __shared__ float PaddedInput_shared[2116];
  __shared__ float kernel_shared[36];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 529) {
    PaddedInput_shared[((((int)threadIdx.x) * 4))] = (((((23 <= ((((int)threadIdx.x) * 4) % 529)) && (((((int)threadIdx.x) * 4) % 529) < 506)) && (1 <= ((((int)threadIdx.x) * 4) % 23))) && (((((int)threadIdx.x) * 4) % 23) < 22)) ? data[((((((((int)blockIdx.x) * 1764) + (((((int)threadIdx.x) * 4) / 529) * 441)) + ((((((int)threadIdx.x) * 4) % 529) / 23) * 21)) + ((((int)threadIdx.x) * 4) % 23)) - 22))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 529) {
    PaddedInput_shared[(((((int)threadIdx.x) * 4) + 1))] = (((((23 <= (((((int)threadIdx.x) * 4) + 1) % 529)) && ((((((int)threadIdx.x) * 4) + 1) % 529) < 506)) && (1 <= (((((int)threadIdx.x) * 4) + 1) % 23))) && ((((((int)threadIdx.x) * 4) + 1) % 23) < 22)) ? data[((((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 4) + 1) / 529) * 441)) + (((((((int)threadIdx.x) * 4) + 1) % 529) / 23) * 21)) + (((((int)threadIdx.x) * 4) + 1) % 23)) - 22))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 529) {
    PaddedInput_shared[(((((int)threadIdx.x) * 4) + 2))] = (((((23 <= (((((int)threadIdx.x) * 4) + 2) % 529)) && ((((((int)threadIdx.x) * 4) + 2) % 529) < 506)) && (1 <= (((((int)threadIdx.x) * 4) + 2) % 23))) && ((((((int)threadIdx.x) * 4) + 2) % 23) < 22)) ? data[((((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 4) + 2) / 529) * 441)) + (((((((int)threadIdx.x) * 4) + 2) % 529) / 23) * 21)) + (((((int)threadIdx.x) * 4) + 2) % 23)) - 22))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 529) {
    PaddedInput_shared[(((((int)threadIdx.x) * 4) + 3))] = (((((23 <= (((((int)threadIdx.x) * 4) + 3) % 529)) && ((((((int)threadIdx.x) * 4) + 3) % 529) < 506)) && (1 <= (((((int)threadIdx.x) * 4) + 3) % 23))) && ((((((int)threadIdx.x) * 4) + 3) % 23) < 22)) ? data[((((((((int)blockIdx.x) * 1764) + ((((((int)threadIdx.x) * 4) + 3) / 529) * 441)) + (((((((int)threadIdx.x) * 4) + 3) % 529) / 23) * 21)) + (((((int)threadIdx.x) * 4) + 3) % 23)) - 22))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 18) {
    kernel_shared[((((int)threadIdx.x) * 2))] = kernel[((((((int)blockIdx.x) % 84) * 36) + (((int)threadIdx.x) * 2)))];
  }
  if (((int)threadIdx.x) < 18) {
    kernel_shared[(((((int)threadIdx.x) * 2) + 1))] = kernel[(((((((int)blockIdx.x) % 84) * 36) + (((int)threadIdx.x) * 2)) + 1))];
  }
  __syncthreads();
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)))] * kernel_shared[(((((int)threadIdx.x) / 147) * 9))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 23))] * kernel_shared[(((((int)threadIdx.x) / 147) * 9))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 46))] * kernel_shared[(((((int)threadIdx.x) / 147) * 9))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 23))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 3))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 46))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 3))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 69))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 3))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 46))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 6))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 69))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 6))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 92))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 6))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 1))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 1))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 24))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 1))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 47))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 1))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 24))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 4))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 47))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 4))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 70))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 4))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 47))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 7))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 70))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 7))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 93))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 7))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 2))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 2))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 25))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 2))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 48))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 2))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 25))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 5))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 48))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 5))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 71))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 5))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 48))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 8))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 71))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 8))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 147) * 529) + (((((int)threadIdx.x) % 147) / 21) * 69)) + (((int)threadIdx.x) % 21)) + 94))] * kernel_shared[((((((int)threadIdx.x) / 147) * 9) + 8))]));
  for (int i_inner = 0; i_inner < 3; ++i_inner) {
    DepthwiseConv2d[(((((((int)blockIdx.x) * 1764) + ((((int)threadIdx.x) / 21) * 63)) + (i_inner * 21)) + (((int)threadIdx.x) % 21)))] = DepthwiseConv2d_local[(i_inner)];
  }
}

