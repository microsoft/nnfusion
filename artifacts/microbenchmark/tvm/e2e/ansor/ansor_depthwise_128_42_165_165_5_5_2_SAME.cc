//223104_1_1_83_1_1
//128_42_165_165_5_2_SAME
//dim3 grid(223104, 1, 1);
//dim3 block(83, 1, 1);

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
extern "C" __global__ void __launch_bounds__(83) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[2];
  __shared__ float PaddedInput_shared[338];
  __shared__ float kernel_shared[10];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  for (int di_outer_outer = 0; di_outer_outer < 5; ++di_outer_outer) {
    __syncthreads();
    PaddedInput_shared[(((int)threadIdx.x))] = ((((2 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 83) * 2) + di_outer_outer) < 167)) && (2 <= ((int)threadIdx.x))) ? data[(((((((((int)blockIdx.x) / 83) * 54450) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + ((int)threadIdx.x)) - 332))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 83))] = (((2 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 83) * 2) + di_outer_outer) < 167)) ? data[(((((((((int)blockIdx.x) / 83) * 54450) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + ((int)threadIdx.x)) - 249))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 166))] = (((((2 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 83) * 2) + di_outer_outer) < 167)) && (2 <= ((((int)threadIdx.x) + 166) % 169))) && (((((int)threadIdx.x) + 166) % 169) < 167)) ? data[((((((((((int)blockIdx.x) / 83) * 54450) + (((((int)threadIdx.x) + 166) / 169) * 27225)) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + ((((int)threadIdx.x) + 166) % 169)) - 332))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 249))] = (((2 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 83) * 2) + di_outer_outer) < 167)) ? data[((((((((((int)blockIdx.x) / 83) * 54450) + (((((int)threadIdx.x) + 249) / 169) * 27225)) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + (((int)threadIdx.x) + 80)) - 332))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 6) {
      PaddedInput_shared[((((int)threadIdx.x) + 332))] = ((((2 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && ((((((int)blockIdx.x) % 83) * 2) + di_outer_outer) < 167)) && (((int)threadIdx.x) < 4)) ? data[((((((((((int)blockIdx.x) / 83) * 54450) + (((((int)threadIdx.x) + 332) / 169) * 27225)) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + (((int)threadIdx.x) + 163)) - 332))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 10) {
      kernel_shared[(((int)threadIdx.x))] = kernel[(((((((((int)blockIdx.x) % 1743) / 83) * 50) + ((((int)threadIdx.x) / 5) * 25)) + (di_outer_outer * 5)) + (((int)threadIdx.x) % 5)))];
    }
    __syncthreads();
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((int)threadIdx.x) * 2))] * kernel_shared[(0)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((int)threadIdx.x) * 2) + 169))] * kernel_shared[(5)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1))] * kernel_shared[(1)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((int)threadIdx.x) * 2) + 170))] * kernel_shared[(6)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((int)threadIdx.x) * 2) + 2))] * kernel_shared[(2)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((int)threadIdx.x) * 2) + 171))] * kernel_shared[(7)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((int)threadIdx.x) * 2) + 3))] * kernel_shared[(3)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((int)threadIdx.x) * 2) + 172))] * kernel_shared[(8)]));
    DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((int)threadIdx.x) * 2) + 4))] * kernel_shared[(4)]));
    DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((int)threadIdx.x) * 2) + 173))] * kernel_shared[(9)]));
  }
  DepthwiseConv2d[(((((((int)blockIdx.x) / 83) * 13778) + ((((int)blockIdx.x) % 83) * 83)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 83) * 13778) + ((((int)blockIdx.x) % 83) * 83)) + ((int)threadIdx.x)) + 6889))] = DepthwiseConv2d_local[(1)];
}

