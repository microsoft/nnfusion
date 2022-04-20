//127488_1_1_332_1_1
//128_96_165_165_7_2_SAME
//dim3 grid(127488, 1, 1);
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
  float DepthwiseConv2d_local[2];
  __shared__ float PaddedInput_shared[1368];
  __shared__ float kernel_shared[28];
  DepthwiseConv2d_local[0] = 0.000000e+00f;
  DepthwiseConv2d_local[1] = 0.000000e+00f;
  for (int di_outer_outer = 0; di_outer_outer < 7; ++di_outer_outer) {
    __syncthreads();
    PaddedInput_shared[(((int)threadIdx.x) * 2)] = (((((3 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && (((di_outer_outer >> 1) + (((int)blockIdx.x) % 83)) < 84)) && (3 <= ((((int)threadIdx.x) * 2) % 171))) && (((((int)threadIdx.x) * 2) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 1992) * 5227200) + (((((int)blockIdx.x) % 1992) / 83) * 108900)) + (((((int)threadIdx.x) * 2) / 171) * 27225)) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + ((((int)threadIdx.x) * 2) % 171)) - 498)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 2) + 1)] = (((((3 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && (((di_outer_outer >> 1) + (((int)blockIdx.x) % 83)) < 84)) && (3 <= (((((int)threadIdx.x) * 2) + 1) % 171))) && ((((((int)threadIdx.x) * 2) + 1) % 171) < 168)) ? data[((((((((((int)blockIdx.x) / 1992) * 5227200) + (((((int)blockIdx.x) % 1992) / 83) * 108900)) + ((((((int)threadIdx.x) * 2) + 1) / 171) * 27225)) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + (((((int)threadIdx.x) * 2) + 1) % 171)) - 498)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 2) + 664)] = (((((3 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && (((di_outer_outer >> 1) + (((int)blockIdx.x) % 83)) < 84)) && (3 <= (((((int)threadIdx.x) * 2) + 151) % 171))) && ((((((int)threadIdx.x) * 2) + 151) % 171) < 168)) ? data[(((((((((((int)blockIdx.x) / 1992) * 5227200) + (((((int)threadIdx.x) + 332) / 342) * 2613600)) + (((((int)blockIdx.x) % 1992) / 83) * 108900)) + (((((((int)threadIdx.x) * 2) + 664) % 684) / 171) * 27225)) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + (((((int)threadIdx.x) * 2) + 151) % 171)) - 498)] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) * 2) + 665)] = (((((3 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && (((di_outer_outer >> 1) + (((int)blockIdx.x) % 83)) < 84)) && (3 <= (((((int)threadIdx.x) * 2) + 152) % 171))) && ((((((int)threadIdx.x) * 2) + 152) % 171) < 168)) ? data[(((((((((((int)blockIdx.x) / 1992) * 5227200) + (((((int)threadIdx.x) + 332) / 342) * 2613600)) + (((((int)blockIdx.x) % 1992) / 83) * 108900)) + (((((((int)threadIdx.x) * 2) + 665) % 684) / 171) * 27225)) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + (((((int)threadIdx.x) * 2) + 152) % 171)) - 498)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 20) {
      PaddedInput_shared[((((int)threadIdx.x) * 2) + 1328)] = ((((3 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && (((di_outer_outer >> 1) + (((int)blockIdx.x) % 83)) < 84)) && (((int)threadIdx.x) < 19)) ? data[(((((((((((int)blockIdx.x) / 1992) * 5227200) + (((((int)threadIdx.x) + 664) / 342) * 2613600)) + (((((int)blockIdx.x) % 1992) / 83) * 108900)) + (((((((int)threadIdx.x) * 2) + 644) % 684) / 171) * 27225)) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + ((((int)threadIdx.x) * 2) + 131)) - 498)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 20) {
      PaddedInput_shared[((((int)threadIdx.x) * 2) + 1329)] = ((((3 <= (((((int)blockIdx.x) % 83) * 2) + di_outer_outer)) && (((di_outer_outer >> 1) + (((int)blockIdx.x) % 83)) < 84)) && (((int)threadIdx.x) < 18)) ? data[(((((((((((int)blockIdx.x) / 1992) * 5227200) + (((((int)threadIdx.x) + 664) / 342) * 2613600)) + (((((int)blockIdx.x) % 1992) / 83) * 108900)) + (((((((int)threadIdx.x) * 2) + 645) % 684) / 171) * 27225)) + ((((int)blockIdx.x) % 83) * 330)) + (di_outer_outer * 165)) + ((((int)threadIdx.x) * 2) + 132)) - 498)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 28) {
      kernel_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) % 1992) / 83) * 196) + ((((int)threadIdx.x) / 7) * 49)) + (di_outer_outer * 7)) + (((int)threadIdx.x) % 7))];
    }
    __syncthreads();
    DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared[(((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2))] * kernel_shared[(((((int)threadIdx.x) % 166) / 83) * 14)]));
    DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 1)]));
    DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 2)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 2)]));
    DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 3)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 3)]));
    DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 4)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 4)]));
    DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 5)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 5)]));
    DepthwiseConv2d_local[0] = (DepthwiseConv2d_local[0] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 6)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 6)]));
    DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 171)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 7)]));
    DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 172)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 8)]));
    DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 173)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 9)]));
    DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 174)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 10)]));
    DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 175)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 11)]));
    DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 176)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 12)]));
    DepthwiseConv2d_local[1] = (DepthwiseConv2d_local[1] + (PaddedInput_shared[((((((int)threadIdx.x) / 83) * 342) + ((((int)threadIdx.x) % 83) * 2)) + 177)] * kernel_shared[((((((int)threadIdx.x) % 166) / 83) * 14) + 13)]));
  }
  for (int c_inner = 0; c_inner < 2; ++c_inner) {
    DepthwiseConv2d[((((((((((int)blockIdx.x) / 1992) * 1322688) + ((((int)threadIdx.x) / 166) * 661344)) + (((((int)blockIdx.x) % 1992) / 83) * 27556)) + (((((int)threadIdx.x) % 166) / 83) * 13778)) + (c_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83))] = DepthwiseConv2d_local[c_inner];
  }
}

