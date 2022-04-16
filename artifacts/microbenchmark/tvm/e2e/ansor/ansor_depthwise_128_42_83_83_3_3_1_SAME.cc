//74368_1_1_83_1_1
//128_42_83_83_3_1_SAME
//dim3 grid(74368, 1, 1);
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
  float DepthwiseConv2d_local[6];
  __shared__ float PaddedInput_shared[510];
  __shared__ float kernel_shared[18];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  for (int di_outer_outer = 0; di_outer_outer < 3; ++di_outer_outer) {
    __syncthreads();
    PaddedInput_shared[(((int)threadIdx.x))] = ((((1 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 84)) && (1 <= ((int)threadIdx.x))) ? data[(((((((((int)blockIdx.x) / 83) * 41334) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + ((int)threadIdx.x)) - 84))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 83))] = (((((1 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 84)) && (1 <= ((((int)threadIdx.x) + 83) % 85))) && (((((int)threadIdx.x) + 83) % 85) < 84)) ? data[((((((((((int)blockIdx.x) / 83) * 41334) + (((((int)threadIdx.x) + 83) / 85) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + ((((int)threadIdx.x) + 83) % 85)) - 84))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 166))] = (((((1 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 84)) && (1 <= ((((int)threadIdx.x) + 81) % 85))) && (((((int)threadIdx.x) + 81) % 85) < 84)) ? data[((((((((((int)blockIdx.x) / 83) * 41334) + (((((int)threadIdx.x) + 166) / 85) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + ((((int)threadIdx.x) + 81) % 85)) - 84))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 249))] = (((((1 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 84)) && (1 <= ((((int)threadIdx.x) + 79) % 85))) && (((((int)threadIdx.x) + 79) % 85) < 84)) ? data[((((((((((int)blockIdx.x) / 83) * 41334) + (((((int)threadIdx.x) + 249) / 85) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + ((((int)threadIdx.x) + 79) % 85)) - 84))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 332))] = (((((1 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 84)) && (1 <= ((((int)threadIdx.x) + 77) % 85))) && (((((int)threadIdx.x) + 77) % 85) < 84)) ? data[((((((((((int)blockIdx.x) / 83) * 41334) + (((((int)threadIdx.x) + 332) / 85) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + ((((int)threadIdx.x) + 77) % 85)) - 84))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 415))] = (((((1 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 84)) && (1 <= ((((int)threadIdx.x) + 75) % 85))) && (((((int)threadIdx.x) + 75) % 85) < 84)) ? data[((((((((((int)blockIdx.x) / 83) * 41334) + (((((int)threadIdx.x) + 415) / 85) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + ((((int)threadIdx.x) + 75) % 85)) - 84))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 12) {
      PaddedInput_shared[((((int)threadIdx.x) + 498))] = ((((1 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 84)) && (((int)threadIdx.x) < 11)) ? data[((((((((((int)blockIdx.x) / 83) * 41334) + (((((int)threadIdx.x) + 498) / 85) * 6889)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) + 73)) - 84))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 6) {
      ((float3*)(kernel_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(kernel + ((((((((int)blockIdx.x) % 581) / 83) * 54) + (((int)threadIdx.x) * 9)) + (di_outer_outer * 3)))))[0];
    }
    __syncthreads();
    for (int dj_outer_inner = 0; dj_outer_inner < 3; ++dj_outer_inner) {
      DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((int)threadIdx.x) + dj_outer_inner))] * kernel_shared[(dj_outer_inner)]));
      DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((int)threadIdx.x) + dj_outer_inner) + 85))] * kernel_shared[((dj_outer_inner + 3))]));
      DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[(((((int)threadIdx.x) + dj_outer_inner) + 170))] * kernel_shared[((dj_outer_inner + 6))]));
      DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[(((((int)threadIdx.x) + dj_outer_inner) + 255))] * kernel_shared[((dj_outer_inner + 9))]));
      DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[(((((int)threadIdx.x) + dj_outer_inner) + 340))] * kernel_shared[((dj_outer_inner + 12))]));
      DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[(((((int)threadIdx.x) + dj_outer_inner) + 425))] * kernel_shared[((dj_outer_inner + 15))]));
    }
  }
  DepthwiseConv2d[(((((((int)blockIdx.x) / 83) * 41334) + ((((int)blockIdx.x) % 83) * 83)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 83) * 41334) + ((((int)blockIdx.x) % 83) * 83)) + ((int)threadIdx.x)) + 6889))] = DepthwiseConv2d_local[(1)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 83) * 41334) + ((((int)blockIdx.x) % 83) * 83)) + ((int)threadIdx.x)) + 13778))] = DepthwiseConv2d_local[(2)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 83) * 41334) + ((((int)blockIdx.x) % 83) * 83)) + ((int)threadIdx.x)) + 20667))] = DepthwiseConv2d_local[(3)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 83) * 41334) + ((((int)blockIdx.x) % 83) * 83)) + ((int)threadIdx.x)) + 27556))] = DepthwiseConv2d_local[(4)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 83) * 41334) + ((((int)blockIdx.x) % 83) * 83)) + ((int)threadIdx.x)) + 34445))] = DepthwiseConv2d_local[(5)];
}

