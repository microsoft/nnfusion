//13944_1_1_332_1_1
//128_42_83_83_7_1_SAME
//dim3 grid(13944, 1, 1);
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
  float DepthwiseConv2d_local[8];
  __shared__ float PaddedInput_shared[2848];
  __shared__ float kernel_shared[7];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(7)] = 0.000000e+00f;
  for (int di_outer_outer = 0; di_outer_outer < 7; ++di_outer_outer) {
    __syncthreads();
    PaddedInput_shared[(((int)threadIdx.x))] = (((((3 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 86)) && (3 <= (((int)threadIdx.x) % 89))) && ((((int)threadIdx.x) % 89) < 86)) ? data[((((((((((int)blockIdx.x) / 3486) * 9258816) + ((((int)threadIdx.x) / 89) * 289338)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 3486) * 83)) + (((int)threadIdx.x) % 89)) - 252))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 332))] = (((((3 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 86)) && (3 <= ((((int)threadIdx.x) + 65) % 89))) && (((((int)threadIdx.x) + 65) % 89) < 86)) ? data[((((((((((int)blockIdx.x) / 3486) * 9258816) + (((((int)threadIdx.x) + 332) / 89) * 289338)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 3486) * 83)) + ((((int)threadIdx.x) + 65) % 89)) - 252))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 664))] = (((((3 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 86)) && (3 <= ((((int)threadIdx.x) + 41) % 89))) && (((((int)threadIdx.x) + 41) % 89) < 86)) ? data[((((((((((int)blockIdx.x) / 3486) * 9258816) + (((((int)threadIdx.x) + 664) / 89) * 289338)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 3486) * 83)) + ((((int)threadIdx.x) + 41) % 89)) - 252))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 996))] = (((((3 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 86)) && (3 <= ((((int)threadIdx.x) + 17) % 89))) && (((((int)threadIdx.x) + 17) % 89) < 86)) ? data[((((((((((int)blockIdx.x) / 3486) * 9258816) + (((((int)threadIdx.x) + 996) / 89) * 289338)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 3486) * 83)) + ((((int)threadIdx.x) + 17) % 89)) - 252))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 1328))] = (((((3 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 86)) && (3 <= ((((int)threadIdx.x) + 82) % 89))) && (((((int)threadIdx.x) + 82) % 89) < 86)) ? data[((((((((((int)blockIdx.x) / 3486) * 9258816) + (((((int)threadIdx.x) + 1328) / 89) * 289338)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 3486) * 83)) + ((((int)threadIdx.x) + 82) % 89)) - 252))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 1660))] = (((((3 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 86)) && (3 <= ((((int)threadIdx.x) + 58) % 89))) && (((((int)threadIdx.x) + 58) % 89) < 86)) ? data[((((((((((int)blockIdx.x) / 3486) * 9258816) + (((((int)threadIdx.x) + 1660) / 89) * 289338)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 3486) * 83)) + ((((int)threadIdx.x) + 58) % 89)) - 252))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 1992))] = (((((3 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 86)) && (3 <= ((((int)threadIdx.x) + 34) % 89))) && (((((int)threadIdx.x) + 34) % 89) < 86)) ? data[((((((((((int)blockIdx.x) / 3486) * 9258816) + (((((int)threadIdx.x) + 1992) / 89) * 289338)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 3486) * 83)) + ((((int)threadIdx.x) + 34) % 89)) - 252))] : 0.000000e+00f);
    PaddedInput_shared[((((int)threadIdx.x) + 2324))] = (((((3 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 86)) && (3 <= ((((int)threadIdx.x) + 10) % 89))) && (((((int)threadIdx.x) + 10) % 89) < 86)) ? data[((((((((((int)blockIdx.x) / 3486) * 9258816) + (((((int)threadIdx.x) + 2324) / 89) * 289338)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 3486) * 83)) + ((((int)threadIdx.x) + 10) % 89)) - 252))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 192) {
      PaddedInput_shared[((((int)threadIdx.x) + 2656))] = (((((3 <= (di_outer_outer + (((int)blockIdx.x) % 83))) && ((di_outer_outer + (((int)blockIdx.x) % 83)) < 86)) && (3 <= ((((int)threadIdx.x) + 75) % 89))) && (((((int)threadIdx.x) + 75) % 89) < 86)) ? data[((((((((((int)blockIdx.x) / 3486) * 9258816) + (((((int)threadIdx.x) + 2656) / 89) * 289338)) + (di_outer_outer * 83)) + ((((int)blockIdx.x) % 3486) * 83)) + ((((int)threadIdx.x) + 75) % 89)) - 252))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 7) {
      kernel_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) % 3486) / 83) * 49) + (di_outer_outer * 7)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int b_c_outer_inner = 0; b_c_outer_inner < 2; ++b_c_outer_inner) {
      for (int dj_inner = 0; dj_inner < 7; ++dj_inner) {
        DepthwiseConv2d_local[(b_c_outer_inner)] = (DepthwiseConv2d_local[(b_c_outer_inner)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 83) * 178) + (b_c_outer_inner * 89)) + dj_inner) + (((int)threadIdx.x) % 83)))] * kernel_shared[(dj_inner)]));
        DepthwiseConv2d_local[((b_c_outer_inner + 2))] = (DepthwiseConv2d_local[((b_c_outer_inner + 2))] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 83) * 178) + (b_c_outer_inner * 89)) + dj_inner) + (((int)threadIdx.x) % 83)) + 712))] * kernel_shared[(dj_inner)]));
        DepthwiseConv2d_local[((b_c_outer_inner + 4))] = (DepthwiseConv2d_local[((b_c_outer_inner + 4))] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 83) * 178) + (b_c_outer_inner * 89)) + dj_inner) + (((int)threadIdx.x) % 83)) + 1424))] * kernel_shared[(dj_inner)]));
        DepthwiseConv2d_local[((b_c_outer_inner + 6))] = (DepthwiseConv2d_local[((b_c_outer_inner + 6))] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 83) * 178) + (b_c_outer_inner * 89)) + dj_inner) + (((int)threadIdx.x) % 83)) + 2136))] * kernel_shared[(dj_inner)]));
      }
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    DepthwiseConv2d[(((((((((int)blockIdx.x) / 3486) * 9258816) + ((((int)threadIdx.x) / 83) * 578676)) + (b_inner * 289338)) + ((((int)blockIdx.x) % 3486) * 83)) + (((int)threadIdx.x) % 83)))] = DepthwiseConv2d_local[(b_inner)];
    DepthwiseConv2d[((((((((((int)blockIdx.x) / 3486) * 9258816) + ((((int)threadIdx.x) / 83) * 578676)) + (b_inner * 289338)) + ((((int)blockIdx.x) % 3486) * 83)) + (((int)threadIdx.x) % 83)) + 2314704))] = DepthwiseConv2d_local[((b_inner + 2))];
    DepthwiseConv2d[((((((((((int)blockIdx.x) / 3486) * 9258816) + ((((int)threadIdx.x) / 83) * 578676)) + (b_inner * 289338)) + ((((int)blockIdx.x) % 3486) * 83)) + (((int)threadIdx.x) % 83)) + 4629408))] = DepthwiseConv2d_local[((b_inner + 4))];
    DepthwiseConv2d[((((((((((int)blockIdx.x) / 3486) * 9258816) + ((((int)threadIdx.x) / 83) * 578676)) + (b_inner * 289338)) + ((((int)blockIdx.x) % 3486) * 83)) + (((int)threadIdx.x) % 83)) + 6944112))] = DepthwiseConv2d_local[((b_inner + 6))];
  }
}

