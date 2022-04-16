//1_21_128_21_1_6
//128_84_42_42_84_1_1_VALID
//dim3 grid(1, 21, 128);
//dim3 block(21, 1, 6);

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
extern "C" __global__ void __launch_bounds__(126) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[56];
  __shared__ float pad_temp_shared[504];
  __shared__ float placeholder_shared[504];
  for (int ff_init = 0; ff_init < 14; ++ff_init) {
    for (int yy_init = 0; yy_init < 2; ++yy_init) {
      compute1[(((ff_init * 2) + yy_init))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 28))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 14; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 84) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((int)blockIdx.z) * 148176) + (rc_outer * 10584)) + (((int)threadIdx.z) * 1764)) + (((int)blockIdx.y) * 84)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[((((((int)threadIdx.z) * 84) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((int)threadIdx.z) * 1176) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 6) * 84)) + (rc_outer * 6)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 6)))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 6; ++rc_inner) {
      for (int ff = 0; ff < 14; ++ff) {
        for (int yy = 0; yy < 2; ++yy) {
          compute1[(((ff * 2) + yy))] = (compute1[(((ff * 2) + yy))] + (pad_temp_shared[((((rc_inner * 84) + (yy * 42)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 84) + (ff * 6)) + rc_inner))]));
          compute1[((((ff * 2) + yy) + 28))] = (compute1[((((ff * 2) + yy) + 28))] + (pad_temp_shared[(((((rc_inner * 84) + (yy * 42)) + ((int)threadIdx.x)) + 21))] * placeholder_shared[((((((int)threadIdx.z) * 84) + (ff * 6)) + rc_inner))]));
        }
      }
    }
  }
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 14; ++i1_inner_inner_inner) {
    for (int i2_inner_inner_inner = 0; i2_inner_inner_inner < 2; ++i2_inner_inner_inner) {
      compute[(((((((((int)blockIdx.z) * 148176) + (((int)threadIdx.z) * 24696)) + (i1_inner_inner_inner * 1764)) + (((int)blockIdx.y) * 84)) + (i2_inner_inner_inner * 42)) + ((int)threadIdx.x)))] = max((compute1[(((i1_inner_inner_inner * 2) + i2_inner_inner_inner))] + input2[(((((((((int)blockIdx.z) * 148176) + (((int)threadIdx.z) * 24696)) + (i1_inner_inner_inner * 1764)) + (((int)blockIdx.y) * 84)) + (i2_inner_inner_inner * 42)) + ((int)threadIdx.x)))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 148176) + (((int)threadIdx.z) * 24696)) + (i1_inner_inner_inner * 1764)) + (((int)blockIdx.y) * 84)) + (i2_inner_inner_inner * 42)) + ((int)threadIdx.x)) + 21))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 28))] + input2[((((((((((int)blockIdx.z) * 148176) + (((int)threadIdx.z) * 24696)) + (i1_inner_inner_inner * 1764)) + (((int)blockIdx.y) * 84)) + (i2_inner_inner_inner * 42)) + ((int)threadIdx.x)) + 21))]), 0.000000e+00f);
    }
  }
}

