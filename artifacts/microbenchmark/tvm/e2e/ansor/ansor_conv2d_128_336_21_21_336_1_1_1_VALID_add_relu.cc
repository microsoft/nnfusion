//1024_1_1_147_1_1
//128_336_21_21_336_1_1_VALID
//dim3 grid(1024, 1, 1);
//dim3 block(147, 1, 1);

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
extern "C" __global__ void __launch_bounds__(147) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[126];
  __shared__ float pad_temp_shared[9261];
  __shared__ float input1_shared[882];
  for (int yy_outer_inner_init = 0; yy_outer_inner_init < 3; ++yy_outer_inner_init) {
    for (int ff_inner_init = 0; ff_inner_init < 3; ++ff_inner_init) {
      for (int yy_inner_init = 0; yy_inner_init < 7; ++yy_inner_init) {
        compute1[((((ff_inner_init * 21) + (yy_outer_inner_init * 7)) + yy_inner_init))] = 0.000000e+00f;
        compute1[(((((ff_inner_init * 21) + (yy_outer_inner_init * 7)) + yy_inner_init) + 63))] = 0.000000e+00f;
      }
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 63; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 147) + ((int)threadIdx.x)))] = input0[((((((((int)blockIdx.x) >> 3) * 148176) + (rc_outer_outer * 9261)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 147)) + ((int)threadIdx.x)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      input1_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 147) + ((int)threadIdx.x)))] = input1[(((((((((int)blockIdx.x) & 7) * 14112) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 2352)) + ((((int)threadIdx.x) / 21) * 336)) + (rc_outer_outer * 21)) + (((int)threadIdx.x) % 21)))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 3; ++rc_outer_inner) {
      for (int yy_outer_inner = 0; yy_outer_inner < 3; ++yy_outer_inner) {
        for (int rc_inner = 0; rc_inner < 7; ++rc_inner) {
          for (int ff_inner = 0; ff_inner < 3; ++ff_inner) {
            for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
              compute1[((((ff_inner * 21) + (yy_outer_inner * 7)) + yy_inner))] = (compute1[((((ff_inner * 21) + (yy_outer_inner * 7)) + yy_inner))] + (pad_temp_shared[((((((rc_outer_inner * 3087) + (rc_inner * 441)) + (yy_outer_inner * 147)) + (yy_inner * 21)) + (((int)threadIdx.x) % 21)))] * input1_shared[((((((((int)threadIdx.x) / 21) * 63) + (ff_inner * 21)) + (rc_outer_inner * 7)) + rc_inner))]));
              compute1[(((((ff_inner * 21) + (yy_outer_inner * 7)) + yy_inner) + 63))] = (compute1[(((((ff_inner * 21) + (yy_outer_inner * 7)) + yy_inner) + 63))] + (pad_temp_shared[((((((rc_outer_inner * 3087) + (rc_inner * 441)) + (yy_outer_inner * 147)) + (yy_inner * 21)) + (((int)threadIdx.x) % 21)))] * input1_shared[(((((((((int)threadIdx.x) / 21) * 63) + (ff_inner * 21)) + (rc_outer_inner * 7)) + rc_inner) + 441))]));
            }
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 3; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 21; ++i2_inner) {
      compute[((((((((int)blockIdx.x) * 18522) + ((((int)threadIdx.x) / 21) * 1323)) + (i1_inner * 441)) + (i2_inner * 21)) + (((int)threadIdx.x) % 21)))] = max((compute1[(((i1_inner * 21) + i2_inner))] + input2[((((((((int)blockIdx.x) * 18522) + ((((int)threadIdx.x) / 21) * 1323)) + (i1_inner * 441)) + (i2_inner * 21)) + (((int)threadIdx.x) % 21)))]), 0.000000e+00f);
      compute[(((((((((int)blockIdx.x) * 18522) + ((((int)threadIdx.x) / 21) * 1323)) + (i1_inner * 441)) + (i2_inner * 21)) + (((int)threadIdx.x) % 21)) + 9261))] = max((compute1[((((i1_inner * 21) + i2_inner) + 63))] + input2[(((((((((int)blockIdx.x) * 18522) + ((((int)threadIdx.x) / 21) * 1323)) + (i1_inner * 441)) + (i2_inner * 21)) + (((int)threadIdx.x) % 21)) + 9261))]), 0.000000e+00f);
    }
  }
}

