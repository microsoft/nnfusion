//1536_1_1_252_1_1
//128_1008_42_42_168_1_1_SAME
//dim3 grid(1536, 1, 1);
//dim3 block(252, 1, 1);

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
extern "C" __global__ void __launch_bounds__(252) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[98];
  __shared__ float pad_temp_shared[5292];
  __shared__ float input1_shared[1512];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 7; ++ff_outer_inner_init) {
    for (int yy_outer_inner_init = 0; yy_outer_inner_init < 7; ++yy_outer_inner_init) {
      compute[(((ff_outer_inner_init * 7) + yy_outer_inner_init))] = 0.000000e+00f;
      compute[((((ff_outer_inner_init * 7) + yy_outer_inner_init) + 49))] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 56; ++rc_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 21; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
      pad_temp_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 252) + ((int)threadIdx.x)))] = input0[((((((((((int)blockIdx.x) / 12) * 1778112) + (rc_outer_outer * 31752)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 6) + (((int)threadIdx.x) / 42)) / 7) * 1764)) + ((((int)blockIdx.x) % 6) * 294)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 6) + (((int)threadIdx.x) / 42)) % 7) * 42)) + (((int)threadIdx.x) % 42)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
      ((float2*)(input1_shared + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 504) + (((int)threadIdx.x) * 2)))))[0] = ((float2*)(input1 + ((((((((((int)blockIdx.x) % 12) / 6) * 84672) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 28224)) + ((((int)threadIdx.x) / 9) * 1008)) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) % 9) * 2)))))[0];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 18; ++rc_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 7; ++ff_outer_inner) {
        for (int yy_outer_inner = 0; yy_outer_inner < 7; ++yy_outer_inner) {
          compute[(((ff_outer_inner * 7) + yy_outer_inner))] = (compute[(((ff_outer_inner * 7) + yy_outer_inner))] + (pad_temp_shared[((((rc_outer_inner * 294) + (yy_outer_inner * 42)) + (((int)threadIdx.x) % 42)))] * input1_shared[(((((((int)threadIdx.x) / 42) * 126) + (ff_outer_inner * 18)) + rc_outer_inner))]));
          compute[((((ff_outer_inner * 7) + yy_outer_inner) + 49))] = (compute[((((ff_outer_inner * 7) + yy_outer_inner) + 49))] + (pad_temp_shared[((((rc_outer_inner * 294) + (yy_outer_inner * 42)) + (((int)threadIdx.x) % 42)))] * input1_shared[((((((((int)threadIdx.x) / 42) * 126) + (ff_outer_inner * 18)) + rc_outer_inner) + 756))]));
        }
      }
    }
  }
  for (int ax1_inner = 0; ax1_inner < 7; ++ax1_inner) {
    for (int ax2_inner = 0; ax2_inner < 7; ++ax2_inner) {
      T_add[((((((((((int)blockIdx.x) / 6) * 148176) + ((((int)threadIdx.x) / 42) * 12348)) + (ax1_inner * 1764)) + ((((int)blockIdx.x) % 6) * 294)) + (ax2_inner * 42)) + (((int)threadIdx.x) % 42)))] = (compute[(((ax1_inner * 7) + ax2_inner))] + input2[((((((((((int)blockIdx.x) / 6) * 148176) + ((((int)threadIdx.x) / 42) * 12348)) + (ax1_inner * 1764)) + ((((int)blockIdx.x) % 6) * 294)) + (ax2_inner * 42)) + (((int)threadIdx.x) % 42)))]);
      T_add[(((((((((((int)blockIdx.x) / 6) * 148176) + ((((int)threadIdx.x) / 42) * 12348)) + (ax1_inner * 1764)) + ((((int)blockIdx.x) % 6) * 294)) + (ax2_inner * 42)) + (((int)threadIdx.x) % 42)) + 74088))] = (compute[((((ax1_inner * 7) + ax2_inner) + 49))] + input2[(((((((((((int)blockIdx.x) / 6) * 148176) + ((((int)threadIdx.x) / 42) * 12348)) + (ax1_inner * 1764)) + ((((int)blockIdx.x) % 6) * 294)) + (ax2_inner * 42)) + (((int)threadIdx.x) % 42)) + 74088))]);
    }
  }
}

