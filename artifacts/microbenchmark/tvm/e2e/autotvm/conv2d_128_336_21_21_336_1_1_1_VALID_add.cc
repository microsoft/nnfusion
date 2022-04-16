//1_7_1024_21_1_3
//128_336_21_21_336_1_1_VALID
//dim3 grid(1, 7, 1024);
//dim3 block(21, 1, 3);

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
extern "C" __global__ void __launch_bounds__(63) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[42];
  __shared__ float pad_temp_shared[252];
  __shared__ float placeholder_shared[168];
  #pragma unroll
  for (int ff_init = 0; ff_init < 7; ++ff_init) {
    compute[(ff_init)] = 0.000000e+00f;
    compute[((ff_init + 21))] = 0.000000e+00f;
    compute[((ff_init + 7))] = 0.000000e+00f;
    compute[((ff_init + 28))] = 0.000000e+00f;
    compute[((ff_init + 14))] = 0.000000e+00f;
    compute[((ff_init + 35))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 84; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 84) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((((((int)blockIdx.z) >> 3) * 148176) + (rc_outer * 1764)) + ((((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 21)) / 3) * 441)) + (((int)blockIdx.y) * 63)) + ((((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 21)) % 3) * 21)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 21)))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 14) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 2)) < 42) {
        if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 168) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 56) {
            placeholder_shared[((((((int)threadIdx.z) * 56) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 14112) + (((int)threadIdx.z) * 4704)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 2) * 336)) + (rc_outer * 4)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 3)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
      #pragma unroll
      for (int ff = 0; ff < 7; ++ff) {
        compute[(ff)] = (compute[(ff)] + (pad_temp_shared[(((rc_inner * 63) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 28) + (ff * 4)) + rc_inner))]));
        compute[((ff + 21))] = (compute[((ff + 21))] + (pad_temp_shared[(((rc_inner * 63) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 28) + (ff * 4)) + rc_inner) + 84))]));
        compute[((ff + 7))] = (compute[((ff + 7))] + (pad_temp_shared[((((rc_inner * 63) + ((int)threadIdx.x)) + 21))] * placeholder_shared[((((((int)threadIdx.z) * 28) + (ff * 4)) + rc_inner))]));
        compute[((ff + 28))] = (compute[((ff + 28))] + (pad_temp_shared[((((rc_inner * 63) + ((int)threadIdx.x)) + 21))] * placeholder_shared[(((((((int)threadIdx.z) * 28) + (ff * 4)) + rc_inner) + 84))]));
        compute[((ff + 14))] = (compute[((ff + 14))] + (pad_temp_shared[((((rc_inner * 63) + ((int)threadIdx.x)) + 42))] * placeholder_shared[((((((int)threadIdx.z) * 28) + (ff * 4)) + rc_inner))]));
        compute[((ff + 35))] = (compute[((ff + 35))] + (pad_temp_shared[((((rc_inner * 63) + ((int)threadIdx.x)) + 42))] * placeholder_shared[(((((((int)threadIdx.z) * 28) + (ff * 4)) + rc_inner) + 84))]));
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 7; ++ax1_inner_inner_inner) {
    T_add[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)))] = (compute[(ax1_inner_inner_inner)] + input2[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)))]);
    T_add[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 9261))] = (compute[((ax1_inner_inner_inner + 21))] + input2[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 9261))]);
    T_add[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 21))] = (compute[((ax1_inner_inner_inner + 7))] + input2[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 21))]);
    T_add[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 9282))] = (compute[((ax1_inner_inner_inner + 28))] + input2[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 9282))]);
    T_add[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 42))] = (compute[((ax1_inner_inner_inner + 14))] + input2[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 42))]);
    T_add[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 9303))] = (compute[((ax1_inner_inner_inner + 35))] + input2[(((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 3087)) + (ax1_inner_inner_inner * 441)) + (((int)blockIdx.y) * 63)) + ((int)threadIdx.x)) + 9303))]);
  }
}

