//1_1_2048_11_11_1
//128_672_11_11_672_1_1_VALID
//dim3 grid(1, 1, 2048);
//dim3 block(11, 11, 1);

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
extern "C" __global__ void __launch_bounds__(121) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[42];
  __shared__ float pad_temp_shared[847];
  __shared__ float placeholder_shared[294];
  #pragma unroll
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    #pragma unroll
    for (int vthread_s = 0; vthread_s < 21; ++vthread_s) {
      compute[(((vthread_s * 2) + ff_init))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 96; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.y) * 77) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((int)blockIdx.z) >> 4) * 81312) + (rc_outer * 847)) + (((int)threadIdx.y) * 77)) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if ((((((int)threadIdx.y) * 27) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 294) {
        if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 27) {
          placeholder_shared[((((((int)threadIdx.y) * 27) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) & 15) * 28224) + (((((((int)threadIdx.y) * 27) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 7) * 672)) + (rc_outer * 7)) + ((((((int)threadIdx.y) * 27) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 7)))];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 7; ++rc_inner) {
      #pragma unroll
      for (int ff = 0; ff < 2; ++ff) {
        #pragma unroll
        for (int vthread_s1 = 0; vthread_s1 < 21; ++vthread_s1) {
          compute[(((vthread_s1 * 2) + ff))] = (compute[(((vthread_s1 * 2) + ff))] + (pad_temp_shared[((((rc_inner * 121) + (((int)threadIdx.y) * 11)) + ((int)threadIdx.x)))] * placeholder_shared[((((vthread_s1 * 14) + (ff * 7)) + rc_inner))]));
        }
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int vthread_s2 = 0; vthread_s2 < 21; ++vthread_s2) {
      T_add[((((((((int)blockIdx.z) * 5082) + (vthread_s2 * 242)) + (ax1_inner_inner_inner * 121)) + (((int)threadIdx.y) * 11)) + ((int)threadIdx.x)))] = (compute[(((vthread_s2 * 2) + ax1_inner_inner_inner))] + input2[((((((((int)blockIdx.z) * 5082) + (vthread_s2 * 242)) + (ax1_inner_inner_inner * 121)) + (((int)threadIdx.y) * 11)) + ((int)threadIdx.x)))]);
    }
  }
}

