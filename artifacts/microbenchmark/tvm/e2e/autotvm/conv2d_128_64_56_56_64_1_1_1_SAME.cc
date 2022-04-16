//1_28_128_56_1_4
//128_64_56_56_64_1_1_SAME
//dim3 grid(1, 28, 128);
//dim3 block(56, 1, 4);

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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[32];
  __shared__ float pad_temp_shared[448];
  __shared__ float placeholder_shared[256];
  #pragma unroll
  for (int yy_c_init = 0; yy_c_init < 2; ++yy_c_init) {
    #pragma unroll
    for (int vthread_s = 0; vthread_s < 16; ++vthread_s) {
      compute_local[(((vthread_s * 2) + yy_c_init))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((int)blockIdx.z) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 2)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 256) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((int)threadIdx.z) * 1024) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 2) * 64)) + (rc_outer * 4)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 3)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
      #pragma unroll
      for (int yy_c = 0; yy_c < 2; ++yy_c) {
        #pragma unroll
        for (int vthread_s1 = 0; vthread_s1 < 16; ++vthread_s1) {
          compute_local[(((vthread_s1 * 2) + yy_c))] = (compute_local[(((vthread_s1 * 2) + yy_c))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 56)) + ((int)threadIdx.x)))] * placeholder_shared[((((vthread_s1 * 16) + (((int)threadIdx.z) * 4)) + rc_inner))]));
        }
      }
    }
  }
  #pragma unroll
  for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 2; ++yy_inner_inner_inner) {
    #pragma unroll
    for (int vthread_s2 = 0; vthread_s2 < 16; ++vthread_s2) {
      compute[(((((((((int)blockIdx.z) * 200704) + (vthread_s2 * 12544)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 56)) + ((int)threadIdx.x)))] = compute_local[(((vthread_s2 * 2) + yy_inner_inner_inner))];
    }
  }
}

