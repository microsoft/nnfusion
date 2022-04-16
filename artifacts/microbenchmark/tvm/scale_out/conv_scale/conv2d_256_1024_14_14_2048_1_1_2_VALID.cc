//1_1_2048_7_1_64
//256_1024_14_14_2048_1_2_VALID
//dim3 grid(1, 1, 2048);
//dim3 block(7, 1, 64);

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
extern "C" __global__ void __launch_bounds__(448) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[28];
  __shared__ float pad_temp_shared[1352];
  __shared__ float placeholder_shared[2048];
  #pragma unroll
  for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
    #pragma unroll
    for (int yy_c_init = 0; yy_c_init < 7; ++yy_c_init) {
      compute_local[(((ff_c_init * 7) + yy_c_init))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 7) + yy_c_init) + 14))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 1352) {
        if (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 22) {
          pad_temp_shared[((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 200704) + (rc_outer * 1568)) + (((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 169) * 196)) + ((((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 169) / 13) * 14)) + ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 13)))];
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3)) < 256) {
        if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 2048) {
          if (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 32) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 262144) + (((int)threadIdx.z) * 4096)) + ((((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3) * 1024)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      #pragma unroll
      for (int ff_c = 0; ff_c < 2; ++ff_c) {
        #pragma unroll
        for (int yy_c = 0; yy_c < 7; ++yy_c) {
          compute_local[(((ff_c * 7) + yy_c))] = (compute_local[(((ff_c * 7) + yy_c))] + (pad_temp_shared[((((rc_inner * 169) + (yy_c * 26)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner))]));
          compute_local[((((ff_c * 7) + yy_c) + 14))] = (compute_local[((((ff_c * 7) + yy_c) + 14))] + (pad_temp_shared[((((rc_inner * 169) + (yy_c * 26)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner) + 1024))]));
        }
      }
    }
  }
  #pragma unroll
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 2; ++ff_inner_inner_inner) {
    #pragma unroll
    for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 7; ++yy_inner_inner_inner) {
      compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 98)) + (ff_inner_inner_inner * 49)) + (yy_inner_inner_inner * 7)) + ((int)threadIdx.x)))] = compute_local[(((ff_inner_inner_inner * 7) + yy_inner_inner_inner))];
      compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 98)) + (ff_inner_inner_inner * 49)) + (yy_inner_inner_inner * 7)) + ((int)threadIdx.x)) + 6272))] = compute_local[((((ff_inner_inner_inner * 7) + yy_inner_inner_inner) + 14))];
    }
  }
}

