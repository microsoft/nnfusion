//1_1_2048_7_1_32
//128_1024_14_14_2048_1_2_VALID
//dim3 grid(1, 1, 2048);
//dim3 block(7, 1, 32);

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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[28];
  __shared__ float pad_temp_shared[1352];
  __shared__ float placeholder_shared[1024];
  #pragma unroll
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 7; ++yy_init) {
      compute[(((ff_init * 7) + yy_init))] = 0.000000e+00f;
      compute[((((ff_init * 7) + yy_init) + 14))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 43) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 1352) {
        if (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 43) {
          pad_temp_shared[((((((int)threadIdx.z) * 43) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((int)blockIdx.z) >> 4) * 200704) + (rc_outer * 1568)) + (((((((int)threadIdx.z) * 43) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 169) * 196)) + ((((((((int)threadIdx.z) * 43) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 169) / 13) * 14)) + ((((((int)threadIdx.z) * 43) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 13)))];
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3)) < 128) {
        if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 32) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) & 15) * 131072) + (((int)threadIdx.z) * 4096)) + ((((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3) * 1024)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      #pragma unroll
      for (int ff = 0; ff < 2; ++ff) {
        #pragma unroll
        for (int yy = 0; yy < 7; ++yy) {
          compute[(((ff * 7) + yy))] = (compute[(((ff * 7) + yy))] + (pad_temp_shared[((((rc_inner * 169) + (yy * 26)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner))]));
          compute[((((ff * 7) + yy) + 14))] = (compute[((((ff * 7) + yy) + 14))] + (pad_temp_shared[((((rc_inner * 169) + (yy * 26)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 512))]));
        }
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
      T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 98)) + (ax1_inner_inner_inner * 49)) + (ax2_inner_inner_inner * 7)) + ((int)threadIdx.x)))] = (compute[(((ax1_inner_inner_inner * 7) + ax2_inner_inner_inner))] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 98)) + (ax1_inner_inner_inner * 49)) + (ax2_inner_inner_inner * 7)) + ((int)threadIdx.x)))]);
      T_add[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 98)) + (ax1_inner_inner_inner * 49)) + (ax2_inner_inner_inner * 7)) + ((int)threadIdx.x)) + 3136))] = (compute[((((ax1_inner_inner_inner * 7) + ax2_inner_inner_inner) + 14))] + input2[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 98)) + (ax1_inner_inner_inner * 49)) + (ax2_inner_inner_inner * 7)) + ((int)threadIdx.x)) + 3136))]);
    }
  }
}

