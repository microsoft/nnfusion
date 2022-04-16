//7_4_256_8_1_16
//128_64_56_56_64_3_1_SAME
//dim3 grid(7, 4, 256);
//dim3 block(8, 1, 16);

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
extern "C" __global__ void __launch_bounds__(128) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[28];
  __shared__ float pad_temp_shared[320];
  __shared__ float placeholder_shared[576];
  #pragma unroll
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 14; ++yy_init) {
      compute1[(((ff_init * 14) + yy_init))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 10)) < 32) {
        if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 320) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 20) {
            pad_temp_shared[((((((int)threadIdx.z) * 20) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 10)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 10)) & 15)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 10)))) && (((((int)blockIdx.x) * 8) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 10)) < 57)) ? placeholder[((((((((((((int)blockIdx.z) >> 1) * 200704) + (rc_outer * 6272)) + ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 10)) >> 4) * 3136)) + (((int)blockIdx.y) * 784)) + ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 10)) & 15) * 56)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 10)) - 57))] : 0.000000e+00f);
          }
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 18)) < 32) {
        if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 9)) < 64) {
          if (((((int)threadIdx.z) * 12) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 3)) < 192) {
            if ((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 576) {
              if (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 36) {
                placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) & 1) * 18432) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 18) * 576)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 18)))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
      #pragma unroll
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        #pragma unroll
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          #pragma unroll
          for (int ff = 0; ff < 2; ++ff) {
            #pragma unroll
            for (int yy = 0; yy < 14; ++yy) {
              compute1[(((ff * 14) + yy))] = (compute1[(((ff * 14) + yy))] + (pad_temp_shared[((((((rc_inner * 160) + (yy * 10)) + (ry_inner * 10)) + ((int)threadIdx.x)) + rx_inner))] * placeholder_shared[((((((((int)threadIdx.z) * 36) + (ff * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner))]));
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 2; ++i1_inner_inner_inner) {
    #pragma unroll
    for (int i2_inner_inner_inner = 0; i2_inner_inner_inner < 14; ++i2_inner_inner_inner) {
      compute[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 784)) + (i2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))] = max((compute1[(((i1_inner_inner_inner * 14) + i2_inner_inner_inner))] + input2[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 784)) + (i2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))]), 0.000000e+00f);
    }
  }
}

