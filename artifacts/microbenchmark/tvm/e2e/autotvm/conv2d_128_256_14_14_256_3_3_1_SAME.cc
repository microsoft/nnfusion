//2_2_1024_4_4_8
//128_256_14_14_256_3_1_SAME
//dim3 grid(2, 2, 1024);
//dim3 block(4, 4, 8);

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
extern "C" __global__ void __launch_bounds__(128) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[36];
  __shared__ float pad_temp_shared[392];
  __shared__ float placeholder_shared[576];
  #pragma unroll
  for (int xx_c_init = 0; xx_c_init < 3; ++xx_c_init) {
    compute_local[(xx_c_init)] = 0.000000e+00f;
    compute_local[((xx_c_init + 9))] = 0.000000e+00f;
    compute_local[((xx_c_init + 18))] = 0.000000e+00f;
    compute_local[((xx_c_init + 27))] = 0.000000e+00f;
    compute_local[((xx_c_init + 3))] = 0.000000e+00f;
    compute_local[((xx_c_init + 12))] = 0.000000e+00f;
    compute_local[((xx_c_init + 21))] = 0.000000e+00f;
    compute_local[((xx_c_init + 30))] = 0.000000e+00f;
    compute_local[((xx_c_init + 6))] = 0.000000e+00f;
    compute_local[((xx_c_init + 15))] = 0.000000e+00f;
    compute_local[((xx_c_init + 24))] = 0.000000e+00f;
    compute_local[((xx_c_init + 33))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 392) {
        if ((((((int)threadIdx.y) * 13) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 49) {
          if (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 13) {
            if (((((int)blockIdx.y) * 12) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 196) / 14)) < 16) {
              if (((((int)blockIdx.x) * 12) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 14)) < 16) {
                pad_temp_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((1 <= ((((int)blockIdx.y) * 12) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 196) / 14))) && (((((int)blockIdx.y) * 12) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 196) / 14)) < 15)) && (1 <= ((((int)blockIdx.x) * 12) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 14)))) && (((((int)blockIdx.x) * 12) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 13)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 14)) < 15)) ? placeholder[(((((((((((((int)blockIdx.z) >> 3) * 50176) + (rc_outer * 392)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 13)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) - 15))] : 0.000000e+00f);
              }
            }
          }
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if ((((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 18)) + ((int)threadIdx.y)) < 32) {
        if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 9)) < 64) {
          if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 3)) < 192) {
            if (((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 576) {
              if ((((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 72) {
                if (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 18) {
                  placeholder_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((((int)blockIdx.z) & 7) * 73728) + (((int)threadIdx.z) * 9216)) + (((int)threadIdx.y) * 2304)) + (rc_outer * 18)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
                }
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
          for (int xx_c = 0; xx_c < 3; ++xx_c) {
            if (((((int)blockIdx.y) * 12) + ((int)threadIdx.y)) < 14) {
              if ((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) * 3)) + xx_c) < 14) {
                compute_local[(xx_c)] = (compute_local[(xx_c)] + (pad_temp_shared[(((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner))]));
                compute_local[((xx_c + 9))] = (compute_local[((xx_c + 9))] + (pad_temp_shared[(((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144))]));
                compute_local[((xx_c + 18))] = (compute_local[((xx_c + 18))] + (pad_temp_shared[(((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
                compute_local[((xx_c + 27))] = (compute_local[((xx_c + 27))] + (pad_temp_shared[(((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432))]));
              }
            }
            if (((((int)blockIdx.y) * 12) + ((int)threadIdx.y)) < 10) {
              if ((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) * 3)) + xx_c) < 14) {
                compute_local[((xx_c + 3))] = (compute_local[((xx_c + 3))] + (pad_temp_shared[((((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner))]));
                compute_local[((xx_c + 12))] = (compute_local[((xx_c + 12))] + (pad_temp_shared[((((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner) + 56))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144))]));
                compute_local[((xx_c + 21))] = (compute_local[((xx_c + 21))] + (pad_temp_shared[((((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner) + 56))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
                compute_local[((xx_c + 30))] = (compute_local[((xx_c + 30))] + (pad_temp_shared[((((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner) + 56))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432))]));
              }
            }
            if (((((int)blockIdx.y) * 12) + ((int)threadIdx.y)) < 6) {
              if ((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) * 3)) + xx_c) < 14) {
                compute_local[((xx_c + 6))] = (compute_local[((xx_c + 6))] + (pad_temp_shared[((((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner) + 112))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner))]));
                compute_local[((xx_c + 15))] = (compute_local[((xx_c + 15))] + (pad_temp_shared[((((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner) + 112))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144))]));
                compute_local[((xx_c + 24))] = (compute_local[((xx_c + 24))] + (pad_temp_shared[((((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner) + 112))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
                compute_local[((xx_c + 33))] = (compute_local[((xx_c + 33))] + (pad_temp_shared[((((((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + (ry_inner * 14)) + (((int)threadIdx.x) * 3)) + xx_c) + rx_inner) + 112))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432))]));
              }
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int xx_inner_inner_inner = 0; xx_inner_inner_inner < 3; ++xx_inner_inner_inner) {
    if (((((int)blockIdx.y) * 12) + ((int)threadIdx.y)) < 14) {
      if ((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) < 14) {
        compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner))] = compute_local[(xx_inner_inner_inner)];
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 1568))] = compute_local[((xx_inner_inner_inner + 9))];
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 3136))] = compute_local[((xx_inner_inner_inner + 18))];
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 4704))] = compute_local[((xx_inner_inner_inner + 27))];
      }
    }
    if (((((int)blockIdx.y) * 12) + ((int)threadIdx.y)) < 10) {
      if ((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) < 14) {
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 56))] = compute_local[((xx_inner_inner_inner + 3))];
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 1624))] = compute_local[((xx_inner_inner_inner + 12))];
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 3192))] = compute_local[((xx_inner_inner_inner + 21))];
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 4760))] = compute_local[((xx_inner_inner_inner + 30))];
      }
    }
    if (((((int)blockIdx.y) * 12) + ((int)threadIdx.y)) < 6) {
      if ((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) < 14) {
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 112))] = compute_local[((xx_inner_inner_inner + 6))];
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 1680))] = compute_local[((xx_inner_inner_inner + 15))];
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 3248))] = compute_local[((xx_inner_inner_inner + 24))];
        compute[(((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 168)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 12)) + (((int)threadIdx.x) * 3)) + xx_inner_inner_inner) + 4816))] = compute_local[((xx_inner_inner_inner + 33))];
      }
    }
  }
}

