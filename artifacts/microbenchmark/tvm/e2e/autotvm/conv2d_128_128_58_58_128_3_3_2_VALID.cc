//1_7_256_4_2_16
//128_128_58_58_128_3_2_VALID
//dim3 grid(1, 7, 256);
//dim3 block(4, 2, 16);

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
  float compute_local[56];
  __shared__ float pad_temp_shared[513];
  __shared__ float placeholder_shared[576];
  #pragma unroll
  for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
    compute_local[(ff_c_init)] = 0.000000e+00f;
    compute_local[((ff_c_init + 28))] = 0.000000e+00f;
    compute_local[((ff_c_init + 14))] = 0.000000e+00f;
    compute_local[((ff_c_init + 42))] = 0.000000e+00f;
    compute_local[((ff_c_init + 2))] = 0.000000e+00f;
    compute_local[((ff_c_init + 30))] = 0.000000e+00f;
    compute_local[((ff_c_init + 16))] = 0.000000e+00f;
    compute_local[((ff_c_init + 44))] = 0.000000e+00f;
    compute_local[((ff_c_init + 4))] = 0.000000e+00f;
    compute_local[((ff_c_init + 32))] = 0.000000e+00f;
    compute_local[((ff_c_init + 18))] = 0.000000e+00f;
    compute_local[((ff_c_init + 46))] = 0.000000e+00f;
    compute_local[((ff_c_init + 6))] = 0.000000e+00f;
    compute_local[((ff_c_init + 34))] = 0.000000e+00f;
    compute_local[((ff_c_init + 20))] = 0.000000e+00f;
    compute_local[((ff_c_init + 48))] = 0.000000e+00f;
    compute_local[((ff_c_init + 8))] = 0.000000e+00f;
    compute_local[((ff_c_init + 36))] = 0.000000e+00f;
    compute_local[((ff_c_init + 22))] = 0.000000e+00f;
    compute_local[((ff_c_init + 50))] = 0.000000e+00f;
    compute_local[((ff_c_init + 10))] = 0.000000e+00f;
    compute_local[((ff_c_init + 38))] = 0.000000e+00f;
    compute_local[((ff_c_init + 24))] = 0.000000e+00f;
    compute_local[((ff_c_init + 52))] = 0.000000e+00f;
    compute_local[((ff_c_init + 12))] = 0.000000e+00f;
    compute_local[((ff_c_init + 40))] = 0.000000e+00f;
    compute_local[((ff_c_init + 26))] = 0.000000e+00f;
    compute_local[((ff_c_init + 54))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.z) * 33) + (((int)threadIdx.y) * 17)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 513) {
        if ((((((int)threadIdx.y) * 17) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 33) {
          if (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 17) {
            pad_temp_shared[(((((((int)threadIdx.z) * 33) + (((int)threadIdx.y) * 17)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((int)blockIdx.z) >> 1) * 430592) + (rc_outer * 3364)) + (((int)blockIdx.y) * 464)) + ((((((((int)threadIdx.z) * 33) + (((int)threadIdx.y) * 17)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 57) * 58)) + (((((((int)threadIdx.z) * 33) + (((int)threadIdx.y) * 17)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 57)))];
          }
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 9)) < 64) {
        if ((((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 6)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 3)) < 192) {
          if (((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 576) {
            if ((((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 36) {
              if (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 18) {
                placeholder_shared[(((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 73728) + (((int)threadIdx.z) * 4608)) + (((int)threadIdx.y) * 2304)) + ((((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 9) * 1152)) + (rc_outer * 9)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 9)))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
      #pragma unroll
      for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
        #pragma unroll
        for (int ff_c = 0; ff_c < 2; ++ff_c) {
          compute_local[(ff_c)] = (compute_local[(ff_c)] + (pad_temp_shared[(((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 28))] = (compute_local[((ff_c + 28))] + (pad_temp_shared[(((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 14))] = (compute_local[((ff_c + 14))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 228))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 42))] = (compute_local[((ff_c + 42))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 228))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 2))] = (compute_local[((ff_c + 2))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 8))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 30))] = (compute_local[((ff_c + 30))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 8))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 16))] = (compute_local[((ff_c + 16))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 236))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 44))] = (compute_local[((ff_c + 44))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 236))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 4))] = (compute_local[((ff_c + 4))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 16))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 32))] = (compute_local[((ff_c + 32))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 16))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 18))] = (compute_local[((ff_c + 18))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 244))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 46))] = (compute_local[((ff_c + 46))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 244))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 6))] = (compute_local[((ff_c + 6))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 24))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 34))] = (compute_local[((ff_c + 34))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 24))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 20))] = (compute_local[((ff_c + 20))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 252))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 48))] = (compute_local[((ff_c + 48))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 252))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 8))] = (compute_local[((ff_c + 8))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 32))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 36))] = (compute_local[((ff_c + 36))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 32))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 22))] = (compute_local[((ff_c + 22))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 260))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 50))] = (compute_local[((ff_c + 50))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 260))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 10))] = (compute_local[((ff_c + 10))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 40))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 38))] = (compute_local[((ff_c + 38))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 40))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 24))] = (compute_local[((ff_c + 24))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 268))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 52))] = (compute_local[((ff_c + 52))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 268))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 12))] = (compute_local[((ff_c + 12))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 48))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 40))] = (compute_local[((ff_c + 40))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 48))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute_local[((ff_c + 26))] = (compute_local[((ff_c + 26))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 276))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute_local[((ff_c + 54))] = (compute_local[((ff_c + 54))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 276))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff_c * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
        }
      }
    }
  }
  #pragma unroll
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 2; ++ff_inner_inner_inner) {
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)))] = compute_local[(ff_inner_inner_inner)];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25088))] = compute_local[((ff_inner_inner_inner + 28))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 56))] = compute_local[((ff_inner_inner_inner + 14))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25144))] = compute_local[((ff_inner_inner_inner + 42))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 4))] = compute_local[((ff_inner_inner_inner + 2))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25092))] = compute_local[((ff_inner_inner_inner + 30))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 60))] = compute_local[((ff_inner_inner_inner + 16))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25148))] = compute_local[((ff_inner_inner_inner + 44))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 8))] = compute_local[((ff_inner_inner_inner + 4))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25096))] = compute_local[((ff_inner_inner_inner + 32))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 64))] = compute_local[((ff_inner_inner_inner + 18))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25152))] = compute_local[((ff_inner_inner_inner + 46))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 12))] = compute_local[((ff_inner_inner_inner + 6))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25100))] = compute_local[((ff_inner_inner_inner + 34))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 68))] = compute_local[((ff_inner_inner_inner + 20))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25156))] = compute_local[((ff_inner_inner_inner + 48))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 16))] = compute_local[((ff_inner_inner_inner + 8))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25104))] = compute_local[((ff_inner_inner_inner + 36))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 72))] = compute_local[((ff_inner_inner_inner + 22))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25160))] = compute_local[((ff_inner_inner_inner + 50))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 20))] = compute_local[((ff_inner_inner_inner + 10))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25108))] = compute_local[((ff_inner_inner_inner + 38))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 76))] = compute_local[((ff_inner_inner_inner + 24))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25164))] = compute_local[((ff_inner_inner_inner + 52))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 24))] = compute_local[((ff_inner_inner_inner + 12))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25112))] = compute_local[((ff_inner_inner_inner + 40))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 80))] = compute_local[((ff_inner_inner_inner + 26))];
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25168))] = compute_local[((ff_inner_inner_inner + 54))];
  }
}

