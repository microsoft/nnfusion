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
extern "C" __global__ void __launch_bounds__(128) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[56];
  __shared__ float pad_temp_shared[513];
  __shared__ float placeholder_shared[576];
  #pragma unroll
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    compute1[(ff_init)] = 0.000000e+00f;
    compute1[((ff_init + 28))] = 0.000000e+00f;
    compute1[((ff_init + 14))] = 0.000000e+00f;
    compute1[((ff_init + 42))] = 0.000000e+00f;
    compute1[((ff_init + 2))] = 0.000000e+00f;
    compute1[((ff_init + 30))] = 0.000000e+00f;
    compute1[((ff_init + 16))] = 0.000000e+00f;
    compute1[((ff_init + 44))] = 0.000000e+00f;
    compute1[((ff_init + 4))] = 0.000000e+00f;
    compute1[((ff_init + 32))] = 0.000000e+00f;
    compute1[((ff_init + 18))] = 0.000000e+00f;
    compute1[((ff_init + 46))] = 0.000000e+00f;
    compute1[((ff_init + 6))] = 0.000000e+00f;
    compute1[((ff_init + 34))] = 0.000000e+00f;
    compute1[((ff_init + 20))] = 0.000000e+00f;
    compute1[((ff_init + 48))] = 0.000000e+00f;
    compute1[((ff_init + 8))] = 0.000000e+00f;
    compute1[((ff_init + 36))] = 0.000000e+00f;
    compute1[((ff_init + 22))] = 0.000000e+00f;
    compute1[((ff_init + 50))] = 0.000000e+00f;
    compute1[((ff_init + 10))] = 0.000000e+00f;
    compute1[((ff_init + 38))] = 0.000000e+00f;
    compute1[((ff_init + 24))] = 0.000000e+00f;
    compute1[((ff_init + 52))] = 0.000000e+00f;
    compute1[((ff_init + 12))] = 0.000000e+00f;
    compute1[((ff_init + 40))] = 0.000000e+00f;
    compute1[((ff_init + 26))] = 0.000000e+00f;
    compute1[((ff_init + 54))] = 0.000000e+00f;
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
        for (int ff = 0; ff < 2; ++ff) {
          compute1[(ff)] = (compute1[(ff)] + (pad_temp_shared[(((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 28))] = (compute1[((ff + 28))] + (pad_temp_shared[(((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 14))] = (compute1[((ff + 14))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 228))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 42))] = (compute1[((ff + 42))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 228))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 2))] = (compute1[((ff + 2))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 8))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 30))] = (compute1[((ff + 30))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 8))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 16))] = (compute1[((ff + 16))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 236))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 44))] = (compute1[((ff + 44))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 236))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 4))] = (compute1[((ff + 4))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 16))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 32))] = (compute1[((ff + 32))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 16))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 18))] = (compute1[((ff + 18))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 244))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 46))] = (compute1[((ff + 46))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 244))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 6))] = (compute1[((ff + 6))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 24))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 34))] = (compute1[((ff + 34))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 24))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 20))] = (compute1[((ff + 20))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 252))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 48))] = (compute1[((ff + 48))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 252))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 8))] = (compute1[((ff + 8))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 32))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 36))] = (compute1[((ff + 36))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 32))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 22))] = (compute1[((ff + 22))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 260))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 50))] = (compute1[((ff + 50))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 260))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 10))] = (compute1[((ff + 10))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 40))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 38))] = (compute1[((ff + 38))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 40))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 24))] = (compute1[((ff + 24))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 268))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 52))] = (compute1[((ff + 52))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 268))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 12))] = (compute1[((ff + 12))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 48))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 40))] = (compute1[((ff + 40))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 48))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
          compute1[((ff + 26))] = (compute1[((ff + 26))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 276))] * placeholder_shared[(((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner))]));
          compute1[((ff + 54))] = (compute1[((ff + 54))] + (pad_temp_shared[((((((((int)threadIdx.y) * 114) + (ry_inner * 57)) + (((int)threadIdx.x) * 2)) + rx_inner) + 276))] * placeholder_shared[((((((((int)threadIdx.z) * 18) + (ff * 9)) + (ry_inner * 3)) + rx_inner) + 288))]));
        }
      }
    }
  }
  #pragma unroll
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 2; ++i1_inner_inner_inner) {
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)))] = max((compute1[(i1_inner_inner_inner)] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25088))] = max((compute1[((i1_inner_inner_inner + 28))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25088))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 56))] = max((compute1[((i1_inner_inner_inner + 14))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 56))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25144))] = max((compute1[((i1_inner_inner_inner + 42))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25144))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 4))] = max((compute1[((i1_inner_inner_inner + 2))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 4))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25092))] = max((compute1[((i1_inner_inner_inner + 30))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25092))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 60))] = max((compute1[((i1_inner_inner_inner + 16))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 60))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25148))] = max((compute1[((i1_inner_inner_inner + 44))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25148))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 8))] = max((compute1[((i1_inner_inner_inner + 4))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 8))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25096))] = max((compute1[((i1_inner_inner_inner + 32))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25096))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 64))] = max((compute1[((i1_inner_inner_inner + 18))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 64))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25152))] = max((compute1[((i1_inner_inner_inner + 46))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25152))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 12))] = max((compute1[((i1_inner_inner_inner + 6))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 12))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25100))] = max((compute1[((i1_inner_inner_inner + 34))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25100))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 68))] = max((compute1[((i1_inner_inner_inner + 20))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 68))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25156))] = max((compute1[((i1_inner_inner_inner + 48))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25156))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 16))] = max((compute1[((i1_inner_inner_inner + 8))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 16))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25104))] = max((compute1[((i1_inner_inner_inner + 36))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25104))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 72))] = max((compute1[((i1_inner_inner_inner + 22))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 72))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25160))] = max((compute1[((i1_inner_inner_inner + 50))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25160))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 20))] = max((compute1[((i1_inner_inner_inner + 10))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 20))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25108))] = max((compute1[((i1_inner_inner_inner + 38))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25108))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 76))] = max((compute1[((i1_inner_inner_inner + 24))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 76))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25164))] = max((compute1[((i1_inner_inner_inner + 52))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25164))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 24))] = max((compute1[((i1_inner_inner_inner + 12))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 24))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25112))] = max((compute1[((i1_inner_inner_inner + 40))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25112))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 80))] = max((compute1[((i1_inner_inner_inner + 26))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 80))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25168))] = max((compute1[((i1_inner_inner_inner + 54))] + input2[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 1568)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) + 25168))]), 0.000000e+00f);
  }
}

