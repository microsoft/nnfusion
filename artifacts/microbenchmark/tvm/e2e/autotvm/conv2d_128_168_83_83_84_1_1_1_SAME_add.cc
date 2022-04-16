//1_83_128_83_1_3
//128_168_83_83_84_1_1_SAME
//dim3 grid(1, 83, 128);
//dim3 block(83, 1, 3);

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
extern "C" __global__ void __launch_bounds__(249) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[28];
  __shared__ float pad_temp_shared[498];
  __shared__ float placeholder_shared[504];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    compute[(ff_init)] = 0.000000e+00f;
    compute[((ff_init + 4))] = 0.000000e+00f;
    compute[((ff_init + 8))] = 0.000000e+00f;
    compute[((ff_init + 12))] = 0.000000e+00f;
    compute[((ff_init + 16))] = 0.000000e+00f;
    compute[((ff_init + 20))] = 0.000000e+00f;
    compute[((ff_init + 24))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 28; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 166) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((int)blockIdx.z) * 1157352) + (rc_outer * 41334)) + (((int)threadIdx.z) * 13778)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 83) * 6889)) + (((int)blockIdx.y) * 83)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 83)))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 28) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 6)) < 84) {
        if ((((((int)threadIdx.z) * 168) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 504) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 168) {
            placeholder_shared[((((((int)threadIdx.z) * 168) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((int)threadIdx.z) * 4704) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 6) * 168)) + (rc_outer * 6)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 6)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 6; ++rc_inner) {
      #pragma unroll
      for (int ff = 0; ff < 4; ++ff) {
        compute[(ff)] = (compute[(ff)] + (pad_temp_shared[(((rc_inner * 83) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 24) + (ff * 6)) + rc_inner))]));
        compute[((ff + 4))] = (compute[((ff + 4))] + (pad_temp_shared[(((rc_inner * 83) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 24) + (ff * 6)) + rc_inner) + 72))]));
        compute[((ff + 8))] = (compute[((ff + 8))] + (pad_temp_shared[(((rc_inner * 83) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 24) + (ff * 6)) + rc_inner) + 144))]));
        compute[((ff + 12))] = (compute[((ff + 12))] + (pad_temp_shared[(((rc_inner * 83) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 24) + (ff * 6)) + rc_inner) + 216))]));
        compute[((ff + 16))] = (compute[((ff + 16))] + (pad_temp_shared[(((rc_inner * 83) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 24) + (ff * 6)) + rc_inner) + 288))]));
        compute[((ff + 20))] = (compute[((ff + 20))] + (pad_temp_shared[(((rc_inner * 83) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 24) + (ff * 6)) + rc_inner) + 360))]));
        compute[((ff + 24))] = (compute[((ff + 24))] + (pad_temp_shared[(((rc_inner * 83) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 24) + (ff * 6)) + rc_inner) + 432))]));
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 4; ++ax1_inner_inner_inner) {
    T_add[((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)))] = (compute[(ax1_inner_inner_inner)] + input2[((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)))]);
    T_add[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 82668))] = (compute[((ax1_inner_inner_inner + 4))] + input2[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 82668))]);
    T_add[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 165336))] = (compute[((ax1_inner_inner_inner + 8))] + input2[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 165336))]);
    T_add[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 248004))] = (compute[((ax1_inner_inner_inner + 12))] + input2[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 248004))]);
    T_add[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 330672))] = (compute[((ax1_inner_inner_inner + 16))] + input2[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 330672))]);
    T_add[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 413340))] = (compute[((ax1_inner_inner_inner + 20))] + input2[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 413340))]);
    T_add[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 496008))] = (compute[((ax1_inner_inner_inner + 24))] + input2[(((((((((int)blockIdx.z) * 578676) + (((int)threadIdx.z) * 27556)) + (ax1_inner_inner_inner * 6889)) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 496008))]);
  }
}

