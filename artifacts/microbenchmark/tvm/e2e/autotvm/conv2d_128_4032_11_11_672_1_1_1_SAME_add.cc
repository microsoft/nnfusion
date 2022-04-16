//1_1_768_11_1_28
//128_4032_11_11_672_1_1_SAME
//dim3 grid(1, 1, 768);
//dim3 block(11, 1, 28);

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
extern "C" __global__ void __launch_bounds__(308) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[44];
  __shared__ float pad_temp_shared[1936];
  __shared__ float placeholder_shared[1792];
  #pragma unroll
  for (int yy_init = 0; yy_init < 11; ++yy_init) {
    compute[(yy_init)] = 0.000000e+00f;
    compute[((yy_init + 11))] = 0.000000e+00f;
    compute[((yy_init + 22))] = 0.000000e+00f;
    compute[((yy_init + 33))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 252; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 70) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 1936) {
        if (((((int)threadIdx.x) * 7) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 70) {
          pad_temp_shared[((((((int)threadIdx.z) * 70) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((int)blockIdx.z) / 6) * 487872) + (rc_outer * 1936)) + (((int)threadIdx.z) * 70)) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 112) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1792) {
          if (((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) % 6) * 451584) + (((int)threadIdx.z) * 16128)) + ((((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 4032)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      #pragma unroll
      for (int yy = 0; yy < 11; ++yy) {
        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 121) + (yy * 11)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
        compute[((yy + 11))] = (compute[((yy + 11))] + (pad_temp_shared[((((rc_inner * 121) + (yy * 11)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 448))]));
        compute[((yy + 22))] = (compute[((yy + 22))] + (pad_temp_shared[((((rc_inner * 121) + (yy * 11)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 896))]));
        compute[((yy + 33))] = (compute[((yy + 33))] + (pad_temp_shared[((((rc_inner * 121) + (yy * 11)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 1344))]));
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 11; ++ax2_inner_inner_inner) {
    T_add[(((((((int)blockIdx.z) * 13552) + (((int)threadIdx.z) * 121)) + (ax2_inner_inner_inner * 11)) + ((int)threadIdx.x)))] = (compute[(ax2_inner_inner_inner)] + input2[(((((((int)blockIdx.z) * 13552) + (((int)threadIdx.z) * 121)) + (ax2_inner_inner_inner * 11)) + ((int)threadIdx.x)))]);
    T_add[((((((((int)blockIdx.z) * 13552) + (((int)threadIdx.z) * 121)) + (ax2_inner_inner_inner * 11)) + ((int)threadIdx.x)) + 3388))] = (compute[((ax2_inner_inner_inner + 11))] + input2[((((((((int)blockIdx.z) * 13552) + (((int)threadIdx.z) * 121)) + (ax2_inner_inner_inner * 11)) + ((int)threadIdx.x)) + 3388))]);
    T_add[((((((((int)blockIdx.z) * 13552) + (((int)threadIdx.z) * 121)) + (ax2_inner_inner_inner * 11)) + ((int)threadIdx.x)) + 6776))] = (compute[((ax2_inner_inner_inner + 22))] + input2[((((((((int)blockIdx.z) * 13552) + (((int)threadIdx.z) * 121)) + (ax2_inner_inner_inner * 11)) + ((int)threadIdx.x)) + 6776))]);
    T_add[((((((((int)blockIdx.z) * 13552) + (((int)threadIdx.z) * 121)) + (ax2_inner_inner_inner * 11)) + ((int)threadIdx.x)) + 10164))] = (compute[((ax2_inner_inner_inner + 33))] + input2[((((((((int)blockIdx.z) * 13552) + (((int)threadIdx.z) * 121)) + (ax2_inner_inner_inner * 11)) + ((int)threadIdx.x)) + 10164))]);
  }
}

