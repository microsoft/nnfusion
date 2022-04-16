//1_28_256_28_1_8
//128_256_56_56_128_1_1_SAME
//dim3 grid(1, 28, 256);
//dim3 block(28, 1, 8);

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
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[512];
  #pragma unroll
  for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
    #pragma unroll
    for (int yy_c_init = 0; yy_c_init < 2; ++yy_c_init) {
      compute_local[(((ff_c_init * 2) + yy_c_init))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 8))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 16))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 24))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 4))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 12))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 20))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 28))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 802816) + (rc_outer * 25088)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 512) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) & 1) * 16384) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3) * 256)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 7)))];
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
        for (int yy_c = 0; yy_c < 2; ++yy_c) {
          compute_local[(((ff_c * 2) + yy_c))] = (compute_local[(((ff_c * 2) + yy_c))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 56)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner))]));
          compute_local[((((ff_c * 2) + yy_c) + 8))] = (compute_local[((((ff_c * 2) + yy_c) + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 56)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner) + 128))]));
          compute_local[((((ff_c * 2) + yy_c) + 16))] = (compute_local[((((ff_c * 2) + yy_c) + 16))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 56)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner) + 256))]));
          compute_local[((((ff_c * 2) + yy_c) + 24))] = (compute_local[((((ff_c * 2) + yy_c) + 24))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 56)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner) + 384))]));
          compute_local[((((ff_c * 2) + yy_c) + 4))] = (compute_local[((((ff_c * 2) + yy_c) + 4))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner))]));
          compute_local[((((ff_c * 2) + yy_c) + 12))] = (compute_local[((((ff_c * 2) + yy_c) + 12))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner) + 128))]));
          compute_local[((((ff_c * 2) + yy_c) + 20))] = (compute_local[((((ff_c * 2) + yy_c) + 20))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner) + 256))]));
          compute_local[((((ff_c * 2) + yy_c) + 28))] = (compute_local[((((ff_c * 2) + yy_c) + 28))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner) + 384))]));
        }
      }
    }
  }
  #pragma unroll
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 2; ++ff_inner_inner_inner) {
    #pragma unroll
    for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 2; ++yy_inner_inner_inner) {
      compute[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 56)) + ((int)threadIdx.x)))] = compute_local[(((ff_inner_inner_inner * 2) + yy_inner_inner_inner))];
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 50176))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 8))];
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 100352))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 16))];
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 150528))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 24))];
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 28))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 4))];
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 50204))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 12))];
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 100380))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 20))];
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 150556))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 28))];
    }
  }
}

