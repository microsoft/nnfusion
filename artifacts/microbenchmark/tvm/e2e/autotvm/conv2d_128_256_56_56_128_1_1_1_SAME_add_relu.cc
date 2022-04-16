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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[32];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[512];
  #pragma unroll
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 2; ++yy_init) {
      compute1[(((ff_init * 2) + yy_init))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 8))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 16))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 24))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 4))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 12))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 20))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 28))] = 0.000000e+00f;
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
      for (int ff = 0; ff < 2; ++ff) {
        #pragma unroll
        for (int yy = 0; yy < 2; ++yy) {
          compute1[(((ff * 2) + yy))] = (compute1[(((ff * 2) + yy))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner))]));
          compute1[((((ff * 2) + yy) + 8))] = (compute1[((((ff * 2) + yy) + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 128))]));
          compute1[((((ff * 2) + yy) + 16))] = (compute1[((((ff * 2) + yy) + 16))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 256))]));
          compute1[((((ff * 2) + yy) + 24))] = (compute1[((((ff * 2) + yy) + 24))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 384))]));
          compute1[((((ff * 2) + yy) + 4))] = (compute1[((((ff * 2) + yy) + 4))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner))]));
          compute1[((((ff * 2) + yy) + 12))] = (compute1[((((ff * 2) + yy) + 12))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 128))]));
          compute1[((((ff * 2) + yy) + 20))] = (compute1[((((ff * 2) + yy) + 20))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 256))]));
          compute1[((((ff * 2) + yy) + 28))] = (compute1[((((ff * 2) + yy) + 28))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 384))]));
        }
      }
    }
  }
  #pragma unroll
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 2; ++i1_inner_inner_inner) {
    #pragma unroll
    for (int i2_inner_inner_inner = 0; i2_inner_inner_inner < 2; ++i2_inner_inner_inner) {
      compute[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)))] = max((compute1[(((i1_inner_inner_inner * 2) + i2_inner_inner_inner))] + input2[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 50176))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 8))] + input2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 50176))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 100352))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 16))] + input2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 100352))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 150528))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 24))] + input2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 150528))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 28))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 4))] + input2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 28))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 50204))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 12))] + input2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 50204))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 100380))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 20))] + input2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 100380))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 150556))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 28))] + input2[((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 56)) + ((int)threadIdx.x)) + 150556))]), 0.000000e+00f);
    }
  }
}

