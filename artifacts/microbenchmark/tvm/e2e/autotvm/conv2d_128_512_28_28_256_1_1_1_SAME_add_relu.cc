//1_7_1024_28_1_2
//128_512_28_28_256_1_1_SAME
//dim3 grid(1, 7, 1024);
//dim3 block(28, 1, 2);

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
extern "C" __global__ void __launch_bounds__(56) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[64];
  __shared__ float pad_temp_shared[224];
  __shared__ float placeholder_shared[64];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 2; ++yy_init) {
      compute1[(((ff_init * 2) + yy_init))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 16))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 32))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 48))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 8))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 24))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 40))] = 0.000000e+00f;
      compute1[((((ff_init * 2) + yy_init) + 56))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 256; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((((((int)blockIdx.z) >> 3) * 401408) + (rc_outer * 1568)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) < 32) {
        if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 32) {
            if (((((((int)blockIdx.z) & 7) * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.x)) < 256) {
              placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 16384) + (((int)threadIdx.z) * 8192)) + (((int)threadIdx.x) * 512)) + (rc_outer * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
            }
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
      #pragma unroll
      for (int ff = 0; ff < 4; ++ff) {
        #pragma unroll
        for (int yy = 0; yy < 2; ++yy) {
          compute1[(((ff * 2) + yy))] = (compute1[(((ff * 2) + yy))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + (ff * 2)) + rc_inner))]));
          compute1[((((ff * 2) + yy) + 16))] = (compute1[((((ff * 2) + yy) + 16))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff * 2)) + rc_inner) + 16))]));
          compute1[((((ff * 2) + yy) + 32))] = (compute1[((((ff * 2) + yy) + 32))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff * 2)) + rc_inner) + 32))]));
          compute1[((((ff * 2) + yy) + 48))] = (compute1[((((ff * 2) + yy) + 48))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff * 2)) + rc_inner) + 48))]));
          compute1[((((ff * 2) + yy) + 8))] = (compute1[((((ff * 2) + yy) + 8))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + (ff * 2)) + rc_inner))]));
          compute1[((((ff * 2) + yy) + 24))] = (compute1[((((ff * 2) + yy) + 24))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff * 2)) + rc_inner) + 16))]));
          compute1[((((ff * 2) + yy) + 40))] = (compute1[((((ff * 2) + yy) + 40))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff * 2)) + rc_inner) + 32))]));
          compute1[((((ff * 2) + yy) + 56))] = (compute1[((((ff * 2) + yy) + 56))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff * 2)) + rc_inner) + 48))]));
        }
      }
    }
  }
  #pragma unroll
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 4; ++i1_inner_inner_inner) {
    #pragma unroll
    for (int i2_inner_inner_inner = 0; i2_inner_inner_inner < 2; ++i2_inner_inner_inner) {
      compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)))] = max((compute1[(((i1_inner_inner_inner * 2) + i2_inner_inner_inner))] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6272))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 16))] + input2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6272))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12544))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 32))] + input2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12544))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18816))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 48))] + input2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18816))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 56))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 8))] + input2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 56))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6328))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 24))] + input2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6328))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12600))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 40))] + input2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12600))]), 0.000000e+00f);
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18872))] = max((compute1[((((i1_inner_inner_inner * 2) + i2_inner_inner_inner) + 56))] + input2[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (i1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (i2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18872))]), 0.000000e+00f);
    }
  }
}

