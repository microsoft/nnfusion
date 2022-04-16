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
extern "C" __global__ void __launch_bounds__(56) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[64];
  __shared__ float pad_temp_shared[224];
  __shared__ float placeholder_shared[64];
  #pragma unroll
  for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
    #pragma unroll
    for (int yy_c_init = 0; yy_c_init < 2; ++yy_c_init) {
      compute_local[(((ff_c_init * 2) + yy_c_init))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 16))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 32))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 48))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 8))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 24))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 40))] = 0.000000e+00f;
      compute_local[((((ff_c_init * 2) + yy_c_init) + 56))] = 0.000000e+00f;
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
      for (int ff_c = 0; ff_c < 4; ++ff_c) {
        #pragma unroll
        for (int yy_c = 0; yy_c < 2; ++yy_c) {
          compute_local[(((ff_c * 2) + yy_c))] = (compute_local[(((ff_c * 2) + yy_c))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + (ff_c * 2)) + rc_inner))]));
          compute_local[((((ff_c * 2) + yy_c) + 16))] = (compute_local[((((ff_c * 2) + yy_c) + 16))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff_c * 2)) + rc_inner) + 16))]));
          compute_local[((((ff_c * 2) + yy_c) + 32))] = (compute_local[((((ff_c * 2) + yy_c) + 32))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff_c * 2)) + rc_inner) + 32))]));
          compute_local[((((ff_c * 2) + yy_c) + 48))] = (compute_local[((((ff_c * 2) + yy_c) + 48))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff_c * 2)) + rc_inner) + 48))]));
          compute_local[((((ff_c * 2) + yy_c) + 8))] = (compute_local[((((ff_c * 2) + yy_c) + 8))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + (ff_c * 2)) + rc_inner))]));
          compute_local[((((ff_c * 2) + yy_c) + 24))] = (compute_local[((((ff_c * 2) + yy_c) + 24))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff_c * 2)) + rc_inner) + 16))]));
          compute_local[((((ff_c * 2) + yy_c) + 40))] = (compute_local[((((ff_c * 2) + yy_c) + 40))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff_c * 2)) + rc_inner) + 32))]));
          compute_local[((((ff_c * 2) + yy_c) + 56))] = (compute_local[((((ff_c * 2) + yy_c) + 56))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 8) + (ff_c * 2)) + rc_inner) + 48))]));
        }
      }
    }
  }
  #pragma unroll
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
    #pragma unroll
    for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 2; ++yy_inner_inner_inner) {
      compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)))] = compute_local[(((ff_inner_inner_inner * 2) + yy_inner_inner_inner))];
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6272))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 16))];
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12544))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 32))];
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18816))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 48))];
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 56))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 8))];
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6328))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 24))];
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12600))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 40))];
      compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18872))] = compute_local[((((ff_inner_inner_inner * 2) + yy_inner_inner_inner) + 56))];
    }
  }
}

