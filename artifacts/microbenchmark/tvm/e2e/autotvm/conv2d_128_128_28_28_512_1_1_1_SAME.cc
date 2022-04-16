//1_7_1024_28_1_8
//128_128_28_28_512_1_1_SAME
//dim3 grid(1, 7, 1024);
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
  for (int yy_c_init = 0; yy_c_init < 2; ++yy_c_init) {
    compute_local[(yy_c_init)] = 0.000000e+00f;
    compute_local[((yy_c_init + 4))] = 0.000000e+00f;
    compute_local[((yy_c_init + 8))] = 0.000000e+00f;
    compute_local[((yy_c_init + 12))] = 0.000000e+00f;
    compute_local[((yy_c_init + 16))] = 0.000000e+00f;
    compute_local[((yy_c_init + 20))] = 0.000000e+00f;
    compute_local[((yy_c_init + 24))] = 0.000000e+00f;
    compute_local[((yy_c_init + 28))] = 0.000000e+00f;
    compute_local[((yy_c_init + 2))] = 0.000000e+00f;
    compute_local[((yy_c_init + 6))] = 0.000000e+00f;
    compute_local[((yy_c_init + 10))] = 0.000000e+00f;
    compute_local[((yy_c_init + 14))] = 0.000000e+00f;
    compute_local[((yy_c_init + 18))] = 0.000000e+00f;
    compute_local[((yy_c_init + 22))] = 0.000000e+00f;
    compute_local[((yy_c_init + 26))] = 0.000000e+00f;
    compute_local[((yy_c_init + 30))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((((((int)blockIdx.z) >> 3) * 100352) + (rc_outer * 6272)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 512) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 8192) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3) * 128)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      #pragma unroll
      for (int yy_c = 0; yy_c < 2; ++yy_c) {
        compute_local[(yy_c)] = (compute_local[(yy_c)] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
        compute_local[((yy_c + 4))] = (compute_local[((yy_c + 4))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 64))]));
        compute_local[((yy_c + 8))] = (compute_local[((yy_c + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))]));
        compute_local[((yy_c + 12))] = (compute_local[((yy_c + 12))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 192))]));
        compute_local[((yy_c + 16))] = (compute_local[((yy_c + 16))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
        compute_local[((yy_c + 20))] = (compute_local[((yy_c + 20))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 320))]));
        compute_local[((yy_c + 24))] = (compute_local[((yy_c + 24))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 384))]));
        compute_local[((yy_c + 28))] = (compute_local[((yy_c + 28))] + (pad_temp_shared[((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 448))]));
        compute_local[((yy_c + 2))] = (compute_local[((yy_c + 2))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
        compute_local[((yy_c + 6))] = (compute_local[((yy_c + 6))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 64))]));
        compute_local[((yy_c + 10))] = (compute_local[((yy_c + 10))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))]));
        compute_local[((yy_c + 14))] = (compute_local[((yy_c + 14))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 192))]));
        compute_local[((yy_c + 18))] = (compute_local[((yy_c + 18))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
        compute_local[((yy_c + 22))] = (compute_local[((yy_c + 22))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 320))]));
        compute_local[((yy_c + 26))] = (compute_local[((yy_c + 26))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 384))]));
        compute_local[((yy_c + 30))] = (compute_local[((yy_c + 30))] + (pad_temp_shared[(((((rc_inner * 112) + (yy_c * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 448))]));
      }
    }
  }
  #pragma unroll
  for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 2; ++yy_inner_inner_inner) {
    compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)))] = compute_local[(yy_inner_inner_inner)];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6272))] = compute_local[((yy_inner_inner_inner + 4))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12544))] = compute_local[((yy_inner_inner_inner + 8))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18816))] = compute_local[((yy_inner_inner_inner + 12))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 25088))] = compute_local[((yy_inner_inner_inner + 16))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 31360))] = compute_local[((yy_inner_inner_inner + 20))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 37632))] = compute_local[((yy_inner_inner_inner + 24))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 43904))] = compute_local[((yy_inner_inner_inner + 28))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 56))] = compute_local[((yy_inner_inner_inner + 2))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6328))] = compute_local[((yy_inner_inner_inner + 6))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12600))] = compute_local[((yy_inner_inner_inner + 10))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18872))] = compute_local[((yy_inner_inner_inner + 14))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 25144))] = compute_local[((yy_inner_inner_inner + 18))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 31416))] = compute_local[((yy_inner_inner_inner + 22))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 37688))] = compute_local[((yy_inner_inner_inner + 26))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (yy_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 43960))] = compute_local[((yy_inner_inner_inner + 30))];
  }
}

