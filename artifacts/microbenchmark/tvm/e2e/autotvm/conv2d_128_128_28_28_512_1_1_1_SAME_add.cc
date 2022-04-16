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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[32];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[512];
  #pragma unroll
  for (int yy_init = 0; yy_init < 2; ++yy_init) {
    compute[(yy_init)] = 0.000000e+00f;
    compute[((yy_init + 4))] = 0.000000e+00f;
    compute[((yy_init + 8))] = 0.000000e+00f;
    compute[((yy_init + 12))] = 0.000000e+00f;
    compute[((yy_init + 16))] = 0.000000e+00f;
    compute[((yy_init + 20))] = 0.000000e+00f;
    compute[((yy_init + 24))] = 0.000000e+00f;
    compute[((yy_init + 28))] = 0.000000e+00f;
    compute[((yy_init + 2))] = 0.000000e+00f;
    compute[((yy_init + 6))] = 0.000000e+00f;
    compute[((yy_init + 10))] = 0.000000e+00f;
    compute[((yy_init + 14))] = 0.000000e+00f;
    compute[((yy_init + 18))] = 0.000000e+00f;
    compute[((yy_init + 22))] = 0.000000e+00f;
    compute[((yy_init + 26))] = 0.000000e+00f;
    compute[((yy_init + 30))] = 0.000000e+00f;
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
      for (int yy = 0; yy < 2; ++yy) {
        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
        compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 64))]));
        compute[((yy + 8))] = (compute[((yy + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))]));
        compute[((yy + 12))] = (compute[((yy + 12))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 192))]));
        compute[((yy + 16))] = (compute[((yy + 16))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
        compute[((yy + 20))] = (compute[((yy + 20))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 320))]));
        compute[((yy + 24))] = (compute[((yy + 24))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 384))]));
        compute[((yy + 28))] = (compute[((yy + 28))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 448))]));
        compute[((yy + 2))] = (compute[((yy + 2))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
        compute[((yy + 6))] = (compute[((yy + 6))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 64))]));
        compute[((yy + 10))] = (compute[((yy + 10))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 128))]));
        compute[((yy + 14))] = (compute[((yy + 14))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 192))]));
        compute[((yy + 18))] = (compute[((yy + 18))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 256))]));
        compute[((yy + 22))] = (compute[((yy + 22))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 320))]));
        compute[((yy + 26))] = (compute[((yy + 26))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 384))]));
        compute[((yy + 30))] = (compute[((yy + 30))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 448))]));
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    T_add[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)))] = (compute[(ax2_inner_inner_inner)] + input2[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6272))] = (compute[((ax2_inner_inner_inner + 4))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6272))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12544))] = (compute[((ax2_inner_inner_inner + 8))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12544))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18816))] = (compute[((ax2_inner_inner_inner + 12))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18816))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 25088))] = (compute[((ax2_inner_inner_inner + 16))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 25088))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 31360))] = (compute[((ax2_inner_inner_inner + 20))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 31360))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 37632))] = (compute[((ax2_inner_inner_inner + 24))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 37632))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 43904))] = (compute[((ax2_inner_inner_inner + 28))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 43904))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 56))] = (compute[((ax2_inner_inner_inner + 2))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 56))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6328))] = (compute[((ax2_inner_inner_inner + 6))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 6328))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12600))] = (compute[((ax2_inner_inner_inner + 10))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 12600))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18872))] = (compute[((ax2_inner_inner_inner + 14))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 18872))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 25144))] = (compute[((ax2_inner_inner_inner + 18))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 25144))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 31416))] = (compute[((ax2_inner_inner_inner + 22))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 31416))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 37688))] = (compute[((ax2_inner_inner_inner + 26))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 37688))]);
    T_add[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 43960))] = (compute[((ax2_inner_inner_inner + 30))] + input2[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + ((int)threadIdx.x)) + 43960))]);
  }
}

