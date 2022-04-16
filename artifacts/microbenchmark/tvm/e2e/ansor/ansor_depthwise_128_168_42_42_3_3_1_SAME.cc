//37632_1_1_144_1_1
//128_168_42_42_3_1_SAME
//dim3 grid(37632, 1, 1);
//dim3 block(144, 1, 1);

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
extern "C" __global__ void __launch_bounds__(144) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[7];
  __shared__ float PaddedInput_shared[1408];
  __shared__ float kernel_shared[36];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 7; ++i_c_outer_inner_init) {
    DepthwiseConv2d_local[(i_c_outer_inner_init)] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 18) + (((int)threadIdx.x) >> 3)) < 176) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 144) + ((int)threadIdx.x)) < 1408) {
        PaddedInput_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 144) + ((int)threadIdx.x)))] = (((((1 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 18) + (((int)threadIdx.x) >> 3)) % 44)) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 18) + (((int)threadIdx.x) >> 3)) % 44) < 43)) && (1 <= (((((int)blockIdx.x) % 7) * 6) + (((int)threadIdx.x) & 7)))) && ((((((int)blockIdx.x) % 7) * 6) + (((int)threadIdx.x) & 7)) < 43)) ? data[((((((((((int)blockIdx.x) / 7) * 7056) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 18) + (((int)threadIdx.x) >> 3)) / 44) * 1764)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 18) + (((int)threadIdx.x) >> 3)) % 44) * 42)) + ((((int)blockIdx.x) % 7) * 6)) + (((int)threadIdx.x) & 7)) - 43))] : 0.000000e+00f);
      }
    }
  }
  if (((int)threadIdx.x) < 12) {
    ((float3*)(kernel_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(kernel + (((((((int)blockIdx.x) % 294) / 7) * 36) + (((int)threadIdx.x) * 3)))))[0];
  }
  __syncthreads();
  for (int i_c_outer_inner = 0; i_c_outer_inner < 7; ++i_c_outer_inner) {
    for (int di_inner = 0; di_inner < 3; ++di_inner) {
      for (int dj_inner = 0; dj_inner < 3; ++dj_inner) {
        DepthwiseConv2d_local[(i_c_outer_inner)] = (DepthwiseConv2d_local[(i_c_outer_inner)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 36) * 352) + (((((int)threadIdx.x) % 36) / 6) * 56)) + (i_c_outer_inner * 8)) + (di_inner * 8)) + dj_inner) + (((int)threadIdx.x) % 6)))] * kernel_shared[(((((((int)threadIdx.x) / 36) * 9) + (di_inner * 3)) + dj_inner))]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 7; ++i_inner) {
    DepthwiseConv2d[(((((((((int)blockIdx.x) / 7) * 7056) + ((((int)threadIdx.x) / 6) * 294)) + (i_inner * 42)) + ((((int)blockIdx.x) % 7) * 6)) + (((int)threadIdx.x) % 6)))] = DepthwiseConv2d_local[(i_inner)];
  }
}

