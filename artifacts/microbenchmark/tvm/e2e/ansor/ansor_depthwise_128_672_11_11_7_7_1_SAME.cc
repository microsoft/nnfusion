//10752_1_1_88_1_1
//128_672_11_11_7_1_SAME
//dim3 grid(10752, 1, 1);
//dim3 block(88, 1, 1);

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
extern "C" __global__ void __launch_bounds__(88) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[11];
  __shared__ float PaddedInput_shared[2312];
  __shared__ float kernel_shared[98];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  DepthwiseConv2d_local[(7)] = 0.000000e+00f;
  DepthwiseConv2d_local[(8)] = 0.000000e+00f;
  DepthwiseConv2d_local[(9)] = 0.000000e+00f;
  DepthwiseConv2d_local[(10)] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 88) + ((int)threadIdx.x)) < 136) {
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_s = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_s < 17; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1496) + (((int)threadIdx.x) * 17)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) < 2312) {
          PaddedInput_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1496) + (((int)threadIdx.x) * 17)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s))] = (((((3 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 88) + ((int)threadIdx.x)) % 17)) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 88) + ((int)threadIdx.x)) % 17) < 14)) && (3 <= ax0_ax1_fused_ax2_fused_ax3_fused_inner_s)) && (ax0_ax1_fused_ax2_fused_ax3_fused_inner_s < 14)) ? data[(((((((((((int)blockIdx.x) / 336) * 325248) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 88) + ((int)threadIdx.x)) / 34) * 81312)) + ((((int)blockIdx.x) % 336) * 242)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 88) + ((int)threadIdx.x)) % 34) / 17) * 121)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 88) + ((int)threadIdx.x)) % 17) * 11)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) - 36))] : 0.000000e+00f);
        }
      }
    }
  }
  kernel_shared[(((int)threadIdx.x))] = kernel[((((((int)blockIdx.x) % 336) * 98) + ((int)threadIdx.x)))];
  if (((int)threadIdx.x) < 10) {
    kernel_shared[((((int)threadIdx.x) + 88))] = kernel[(((((((int)blockIdx.x) % 336) * 98) + ((int)threadIdx.x)) + 88))];
  }
  __syncthreads();
  for (int dj_outer_inner = 0; dj_outer_inner < 7; ++dj_outer_inner) {
    for (int i_c_outer_inner = 0; i_c_outer_inner < 11; ++i_c_outer_inner) {
      DepthwiseConv2d_local[(i_c_outer_inner)] = (DepthwiseConv2d_local[(i_c_outer_inner)] + (PaddedInput_shared[((((((((int)threadIdx.x) / 11) * 289) + (i_c_outer_inner * 17)) + dj_outer_inner) + (((int)threadIdx.x) % 11)))] * kernel_shared[(((((((int)threadIdx.x) % 22) / 11) * 49) + dj_outer_inner))]));
      DepthwiseConv2d_local[(i_c_outer_inner)] = (DepthwiseConv2d_local[(i_c_outer_inner)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 11) * 289) + (i_c_outer_inner * 17)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 17))] * kernel_shared[((((((((int)threadIdx.x) % 22) / 11) * 49) + dj_outer_inner) + 7))]));
      DepthwiseConv2d_local[(i_c_outer_inner)] = (DepthwiseConv2d_local[(i_c_outer_inner)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 11) * 289) + (i_c_outer_inner * 17)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 34))] * kernel_shared[((((((((int)threadIdx.x) % 22) / 11) * 49) + dj_outer_inner) + 14))]));
      DepthwiseConv2d_local[(i_c_outer_inner)] = (DepthwiseConv2d_local[(i_c_outer_inner)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 11) * 289) + (i_c_outer_inner * 17)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 51))] * kernel_shared[((((((((int)threadIdx.x) % 22) / 11) * 49) + dj_outer_inner) + 21))]));
      DepthwiseConv2d_local[(i_c_outer_inner)] = (DepthwiseConv2d_local[(i_c_outer_inner)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 11) * 289) + (i_c_outer_inner * 17)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 68))] * kernel_shared[((((((((int)threadIdx.x) % 22) / 11) * 49) + dj_outer_inner) + 28))]));
      DepthwiseConv2d_local[(i_c_outer_inner)] = (DepthwiseConv2d_local[(i_c_outer_inner)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 11) * 289) + (i_c_outer_inner * 17)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 85))] * kernel_shared[((((((((int)threadIdx.x) % 22) / 11) * 49) + dj_outer_inner) + 35))]));
      DepthwiseConv2d_local[(i_c_outer_inner)] = (DepthwiseConv2d_local[(i_c_outer_inner)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 11) * 289) + (i_c_outer_inner * 17)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 102))] * kernel_shared[((((((((int)threadIdx.x) % 22) / 11) * 49) + dj_outer_inner) + 42))]));
    }
  }
  for (int i_inner = 0; i_inner < 11; ++i_inner) {
    DepthwiseConv2d[((((((((((int)blockIdx.x) / 336) * 325248) + ((((int)threadIdx.x) / 22) * 81312)) + ((((int)blockIdx.x) % 336) * 242)) + (((((int)threadIdx.x) % 22) / 11) * 121)) + (i_inner * 11)) + (((int)threadIdx.x) % 11)))] = DepthwiseConv2d_local[(i_inner)];
  }
}

