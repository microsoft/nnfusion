//3584_1_1_363_1_1
//128_672_11_11_5_1_SAME
//dim3 grid(3584, 1, 1);
//dim3 block(363, 1, 1);

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
extern "C" __global__ void __launch_bounds__(363) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[8];
  __shared__ float PaddedInput_shared[5400];
  __shared__ float kernel_shared[75];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  DepthwiseConv2d_local[(7)] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_s = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_s < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1815) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) < 5400) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 363) + ((int)threadIdx.x)) < 1080) {
          PaddedInput_shared[((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1815) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) / 225) * 225) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 121) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) / 15)) % 15) * 15)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) % 15)))] = (((((2 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 121) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) / 15)) % 15)) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 121) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) / 15)) % 15) < 13)) && (2 <= (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) % 15))) && ((((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) % 15) < 13)) ? data[(((((((((((int)blockIdx.x) / 224) * 650496) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1815) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) / 675) * 81312)) + ((((int)blockIdx.x) % 224) * 363)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 1815) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) % 675) / 225) * 121)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 121) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) / 15)) % 15) * 11)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) % 15)) - 24))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((int)threadIdx.x) < 75) {
    kernel_shared[(((int)threadIdx.x))] = kernel[((((((int)blockIdx.x) % 224) * 75) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int di_outer_inner = 0; di_outer_inner < 5; ++di_outer_inner) {
    for (int dj_outer_inner = 0; dj_outer_inner < 5; ++dj_outer_inner) {
      DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((((int)threadIdx.x) / 121) * 225) + (((((int)threadIdx.x) % 121) / 11) * 15)) + (di_outer_inner * 15)) + dj_outer_inner) + (((int)threadIdx.x) % 11)))] * kernel_shared[(((((((int)threadIdx.x) / 121) * 25) + (di_outer_inner * 5)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 121) * 225) + (((((int)threadIdx.x) % 121) / 11) * 15)) + (di_outer_inner * 15)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 675))] * kernel_shared[(((((((int)threadIdx.x) / 121) * 25) + (di_outer_inner * 5)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 121) * 225) + (((((int)threadIdx.x) % 121) / 11) * 15)) + (di_outer_inner * 15)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 1350))] * kernel_shared[(((((((int)threadIdx.x) / 121) * 25) + (di_outer_inner * 5)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 121) * 225) + (((((int)threadIdx.x) % 121) / 11) * 15)) + (di_outer_inner * 15)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 2025))] * kernel_shared[(((((((int)threadIdx.x) / 121) * 25) + (di_outer_inner * 5)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 121) * 225) + (((((int)threadIdx.x) % 121) / 11) * 15)) + (di_outer_inner * 15)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 2700))] * kernel_shared[(((((((int)threadIdx.x) / 121) * 25) + (di_outer_inner * 5)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 121) * 225) + (((((int)threadIdx.x) % 121) / 11) * 15)) + (di_outer_inner * 15)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 3375))] * kernel_shared[(((((((int)threadIdx.x) / 121) * 25) + (di_outer_inner * 5)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 121) * 225) + (((((int)threadIdx.x) % 121) / 11) * 15)) + (di_outer_inner * 15)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 4050))] * kernel_shared[(((((((int)threadIdx.x) / 121) * 25) + (di_outer_inner * 5)) + dj_outer_inner))]));
      DepthwiseConv2d_local[(7)] = (DepthwiseConv2d_local[(7)] + (PaddedInput_shared[((((((((((int)threadIdx.x) / 121) * 225) + (((((int)threadIdx.x) % 121) / 11) * 15)) + (di_outer_inner * 15)) + dj_outer_inner) + (((int)threadIdx.x) % 11)) + 4725))] * kernel_shared[(((((((int)threadIdx.x) / 121) * 25) + (di_outer_inner * 5)) + dj_outer_inner))]));
    }
  }
  DepthwiseConv2d[(((((((int)blockIdx.x) / 224) * 650496) + ((((int)blockIdx.x) % 224) * 363)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 224) * 650496) + ((((int)blockIdx.x) % 224) * 363)) + ((int)threadIdx.x)) + 81312))] = DepthwiseConv2d_local[(1)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 224) * 650496) + ((((int)blockIdx.x) % 224) * 363)) + ((int)threadIdx.x)) + 162624))] = DepthwiseConv2d_local[(2)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 224) * 650496) + ((((int)blockIdx.x) % 224) * 363)) + ((int)threadIdx.x)) + 243936))] = DepthwiseConv2d_local[(3)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 224) * 650496) + ((((int)blockIdx.x) % 224) * 363)) + ((int)threadIdx.x)) + 325248))] = DepthwiseConv2d_local[(4)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 224) * 650496) + ((((int)blockIdx.x) % 224) * 363)) + ((int)threadIdx.x)) + 406560))] = DepthwiseConv2d_local[(5)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 224) * 650496) + ((((int)blockIdx.x) % 224) * 363)) + ((int)threadIdx.x)) + 487872))] = DepthwiseConv2d_local[(6)];
  DepthwiseConv2d[((((((((int)blockIdx.x) / 224) * 650496) + ((((int)blockIdx.x) % 224) * 363)) + ((int)threadIdx.x)) + 569184))] = DepthwiseConv2d_local[(7)];
}

