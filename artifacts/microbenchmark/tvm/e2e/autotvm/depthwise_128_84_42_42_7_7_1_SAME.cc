//1_1_10752_42_7_1
//128_84_42_42_7_1_SAME
//dim3 grid(1, 1, 10752);
//dim3 block(42, 7, 1);

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
extern "C" __global__ void __launch_bounds__(294) depthwise_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ DepthwiseConv2d) {
  __shared__ float PaddedInput_shared[2304];
  __shared__ float placeholder_shared[49];
  float PaddedInput_shared_local[126];
  float placeholder_shared_local[49];
  float DepthwiseConv2d_local[6];
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 294) + (((int)threadIdx.y) * 42)) + ((int)threadIdx.x)) < 2304) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 7) + ((int)threadIdx.y)) < 55) {
        PaddedInput_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 294) + (((int)threadIdx.y) * 42)) + ((int)threadIdx.x)))] = (((((144 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 294) + (((int)threadIdx.y) * 42)) + ((int)threadIdx.x))) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 294) + (((int)threadIdx.y) * 42)) + ((int)threadIdx.x)) < 2160)) && (3 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 294) + (((int)threadIdx.y) * 42)) + ((int)threadIdx.x)) % 48))) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 294) + (((int)threadIdx.y) * 42)) + ((int)threadIdx.x)) % 48) < 45)) ? placeholder[(((((((int)blockIdx.z) * 1764) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 294) + (((int)threadIdx.y) * 42)) + ((int)threadIdx.x)) / 48) * 42)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 294) + (((int)threadIdx.y) * 42)) + ((int)threadIdx.x)) % 48)) - 129))] : 0.000000e+00f);
      }
    }
  }
  if (((((int)threadIdx.y) * 6) + (((int)threadIdx.x) / 7)) < 7) {
    if (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) < 49) {
      if (((int)threadIdx.y) < 2) {
        placeholder_shared[(((((int)threadIdx.y) * 42) + ((int)threadIdx.x)))] = placeholder1[(((((((int)blockIdx.z) % 84) * 49) + (((int)threadIdx.y) * 42)) + ((int)threadIdx.x)))];
      }
    }
  }
  __syncthreads();
  for (int ax2 = 0; ax2 < 9; ++ax2) {
    for (int ax3 = 0; ax3 < 7; ++ax3) {
      PaddedInput_shared_local[(((ax2 * 7) + ax3))] = PaddedInput_shared[(((((((int)threadIdx.y) * 144) + (ax2 * 48)) + ax3) + ((int)threadIdx.x)))];
      PaddedInput_shared_local[((((ax2 * 7) + ax3) + 63))] = PaddedInput_shared[((((((((int)threadIdx.y) * 144) + (ax2 * 48)) + ax3) + ((int)threadIdx.x)) + 1008))];
    }
  }
  for (int ax21 = 0; ax21 < 7; ++ax21) {
    for (int ax31 = 0; ax31 < 7; ++ax31) {
      placeholder_shared_local[(((ax21 * 7) + ax31))] = placeholder_shared[(((ax21 * 7) + ax31))];
    }
  }
  for (int i_c = 0; i_c < 3; ++i_c) {
    DepthwiseConv2d_local[(i_c)] = 0.000000e+00f;
    DepthwiseConv2d_local[((i_c + 3))] = 0.000000e+00f;
    for (int di = 0; di < 7; ++di) {
      for (int dj = 0; dj < 7; ++dj) {
        DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[((((i_c * 7) + (di * 7)) + dj))] * placeholder_shared_local[(((di * 7) + dj))]));
        DepthwiseConv2d_local[((i_c + 3))] = (DepthwiseConv2d_local[((i_c + 3))] + (PaddedInput_shared_local[(((((i_c * 7) + (di * 7)) + dj) + 63))] * placeholder_shared_local[(((di * 7) + dj))]));
      }
    }
  }
  for (int i_inner_inner_inner = 0; i_inner_inner_inner < 3; ++i_inner_inner_inner) {
    DepthwiseConv2d[(((((((int)blockIdx.z) * 1764) + (((int)threadIdx.y) * 126)) + (i_inner_inner_inner * 42)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(i_inner_inner_inner)];
    DepthwiseConv2d[((((((((int)blockIdx.z) * 1764) + (((int)threadIdx.y) * 126)) + (i_inner_inner_inner * 42)) + ((int)threadIdx.x)) + 882))] = DepthwiseConv2d_local[((i_inner_inner_inner + 3))];
  }
}

