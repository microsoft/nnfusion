//1_83_5376_83_1_1
//128_42_83_83_7_1_SAME
//dim3 grid(1, 83, 5376);
//dim3 block(83, 1, 1);

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
extern "C" __global__ void __launch_bounds__(83) depthwise_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ DepthwiseConv2d) {
  __shared__ float PaddedInput_shared[623];
  __shared__ float placeholder_shared[49];
  float PaddedInput_shared_local[49];
  float placeholder_shared_local[49];
  float DepthwiseConv2d_local[1];
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 83) + ((int)threadIdx.x)) < 623) {
      PaddedInput_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 83) + ((int)threadIdx.x)))] = (((((3 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 83) + ((int)threadIdx.x)) / 89) + ((int)blockIdx.y))) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 83) + ((int)threadIdx.x)) / 89) + ((int)blockIdx.y)) < 86)) && (3 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 83) + ((int)threadIdx.x)) % 89))) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 83) + ((int)threadIdx.x)) % 89) < 86)) ? placeholder[((((((((int)blockIdx.z) * 6889) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 83) + ((int)threadIdx.x)) / 89) * 83)) + (((int)blockIdx.y) * 83)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 83) + ((int)threadIdx.x)) % 89)) - 252))] : 0.000000e+00f);
    }
  }
  if (((int)threadIdx.x) < 49) {
    placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((((int)blockIdx.z) % 42) * 49) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  #pragma unroll
  for (int ax2 = 0; ax2 < 7; ++ax2) {
    #pragma unroll
    for (int ax3 = 0; ax3 < 7; ++ax3) {
      PaddedInput_shared_local[(((ax2 * 7) + ax3))] = PaddedInput_shared[((((ax2 * 89) + ax3) + ((int)threadIdx.x)))];
    }
  }
  #pragma unroll
  for (int ax21 = 0; ax21 < 7; ++ax21) {
    #pragma unroll
    for (int ax31 = 0; ax31 < 7; ++ax31) {
      placeholder_shared_local[(((ax21 * 7) + ax31))] = placeholder_shared[(((ax21 * 7) + ax31))];
    }
  }
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  #pragma unroll
  for (int di = 0; di < 7; ++di) {
    #pragma unroll
    for (int dj = 0; dj < 7; ++dj) {
      DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(((di * 7) + dj))] * placeholder_shared_local[(((di * 7) + dj))]));
    }
  }
  DepthwiseConv2d[((((((int)blockIdx.z) * 6889) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
}

