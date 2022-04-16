//1_1_43008_21_3_1
//128_336_21_21_5_1_SAME
//dim3 grid(1, 1, 43008);
//dim3 block(21, 3, 1);

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
extern "C" __global__ void __launch_bounds__(63) depthwise_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ DepthwiseConv2d) {
  __shared__ float PaddedInput_shared[625];
  __shared__ float placeholder_shared[25];
  float PaddedInput_shared_local[55];
  float placeholder_shared_local[25];
  float DepthwiseConv2d_local[7];
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 63) + (((int)threadIdx.y) * 21)) + ((int)threadIdx.x)) < 625) {
      PaddedInput_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 63) + (((int)threadIdx.y) * 21)) + ((int)threadIdx.x)))] = (((((50 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 63) + (((int)threadIdx.y) * 21)) + ((int)threadIdx.x))) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 63) + (((int)threadIdx.y) * 21)) + ((int)threadIdx.x)) < 575)) && (2 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 63) + (((int)threadIdx.y) * 21)) + ((int)threadIdx.x)) % 25))) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 63) + (((int)threadIdx.y) * 21)) + ((int)threadIdx.x)) % 25) < 23)) ? placeholder[(((((((int)blockIdx.z) * 441) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 63) + (((int)threadIdx.y) * 21)) + ((int)threadIdx.x)) / 25) * 21)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 63) + (((int)threadIdx.y) * 21)) + ((int)threadIdx.x)) % 25)) - 44))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) < 25) {
    if (((int)threadIdx.y) < 2) {
      placeholder_shared[(((((int)threadIdx.y) * 21) + ((int)threadIdx.x)))] = placeholder1[(((((((int)blockIdx.z) % 336) * 25) + (((int)threadIdx.y) * 21)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax2 = 0; ax2 < 11; ++ax2) {
    #pragma unroll
    for (int ax3 = 0; ax3 < 5; ++ax3) {
      PaddedInput_shared_local[(((ax2 * 5) + ax3))] = PaddedInput_shared[(((((((int)threadIdx.y) * 175) + (ax2 * 25)) + ax3) + ((int)threadIdx.x)))];
    }
  }
  #pragma unroll
  for (int ax21 = 0; ax21 < 5; ++ax21) {
    #pragma unroll
    for (int ax31 = 0; ax31 < 5; ++ax31) {
      placeholder_shared_local[(((ax21 * 5) + ax31))] = placeholder_shared[(((ax21 * 5) + ax31))];
    }
  }
  #pragma unroll
  for (int i_c = 0; i_c < 7; ++i_c) {
    DepthwiseConv2d_local[(i_c)] = 0.000000e+00f;
    #pragma unroll
    for (int di = 0; di < 5; ++di) {
      #pragma unroll
      for (int dj = 0; dj < 5; ++dj) {
        DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[((((i_c * 5) + (di * 5)) + dj))] * placeholder_shared_local[(((di * 5) + dj))]));
      }
    }
  }
  #pragma unroll
  for (int i_inner_inner_inner = 0; i_inner_inner_inner < 7; ++i_inner_inner_inner) {
    DepthwiseConv2d[(((((((int)blockIdx.z) * 441) + (((int)threadIdx.y) * 147)) + (i_inner_inner_inner * 21)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(i_inner_inner_inner)];
  }
}

