//1_1_4096_14_2_8
//128_256_14_14_1024_1_1_SAME
//dim3 grid(1, 1, 4096);
//dim3 block(14, 2, 8);

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
  float compute_local[28];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[128];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((int)threadIdx.z) * 7) + ((((((int)threadIdx.y) * 49) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 14)) < 56) {
        if (((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 49)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 784) {
          if ((((((int)threadIdx.y) * 49) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 98) {
            if (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 49) {
              pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 49)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((((((int)blockIdx.z) >> 5) * 50176) + (rc_outer * 784)) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 49)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 32) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 128) {
        if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 16) {
          if (((int)threadIdx.x) < 8) {
            placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder1[((((((((((int)blockIdx.z) & 31) * 8192) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 256)) + (rc_outer * 4)) + (((int)threadIdx.x) & 3)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
    }
  }
  compute[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1568))] = compute_local[(7)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3136))] = compute_local[(14)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4704))] = compute_local[(21)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1596))] = compute_local[(8)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3164))] = compute_local[(15)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4732))] = compute_local[(22)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1624))] = compute_local[(9)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3192))] = compute_local[(16)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4760))] = compute_local[(23)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] = compute_local[(3)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1652))] = compute_local[(10)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3220))] = compute_local[(17)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4788))] = compute_local[(24)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] = compute_local[(4)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1680))] = compute_local[(11)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3248))] = compute_local[(18)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4816))] = compute_local[(25)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] = compute_local[(5)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1708))] = compute_local[(12)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3276))] = compute_local[(19)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4844))] = compute_local[(26)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] = compute_local[(6)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1736))] = compute_local[(13)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3304))] = compute_local[(20)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4872))] = compute_local[(27)];
}

