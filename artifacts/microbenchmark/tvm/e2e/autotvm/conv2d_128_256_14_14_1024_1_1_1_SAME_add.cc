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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[28];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[128];
  compute[(0)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
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
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + rc_inner))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 32))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 64))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[((((((int)threadIdx.z) * 4) + rc_inner) + 96))]));
    }
  }
  T_add[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = (compute[(0)] + input2[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1568))] = (compute[(7)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1568))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3136))] = (compute[(14)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3136))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4704))] = (compute[(21)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4704))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] = (compute[(1)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1596))] = (compute[(8)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1596))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3164))] = (compute[(15)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3164))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4732))] = (compute[(22)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4732))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] = (compute[(2)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1624))] = (compute[(9)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1624))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3192))] = (compute[(16)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3192))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4760))] = (compute[(23)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4760))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] = (compute[(3)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1652))] = (compute[(10)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1652))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3220))] = (compute[(17)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3220))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4788))] = (compute[(24)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4788))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] = (compute[(4)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1680))] = (compute[(11)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1680))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3248))] = (compute[(18)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3248))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4816))] = (compute[(25)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4816))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] = (compute[(5)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1708))] = (compute[(12)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1708))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3276))] = (compute[(19)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3276))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4844))] = (compute[(26)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4844))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] = (compute[(6)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1736))] = (compute[(13)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1736))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3304))] = (compute[(20)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 3304))]);
  T_add[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4872))] = (compute[(27)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4872))]);
}

