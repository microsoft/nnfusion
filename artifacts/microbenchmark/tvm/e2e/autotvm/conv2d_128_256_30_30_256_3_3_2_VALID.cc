//1_2_1024_14_1_8
//128_256_30_30_256_3_2_VALID
//dim3 grid(1, 2, 1024);
//dim3 block(14, 1, 8);

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
extern "C" __global__ void __launch_bounds__(112) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[28];
  __shared__ float pad_temp_shared[435];
  __shared__ float placeholder_shared[288];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 256; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) < 435) {
      pad_temp_shared[(((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 230400) + (rc_outer * 900)) + (((int)blockIdx.y) * 420)) + ((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) / 29) * 30)) + (((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) % 29)))];
    }
    if (((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) < 434) {
      pad_temp_shared[((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 230400) + (rc_outer * 900)) + (((int)blockIdx.y) * 420)) + (((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) + 1) / 29) * 30)) + ((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) + 1) % 29)))];
    }
    if (((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) < 433) {
      pad_temp_shared[((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 230400) + (rc_outer * 900)) + (((int)blockIdx.y) * 420)) + (((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) + 2) / 29) * 30)) + ((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) + 2) % 29)))];
    }
    if (((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) < 432) {
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 230400) + (rc_outer * 900)) + (((int)blockIdx.y) * 420)) + (((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) + 3) / 29) * 30)) + ((((((int)threadIdx.z) * 55) + (((int)threadIdx.x) * 4)) + 3) % 29)))];
      }
    }
    if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 12) + ((int)threadIdx.x)) < 96) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) < 288) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 73728) + (((int)threadIdx.z) * 9216)) + ((((int)threadIdx.x) / 3) * 2304)) + (rc_outer * 9)) + ((((int)threadIdx.x) % 3) * 3)))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 12) + ((int)threadIdx.x)) < 96) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) < 287) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[((((((((((int)blockIdx.z) & 7) * 73728) + (((int)threadIdx.z) * 9216)) + ((((int)threadIdx.x) / 3) * 2304)) + (rc_outer * 9)) + ((((int)threadIdx.x) % 3) * 3)) + 1))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 12) + ((int)threadIdx.x)) < 96) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) < 286) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[((((((((((int)blockIdx.z) & 7) * 73728) + (((int)threadIdx.z) * 9216)) + ((((int)threadIdx.x) / 3) * 2304)) + (rc_outer * 9)) + ((((int)threadIdx.x) % 3) * 3)) + 2))];
          }
        }
      }
    }
    __syncthreads();
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 145))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 145))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 203))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 203))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 145))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 145))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 203))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 203))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 146))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 146))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 204))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 204))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 320))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 320))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 146))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 146))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 204))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 204))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 320))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 320))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 205))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 205))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 321))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 321))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 379))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 379))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 205))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 205))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 321))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 321))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 379))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 379))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 408))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 408))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 408))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 408))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
  }
  compute[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3136))] = compute_local[(14)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 14))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3150))] = compute_local[(15)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 28))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3164))] = compute_local[(16)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 42))] = compute_local[(3)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3178))] = compute_local[(17)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 56))] = compute_local[(4)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3192))] = compute_local[(18)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 70))] = compute_local[(5)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3206))] = compute_local[(19)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 84))] = compute_local[(6)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3220))] = compute_local[(20)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 196))] = compute_local[(7)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3332))] = compute_local[(21)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 210))] = compute_local[(8)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3346))] = compute_local[(22)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 224))] = compute_local[(9)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3360))] = compute_local[(23)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 238))] = compute_local[(10)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3374))] = compute_local[(24)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 252))] = compute_local[(11)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3388))] = compute_local[(25)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 266))] = compute_local[(12)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3402))] = compute_local[(26)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 280))] = compute_local[(13)];
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3416))] = compute_local[(27)];
}

