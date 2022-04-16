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
extern "C" __global__ void __launch_bounds__(112) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[28];
  __shared__ float pad_temp_shared[435];
  __shared__ float placeholder_shared[288];
  compute1[(0)] = 0.000000e+00f;
  compute1[(14)] = 0.000000e+00f;
  compute1[(1)] = 0.000000e+00f;
  compute1[(15)] = 0.000000e+00f;
  compute1[(2)] = 0.000000e+00f;
  compute1[(16)] = 0.000000e+00f;
  compute1[(3)] = 0.000000e+00f;
  compute1[(17)] = 0.000000e+00f;
  compute1[(4)] = 0.000000e+00f;
  compute1[(18)] = 0.000000e+00f;
  compute1[(5)] = 0.000000e+00f;
  compute1[(19)] = 0.000000e+00f;
  compute1[(6)] = 0.000000e+00f;
  compute1[(20)] = 0.000000e+00f;
  compute1[(7)] = 0.000000e+00f;
  compute1[(21)] = 0.000000e+00f;
  compute1[(8)] = 0.000000e+00f;
  compute1[(22)] = 0.000000e+00f;
  compute1[(9)] = 0.000000e+00f;
  compute1[(23)] = 0.000000e+00f;
  compute1[(10)] = 0.000000e+00f;
  compute1[(24)] = 0.000000e+00f;
  compute1[(11)] = 0.000000e+00f;
  compute1[(25)] = 0.000000e+00f;
  compute1[(12)] = 0.000000e+00f;
  compute1[(26)] = 0.000000e+00f;
  compute1[(13)] = 0.000000e+00f;
  compute1[(27)] = 0.000000e+00f;
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
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 144))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 153))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 145))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 154))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 146))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 155))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 145))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 145))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 203))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 203))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 147))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 145))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 145))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 203))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 203))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 261))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 156))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 146))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 146))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 204))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 204))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 320))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 320))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 148))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 146))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 146))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 204))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 204))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 262))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 320))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 320))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 157))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 205))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 205))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 321))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 321))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 379))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 379))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 149))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 205))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 205))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 263))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 321))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 321))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 379))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 379))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 158))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 150))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 58))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 116))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 174))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 232))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 290))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 348))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 159))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 151))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 59))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 117))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 233))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 291))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 349))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 160))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 408))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 408))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 152))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 118))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 234))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 292))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 408))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 408))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 161))]));
  }
  compute[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)))] = max((compute1[(0)] + input2[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3136))] = max((compute1[(14)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3136))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 14))] = max((compute1[(1)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 14))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3150))] = max((compute1[(15)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3150))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 28))] = max((compute1[(2)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 28))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3164))] = max((compute1[(16)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3164))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 42))] = max((compute1[(3)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 42))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3178))] = max((compute1[(17)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3178))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 56))] = max((compute1[(4)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 56))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3192))] = max((compute1[(18)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3192))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 70))] = max((compute1[(5)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 70))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3206))] = max((compute1[(19)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3206))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 84))] = max((compute1[(6)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 84))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3220))] = max((compute1[(20)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3220))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 196))] = max((compute1[(7)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 196))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3332))] = max((compute1[(21)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3332))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 210))] = max((compute1[(8)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 210))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3346))] = max((compute1[(22)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3346))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 224))] = max((compute1[(9)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 224))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3360))] = max((compute1[(23)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3360))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 238))] = max((compute1[(10)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 238))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3374))] = max((compute1[(24)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3374))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 252))] = max((compute1[(11)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 252))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3388))] = max((compute1[(25)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3388))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 266))] = max((compute1[(12)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 266))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3402))] = max((compute1[(26)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3402))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 280))] = max((compute1[(13)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 280))]), 0.000000e+00f);
  compute[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3416))] = max((compute1[(27)] + input2[((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3416))]), 0.000000e+00f);
}

