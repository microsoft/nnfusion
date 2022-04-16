//1_1_1024_7_1_16
//128_512_16_16_512_3_2_VALID
//dim3 grid(1, 1, 1024);
//dim3 block(7, 1, 16);

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
  __shared__ float pad_temp_shared[450];
  __shared__ float placeholder_shared[1152];
  compute1[(0)] = 0.000000e+00f;
  compute1[(7)] = 0.000000e+00f;
  compute1[(14)] = 0.000000e+00f;
  compute1[(21)] = 0.000000e+00f;
  compute1[(1)] = 0.000000e+00f;
  compute1[(8)] = 0.000000e+00f;
  compute1[(15)] = 0.000000e+00f;
  compute1[(22)] = 0.000000e+00f;
  compute1[(2)] = 0.000000e+00f;
  compute1[(9)] = 0.000000e+00f;
  compute1[(16)] = 0.000000e+00f;
  compute1[(23)] = 0.000000e+00f;
  compute1[(3)] = 0.000000e+00f;
  compute1[(10)] = 0.000000e+00f;
  compute1[(17)] = 0.000000e+00f;
  compute1[(24)] = 0.000000e+00f;
  compute1[(4)] = 0.000000e+00f;
  compute1[(11)] = 0.000000e+00f;
  compute1[(18)] = 0.000000e+00f;
  compute1[(25)] = 0.000000e+00f;
  compute1[(5)] = 0.000000e+00f;
  compute1[(12)] = 0.000000e+00f;
  compute1[(19)] = 0.000000e+00f;
  compute1[(26)] = 0.000000e+00f;
  compute1[(6)] = 0.000000e+00f;
  compute1[(13)] = 0.000000e+00f;
  compute1[(20)] = 0.000000e+00f;
  compute1[(27)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 256; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) < 450) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[(((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 131072) + (rc_outer * 512)) + ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) / 225) * 256)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) % 225) / 15) * 16)) + (((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) % 15)))];
      }
    }
    if (((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) < 449) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 1))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 131072) + (rc_outer * 512)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 1) / 225) * 256)) + ((((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 1) % 225) / 15) * 16)) + ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 1) % 15)))];
      }
    }
    if (((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) < 448) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 2))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 131072) + (rc_outer * 512)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 2) / 225) * 256)) + ((((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 2) % 225) / 15) * 16)) + ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 2) % 15)))];
      }
    }
    if (((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) < 447) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 3))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 131072) + (rc_outer * 512)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 3) / 225) * 256)) + ((((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 3) % 225) / 15) * 16)) + ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 3) % 15)))];
      }
    }
    if (((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) < 446) {
      if (((int)threadIdx.x) < 5) {
        pad_temp_shared[((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 4))] = placeholder[(((((((((int)blockIdx.z) >> 3) * 131072) + (rc_outer * 512)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 4) / 225) * 256)) + ((((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 4) % 225) / 15) * 16)) + ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 5)) + 4) % 15)))];
      }
    }
    placeholder_shared[(((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 11) / 18) * 4608)) + (rc_outer * 18)) + ((((int)threadIdx.x) * 11) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 1))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 1) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 1) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 2))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 2) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 2) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 3))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 3) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 3) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 4))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 4) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 4) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 5))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 5) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 5) % 18)))];
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 11) + 6) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 11) + 6) / 9)) < 128) {
        if (((((int)threadIdx.z) * 24) + ((((int)threadIdx.x) * 11) / 3)) < 382) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) < 1146) {
            if (((int)threadIdx.x) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 6))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 6) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 6) % 18)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 11) + 7) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 11) + 7) / 9)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((((int)threadIdx.x) * 11) + 7) / 3)) < 384) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) < 1145) {
            if (((int)threadIdx.x) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 7))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 7) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 7) % 18)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 11) + 8) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 11) + 8) / 9)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((((int)threadIdx.x) * 11) + 8) / 3)) < 384) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) < 1144) {
            if (((int)threadIdx.x) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 8))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 8) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 8) % 18)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 11) + 9) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 11) / 9)) < 127) {
        if (((((int)threadIdx.z) * 24) + ((((int)threadIdx.x) * 11) / 3)) < 381) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) < 1143) {
            if (((int)threadIdx.x) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 9))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 9) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 9) % 18)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 11) + 10) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 11) + 10) / 9)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((((int)threadIdx.x) * 11) + 10) / 3)) < 384) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) < 1142) {
            if (((int)threadIdx.x) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 11)) + 10))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 11) + 10) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 11) + 10) % 18)))];
            }
          }
        }
      }
    }
    __syncthreads();
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 288))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 576))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 864))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 288))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 576))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 864))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 288))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 576))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 864))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 90))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 288))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 576))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 864))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 120))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 120))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 288))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 120))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 576))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 120))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 864))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 150))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 150))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 288))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 150))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 576))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 150))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 864))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 180))] * placeholder_shared[((((int)threadIdx.z) * 18))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 180))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 288))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 180))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 576))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 180))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 864))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 289))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 577))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 865))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 289))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 577))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 865))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 61))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 61))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 289))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 61))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 577))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 61))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 865))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 289))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 577))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 865))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 289))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 577))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 865))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 151))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 151))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 289))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 151))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 577))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 151))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 865))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 181))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 1))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 181))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 289))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 181))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 577))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 181))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 865))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 290))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 578))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 866))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 290))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 578))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 866))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 62))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 62))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 290))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 62))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 578))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 62))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 866))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 290))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 578))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 866))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 122))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 122))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 290))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 122))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 578))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 122))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 866))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 152))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 152))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 290))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 152))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 578))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 152))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 866))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 2))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 290))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 578))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 866))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 291))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 579))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 867))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 45))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 45))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 291))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 45))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 579))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 45))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 867))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 75))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 75))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 291))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 75))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 579))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 75))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 867))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 291))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 579))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 867))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 291))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 579))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 867))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 291))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 579))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 867))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 3))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 291))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 579))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 867))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 292))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 580))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 868))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 46))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 46))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 292))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 46))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 580))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 46))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 868))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 76))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 76))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 292))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 76))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 580))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 76))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 868))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 292))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 580))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 868))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 136))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 136))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 292))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 136))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 580))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 136))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 868))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 166))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 166))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 292))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 166))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 580))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 166))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 868))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 4))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 292))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 580))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 868))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 293))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 581))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 869))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 47))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 47))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 293))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 47))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 581))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 47))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 869))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 293))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 581))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 869))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 107))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 107))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 293))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 107))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 581))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 107))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 869))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 137))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 137))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 293))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 137))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 581))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 137))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 869))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 167))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 167))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 293))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 167))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 581))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 167))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 869))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 5))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 293))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 581))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 869))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 294))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 582))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 870))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 294))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 582))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 870))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 294))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 582))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 870))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 120))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 120))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 294))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 120))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 582))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 120))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 870))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 150))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 150))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 294))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 150))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 582))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 150))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 870))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 180))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 180))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 294))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 180))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 582))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 180))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 870))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 6))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 294))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 582))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 870))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 295))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 583))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 871))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 61))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 61))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 295))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 61))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 583))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 61))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 871))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 295))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 583))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 871))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 295))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 583))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 871))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 151))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 151))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 295))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 151))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 583))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 151))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 871))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 181))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 181))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 295))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 181))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 583))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 181))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 871))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 7))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 295))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 583))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 871))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 296))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 584))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 872))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 62))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 62))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 296))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 62))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 584))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 62))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 872))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 296))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 584))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 872))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 122))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 122))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 296))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 122))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 584))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 122))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 872))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 152))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 152))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 296))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 152))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 584))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 152))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 872))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 296))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 584))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 872))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 8))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 296))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 584))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 212))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 872))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 297))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 585))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 873))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 297))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 585))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 873))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 297))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 585))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 873))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 297))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 585))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 873))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 345))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 345))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 297))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 345))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 585))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 345))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 873))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 375))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 375))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 297))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 375))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 585))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 375))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 873))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 9))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 297))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 585))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 873))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 298))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 586))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 226))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 874))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 298))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 586))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 874))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 298))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 586))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 874))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 316))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 316))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 298))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 316))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 586))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 316))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 874))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 346))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 346))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 298))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 346))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 586))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 346))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 874))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 376))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 376))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 298))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 376))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 586))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 376))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 874))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 10))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 298))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 586))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 874))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 299))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 587))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 227))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 875))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 299))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 587))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 875))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 299))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 587))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 875))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 317))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 317))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 299))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 317))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 587))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 317))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 875))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 347))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 347))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 299))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 347))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 587))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 347))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 875))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 299))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 587))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 875))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 11))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 299))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 587))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 875))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 300))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 588))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 240))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 876))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 270))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 270))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 300))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 270))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 588))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 270))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 876))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 300))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 300))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 300))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 300))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 588))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 300))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 876))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 300))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 588))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 876))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 360))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 360))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 300))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 360))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 588))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 360))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 876))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 300))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 588))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 876))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 12))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 300))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 588))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 876))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 301))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 589))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 241))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 877))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 271))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 271))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 301))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 271))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 589))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 271))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 877))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 301))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 301))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 301))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 301))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 589))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 301))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 877))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 331))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 331))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 301))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 331))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 589))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 331))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 877))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 361))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 361))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 301))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 361))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 589))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 361))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 877))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 391))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 391))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 301))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 391))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 589))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 391))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 877))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 421))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 13))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 421))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 301))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 421))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 589))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 421))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 877))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 302))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 590))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 878))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 272))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 272))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 302))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 272))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 590))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 272))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 878))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 302))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 302))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 302))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 302))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 590))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 302))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 878))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 332))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 332))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 302))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 332))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 590))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 332))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 878))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 362))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 362))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 302))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 362))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 590))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 362))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 878))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 302))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 590))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 878))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 422))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 14))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 422))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 302))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 422))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 590))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 422))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 878))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 303))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 591))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 255))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 879))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 303))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 591))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 285))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 879))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 303))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 591))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 315))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 879))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 345))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 345))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 303))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 345))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 591))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 345))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 879))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 375))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 375))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 303))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 375))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 591))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 375))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 879))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 303))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 591))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 405))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 879))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 435))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 15))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 435))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 303))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 435))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 591))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 435))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 879))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 304))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 592))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 256))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 880))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 304))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 592))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 880))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 316))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 316))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 304))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 316))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 592))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 316))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 880))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 346))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 346))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 304))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 346))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 592))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 346))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 880))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 376))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 376))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 304))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 376))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 592))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 376))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 880))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 304))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 592))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 880))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 436))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 16))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 436))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 304))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 436))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 592))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 436))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 880))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 305))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 593))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 257))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 881))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 305))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 593))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 287))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 881))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 317))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 317))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 305))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 317))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 593))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 317))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 881))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 347))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 347))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 305))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 347))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 593))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 347))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 881))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 305))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 593))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 377))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 881))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 305))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 593))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 881))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 437))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 17))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 437))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 305))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 437))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 593))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 437))] * placeholder_shared[(((((int)threadIdx.z) * 18) + 881))]));
  }
  compute[((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)))] = max((compute1[(0)] + input2[((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 784))] = max((compute1[(7)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 784))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1568))] = max((compute1[(14)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1568))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2352))] = max((compute1[(21)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2352))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 7))] = max((compute1[(1)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 7))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 791))] = max((compute1[(8)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 791))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1575))] = max((compute1[(15)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1575))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2359))] = max((compute1[(22)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2359))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 14))] = max((compute1[(2)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 14))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 798))] = max((compute1[(9)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 798))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1582))] = max((compute1[(16)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1582))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2366))] = max((compute1[(23)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2366))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 21))] = max((compute1[(3)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 21))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 805))] = max((compute1[(10)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 805))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1589))] = max((compute1[(17)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1589))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2373))] = max((compute1[(24)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2373))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 28))] = max((compute1[(4)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 28))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 812))] = max((compute1[(11)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 812))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1596))] = max((compute1[(18)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1596))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2380))] = max((compute1[(25)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2380))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 35))] = max((compute1[(5)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 35))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 819))] = max((compute1[(12)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 819))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1603))] = max((compute1[(19)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1603))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2387))] = max((compute1[(26)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2387))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 42))] = max((compute1[(6)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 42))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 826))] = max((compute1[(13)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 826))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1610))] = max((compute1[(20)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 1610))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2394))] = max((compute1[(27)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 49)) + ((int)threadIdx.x)) + 2394))]), 0.000000e+00f);
}

