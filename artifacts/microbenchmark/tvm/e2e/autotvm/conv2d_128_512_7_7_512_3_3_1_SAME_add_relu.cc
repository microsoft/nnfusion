//1_1_1024_1_7_16
//128_512_7_7_512_3_1_SAME
//dim3 grid(1, 1, 1024);
//dim3 block(1, 7, 16);

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
  __shared__ float pad_temp_shared[162];
  __shared__ float placeholder_shared[1152];
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
    if (((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) < 162) {
      if (((int)threadIdx.y) < 6) {
        pad_temp_shared[(((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)))] = (((((9 <= (((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) % 81)) && ((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) % 81) < 72)) && (1 <= (((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) % 9))) && ((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) % 9) < 8)) ? placeholder[((((((((((int)blockIdx.z) >> 3) * 25088) + (rc_outer * 98)) + ((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) / 81) * 49)) + (((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) % 81) / 9) * 7)) + (((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) % 9)) - 8))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) < 161) {
      if (((int)threadIdx.y) < 5) {
        pad_temp_shared[((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) + 1))] = (((((9 <= ((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) + 1) % 81)) && (((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) + 1) % 81) < 72)) && (1 <= ((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) + 1) % 9))) && (((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) + 1) % 9) < 8)) ? placeholder[((((((((((int)blockIdx.z) >> 3) * 25088) + (rc_outer * 98)) + (((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) + 1) / 81) * 49)) + ((((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) + 1) % 81) / 9) * 7)) + ((((((int)threadIdx.z) * 11) + (((int)threadIdx.y) * 2)) + 1) % 9)) - 8))] : 0.000000e+00f);
      }
    }
    placeholder_shared[(((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.y) * 11) / 18) * 4608)) + (rc_outer * 18)) + ((((int)threadIdx.y) * 11) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 1))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 1) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 1) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 2))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 2) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 2) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 3))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 3) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 3) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 4))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 4) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 4) % 18)))];
    placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 5))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 5) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 5) % 18)))];
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 11) + 6) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 11) + 6) / 9)) < 128) {
        if (((((int)threadIdx.z) * 24) + ((((int)threadIdx.y) * 11) / 3)) < 382) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) < 1146) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 6))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 6) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 6) % 18)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 11) + 7) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 11) + 7) / 9)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((((int)threadIdx.y) * 11) + 7) / 3)) < 384) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) < 1145) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 7))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 7) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 7) % 18)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 11) + 8) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 11) + 8) / 9)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((((int)threadIdx.y) * 11) + 8) / 3)) < 384) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) < 1144) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 8))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 8) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 8) % 18)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 11) + 9) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.y) * 11) / 9)) < 127) {
        if (((((int)threadIdx.z) * 24) + ((((int)threadIdx.y) * 11) / 3)) < 381) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) < 1143) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 9))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 9) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 9) % 18)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 11) + 10) / 18)) < 64) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.y) * 11) + 10) / 9)) < 128) {
        if (((((int)threadIdx.z) * 24) + (((((int)threadIdx.y) * 11) + 10) / 3)) < 384) {
          if (((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) < 1142) {
            if (((int)threadIdx.y) < 6) {
              placeholder_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + 10))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 294912) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.y) * 11) + 10) / 18) * 4608)) + (rc_outer * 18)) + (((((int)threadIdx.y) * 11) + 10) % 18)))];
            }
          }
        }
      }
    }
    __syncthreads();
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[((((int)threadIdx.y) * 9))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((int)threadIdx.y) * 9))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 576))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 1))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 576))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 576))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 576))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 576))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 576))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 576))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[((((int)threadIdx.y) * 9))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[((((int)threadIdx.y) * 9))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 594))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 594))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 594))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 594))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 594))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 594))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 594))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 577))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 577))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 577))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 577))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 577))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 577))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 577))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 595))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 595))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 595))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 595))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 595))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 595))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 595))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 578))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 578))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 578))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 578))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 578))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 578))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 578))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 596))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 596))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 596))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 596))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 596))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 596))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 8))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 596))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 579))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 579))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 579))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 579))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 579))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 579))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 579))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 9))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 597))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 597))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 597))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 597))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 597))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 597))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 597))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 580))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 580))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 580))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 580))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 580))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 580))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 580))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 10))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 598))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 598))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 598))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 598))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 598))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 598))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 598))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 581))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 581))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 581))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 581))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 581))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 581))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 581))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 599))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 12))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 599))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 13))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 599))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 599))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 599))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 599))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 599))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 582))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 582))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 582))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 582))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 582))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 582))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 582))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 18))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 600))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 600))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 600))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 600))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 600))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 600))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 600))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 583))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 583))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 583))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 583))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 583))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 583))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 583))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 19))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 601))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 601))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 601))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 601))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 601))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 601))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 601))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 584))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 584))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 584))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 584))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 584))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 584))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 584))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 20))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 602))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 602))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 602))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 23))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 602))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 24))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 602))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 25))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 602))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 602))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 585))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 585))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 585))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 585))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 585))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 585))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 585))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 603))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 603))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 603))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 603))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 603))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 603))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 603))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 586))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 586))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 586))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 586))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 586))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 586))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 586))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 82))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 604))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 604))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 604))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 604))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 604))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 604))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 604))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 587))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 587))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 587))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 587))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 587))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 587))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 587))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 83))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 605))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 605))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 85))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 605))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 86))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 605))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 87))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 605))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 605))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 89))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 605))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 588))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 588))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 588))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 588))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 588))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 588))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 588))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 606))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 606))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 606))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 606))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 606))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 606))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 606))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 589))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 589))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 589))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 589))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 589))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 589))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 589))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 607))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 607))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 607))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 607))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 607))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 607))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 607))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 590))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 590))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 590))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 590))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 590))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 590))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 590))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 608))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 93))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 608))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 94))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 608))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 95))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 608))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 96))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 608))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 97))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 608))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 608))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 591))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 100))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 100))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 591))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 591))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 591))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 591))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 591))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 591))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 609))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 100))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 100))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 609))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 609))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 609))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 609))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 609))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 609))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 100))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 100))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 592))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 592))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 592))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 592))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 592))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 592))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 592))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 100))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 100))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 610))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 610))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 610))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 610))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 610))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 610))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 610))]));
    compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute1[(14)] = (compute1[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 593))]));
    compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute1[(15)] = (compute1[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 593))]));
    compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute1[(16)] = (compute1[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 593))]));
    compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute1[(17)] = (compute1[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 593))]));
    compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute1[(18)] = (compute1[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 593))]));
    compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute1[(19)] = (compute1[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 593))]));
    compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 107))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
    compute1[(20)] = (compute1[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 107))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 593))]));
    compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute1[(21)] = (compute1[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 101))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 611))]));
    compute1[(8)] = (compute1[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute1[(22)] = (compute1[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 102))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 611))]));
    compute1[(9)] = (compute1[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute1[(23)] = (compute1[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 103))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 611))]));
    compute1[(10)] = (compute1[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute1[(24)] = (compute1[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 611))]));
    compute1[(11)] = (compute1[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute1[(25)] = (compute1[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 611))]));
    compute1[(12)] = (compute1[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute1[(26)] = (compute1[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 611))]));
    compute1[(13)] = (compute1[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 107))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    compute1[(27)] = (compute1[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 9) + 107))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 611))]));
  }
  compute[((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)))] = max((compute1[(0)] + input2[((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1568))] = max((compute1[(14)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1568))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1))] = max((compute1[(1)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1569))] = max((compute1[(15)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1569))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 2))] = max((compute1[(2)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 2))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1570))] = max((compute1[(16)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1570))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 3))] = max((compute1[(3)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 3))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1571))] = max((compute1[(17)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1571))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 4))] = max((compute1[(4)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 4))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1572))] = max((compute1[(18)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1572))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 5))] = max((compute1[(5)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 5))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1573))] = max((compute1[(19)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1573))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 6))] = max((compute1[(6)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 6))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1574))] = max((compute1[(20)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1574))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 49))] = max((compute1[(7)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 49))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1617))] = max((compute1[(21)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1617))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 50))] = max((compute1[(8)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 50))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1618))] = max((compute1[(22)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1618))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 51))] = max((compute1[(9)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 51))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1619))] = max((compute1[(23)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1619))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 52))] = max((compute1[(10)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 52))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1620))] = max((compute1[(24)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1620))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 53))] = max((compute1[(11)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 53))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1621))] = max((compute1[(25)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1621))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 54))] = max((compute1[(12)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 54))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1622))] = max((compute1[(26)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1622))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 55))] = max((compute1[(13)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 55))]), 0.000000e+00f);
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1623))] = max((compute1[(27)] + input2[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 7)) + 1623))]), 0.000000e+00f);
}

