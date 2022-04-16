//7_7_256_16_2_4
//128_3_230_230_64_7_2_VALID
//dim3 grid(7, 7, 256);
//dim3 block(16, 2, 4);

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
extern "C" __global__ void __launch_bounds__(128) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[64];
  __shared__ float pad_temp_shared[1369];
  __shared__ float placeholder_shared[1568];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(32)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(36)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(40)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(44)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(48)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(52)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(56)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(60)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(33)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(37)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(41)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(45)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(49)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(53)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(57)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(61)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(34)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(38)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(42)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(46)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(50)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(54)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(58)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(62)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(35)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(39)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(43)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(47)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(51)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(55)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(59)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  compute_local[(63)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) / 37) * 230)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) % 37)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 1))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 1) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 1) % 37)))];
    pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 2))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 2) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 2) % 37)))];
    if ((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) < 1366) {
      pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 3))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 3) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 3) % 37)))];
    }
    if ((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) < 1365) {
      pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 4))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 4) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 4) % 37)))];
    }
    if ((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) < 1364) {
      pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 5))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 5) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 5) % 37)))];
    }
    if ((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) < 1363) {
      if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 11)) < 337) {
        pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 6))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 6) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 6) % 37)))];
      }
    }
    if ((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) < 1362) {
      if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 11)) < 336) {
        if (((int)threadIdx.x) < 15) {
          pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 7))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 7) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 7) % 37)))];
        }
      }
    }
    if ((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) < 1361) {
      if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 11)) < 335) {
        if (((int)threadIdx.x) < 15) {
          pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 8))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 8) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 8) % 37)))];
        }
      }
    }
    if ((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) < 1360) {
      if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 11)) < 334) {
        if (((int)threadIdx.x) < 15) {
          pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 9))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 9) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 9) % 37)))];
        }
      }
    }
    if ((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) < 1359) {
      if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 11)) < 333) {
        if (((int)threadIdx.x) < 15) {
          pad_temp_shared[(((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 10))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 158700) + (rc_outer * 52900)) + (((int)blockIdx.y) * 7360)) + ((((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 10) / 37) * 230)) + (((int)blockIdx.x) * 32)) + (((((((int)threadIdx.z) * 343) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 11)) + 10) % 37)))];
        }
      }
    }
    placeholder_shared[((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + (((((int)threadIdx.x) * 13) / 49) * 147)) + (rc_outer * 49)) + ((((int)threadIdx.x) * 13) % 49)))];
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 1) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 1) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1567) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 391) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 1))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 1) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 1) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 2) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 2) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1566) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 390) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 2))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 2) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 2) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 3) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 3) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1565) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 389) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 3))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 3) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 3) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 4) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 4) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1564) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 388) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 4))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 4) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 4) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 5) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 5) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1563) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 387) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 5))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 5) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 5) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 6) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 6) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1562) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 386) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 6))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 6) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 6) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 7) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + ((((int)threadIdx.x) * 13) / 7)) < 223) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1561) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 385) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 7))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 7) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 7) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 8) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 8) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1560) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 384) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 8))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 8) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 8) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 9) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 9) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1559) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 383) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 9))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 9) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 9) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 10) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 10) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1558) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 382) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 10))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 10) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 10) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 11) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 11) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1557) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 381) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 11))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 11) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 11) % 49)))];
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 13) + 12) / 49)) < 32) {
      if ((((((int)threadIdx.z) * 56) + (((int)threadIdx.y) * 28)) + (((((int)threadIdx.x) * 13) + 12) / 7)) < 224) {
        if ((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) < 1556) {
          if (((((int)threadIdx.y) * 196) + (((int)threadIdx.x) * 13)) < 380) {
            if (((int)threadIdx.x) < 15) {
              placeholder_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 13)) + 12))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 4704) + (((int)threadIdx.z) * 1176)) + (((int)threadIdx.y) * 588)) + ((((((int)threadIdx.x) * 13) + 12) / 49) * 147)) + (rc_outer * 49)) + (((((int)threadIdx.x) * 13) + 12) % 49)))];
            }
          }
        }
      }
    }
    __syncthreads();
    for (int ry_inner = 0; ry_inner < 7; ++ry_inner) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[(((((int)threadIdx.z) * 196) + (ry_inner * 7)))]));
      compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 784))]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 148))] * placeholder_shared[(((((int)threadIdx.z) * 196) + (ry_inner * 7)))]));
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 148))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 784))]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 296))] * placeholder_shared[(((((int)threadIdx.z) * 196) + (ry_inner * 7)))]));
      compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 296))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 784))]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 444))] * placeholder_shared[(((((int)threadIdx.z) * 196) + (ry_inner * 7)))]));
      compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 444))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 784))]));
      compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 592))] * placeholder_shared[(((((int)threadIdx.z) * 196) + (ry_inner * 7)))]));
      compute_local[(48)] = (compute_local[(48)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 592))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 784))]));
      compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 740))] * placeholder_shared[(((((int)threadIdx.z) * 196) + (ry_inner * 7)))]));
      compute_local[(52)] = (compute_local[(52)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 740))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 784))]));
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 888))] * placeholder_shared[(((((int)threadIdx.z) * 196) + (ry_inner * 7)))]));
      compute_local[(56)] = (compute_local[(56)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 888))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 784))]));
      compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1036))] * placeholder_shared[(((((int)threadIdx.z) * 196) + (ry_inner * 7)))]));
      compute_local[(60)] = (compute_local[(60)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1036))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 784))]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 49))]));
      compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 833))]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 148))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 49))]));
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 148))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 833))]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 296))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 49))]));
      compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 296))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 833))]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 444))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 49))]));
      compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 444))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 833))]));
      compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 592))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 49))]));
      compute_local[(49)] = (compute_local[(49)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 592))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 833))]));
      compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 740))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 49))]));
      compute_local[(53)] = (compute_local[(53)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 740))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 833))]));
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 888))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 49))]));
      compute_local[(57)] = (compute_local[(57)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 888))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 833))]));
      compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1036))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 49))]));
      compute_local[(61)] = (compute_local[(61)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1036))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 833))]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 98))]));
      compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 882))]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 148))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 98))]));
      compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 148))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 882))]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 296))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 98))]));
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 296))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 882))]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 444))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 98))]));
      compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 444))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 882))]));
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 592))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 98))]));
      compute_local[(50)] = (compute_local[(50)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 592))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 882))]));
      compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 740))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 98))]));
      compute_local[(54)] = (compute_local[(54)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 740))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 882))]));
      compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 888))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 98))]));
      compute_local[(58)] = (compute_local[(58)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 888))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 882))]));
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1036))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 98))]));
      compute_local[(62)] = (compute_local[(62)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1036))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 882))]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 147))]));
      compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 931))]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 148))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 147))]));
      compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 148))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 931))]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 296))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 147))]));
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 296))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 931))]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 444))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 147))]));
      compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 444))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 931))]));
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 592))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 147))]));
      compute_local[(51)] = (compute_local[(51)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 592))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 931))]));
      compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 740))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 147))]));
      compute_local[(55)] = (compute_local[(55)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 740))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 931))]));
      compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 888))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 147))]));
      compute_local[(59)] = (compute_local[(59)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 888))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 931))]));
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1036))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 147))]));
      compute_local[(63)] = (compute_local[(63)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1036))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 931))]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 1))]));
      compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 785))]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 149))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 1))]));
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 149))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 785))]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 297))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 1))]));
      compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 297))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 785))]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 445))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 1))]));
      compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 445))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 785))]));
      compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 593))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 1))]));
      compute_local[(48)] = (compute_local[(48)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 593))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 785))]));
      compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 741))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 1))]));
      compute_local[(52)] = (compute_local[(52)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 741))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 785))]));
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 889))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 1))]));
      compute_local[(56)] = (compute_local[(56)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 889))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 785))]));
      compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1037))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 1))]));
      compute_local[(60)] = (compute_local[(60)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1037))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 785))]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 50))]));
      compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 834))]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 149))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 50))]));
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 149))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 834))]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 297))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 50))]));
      compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 297))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 834))]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 445))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 50))]));
      compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 445))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 834))]));
      compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 593))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 50))]));
      compute_local[(49)] = (compute_local[(49)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 593))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 834))]));
      compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 741))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 50))]));
      compute_local[(53)] = (compute_local[(53)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 741))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 834))]));
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 889))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 50))]));
      compute_local[(57)] = (compute_local[(57)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 889))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 834))]));
      compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1037))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 50))]));
      compute_local[(61)] = (compute_local[(61)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1037))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 834))]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 99))]));
      compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 883))]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 149))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 99))]));
      compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 149))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 883))]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 297))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 99))]));
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 297))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 883))]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 445))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 99))]));
      compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 445))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 883))]));
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 593))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 99))]));
      compute_local[(50)] = (compute_local[(50)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 593))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 883))]));
      compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 741))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 99))]));
      compute_local[(54)] = (compute_local[(54)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 741))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 883))]));
      compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 889))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 99))]));
      compute_local[(58)] = (compute_local[(58)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 889))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 883))]));
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1037))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 99))]));
      compute_local[(62)] = (compute_local[(62)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1037))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 883))]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 148))]));
      compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 932))]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 149))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 148))]));
      compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 149))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 932))]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 297))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 148))]));
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 297))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 932))]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 445))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 148))]));
      compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 445))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 932))]));
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 593))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 148))]));
      compute_local[(51)] = (compute_local[(51)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 593))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 932))]));
      compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 741))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 148))]));
      compute_local[(55)] = (compute_local[(55)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 741))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 932))]));
      compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 889))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 148))]));
      compute_local[(59)] = (compute_local[(59)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 889))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 932))]));
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1037))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 148))]));
      compute_local[(63)] = (compute_local[(63)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1037))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 932))]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 2))]));
      compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 786))]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 150))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 2))]));
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 150))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 786))]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 298))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 2))]));
      compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 298))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 786))]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 446))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 2))]));
      compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 446))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 786))]));
      compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 594))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 2))]));
      compute_local[(48)] = (compute_local[(48)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 594))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 786))]));
      compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 742))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 2))]));
      compute_local[(52)] = (compute_local[(52)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 742))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 786))]));
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 890))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 2))]));
      compute_local[(56)] = (compute_local[(56)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 890))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 786))]));
      compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1038))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 2))]));
      compute_local[(60)] = (compute_local[(60)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1038))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 786))]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 51))]));
      compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 835))]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 150))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 51))]));
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 150))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 835))]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 298))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 51))]));
      compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 298))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 835))]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 446))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 51))]));
      compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 446))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 835))]));
      compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 594))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 51))]));
      compute_local[(49)] = (compute_local[(49)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 594))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 835))]));
      compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 742))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 51))]));
      compute_local[(53)] = (compute_local[(53)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 742))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 835))]));
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 890))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 51))]));
      compute_local[(57)] = (compute_local[(57)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 890))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 835))]));
      compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1038))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 51))]));
      compute_local[(61)] = (compute_local[(61)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1038))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 835))]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 100))]));
      compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 884))]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 150))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 100))]));
      compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 150))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 884))]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 298))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 100))]));
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 298))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 884))]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 446))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 100))]));
      compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 446))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 884))]));
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 594))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 100))]));
      compute_local[(50)] = (compute_local[(50)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 594))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 884))]));
      compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 742))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 100))]));
      compute_local[(54)] = (compute_local[(54)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 742))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 884))]));
      compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 890))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 100))]));
      compute_local[(58)] = (compute_local[(58)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 890))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 884))]));
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1038))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 100))]));
      compute_local[(62)] = (compute_local[(62)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1038))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 884))]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 149))]));
      compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 2))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 933))]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 150))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 149))]));
      compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 150))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 933))]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 298))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 149))]));
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 298))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 933))]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 446))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 149))]));
      compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 446))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 933))]));
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 594))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 149))]));
      compute_local[(51)] = (compute_local[(51)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 594))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 933))]));
      compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 742))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 149))]));
      compute_local[(55)] = (compute_local[(55)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 742))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 933))]));
      compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 890))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 149))]));
      compute_local[(59)] = (compute_local[(59)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 890))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 933))]));
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1038))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 149))]));
      compute_local[(63)] = (compute_local[(63)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1038))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 933))]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 3))]));
      compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 787))]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 151))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 3))]));
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 151))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 787))]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 299))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 3))]));
      compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 299))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 787))]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 447))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 3))]));
      compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 447))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 787))]));
      compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 595))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 3))]));
      compute_local[(48)] = (compute_local[(48)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 595))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 787))]));
      compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 743))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 3))]));
      compute_local[(52)] = (compute_local[(52)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 743))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 787))]));
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 891))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 3))]));
      compute_local[(56)] = (compute_local[(56)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 891))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 787))]));
      compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1039))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 3))]));
      compute_local[(60)] = (compute_local[(60)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1039))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 787))]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 52))]));
      compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 836))]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 151))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 52))]));
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 151))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 836))]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 299))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 52))]));
      compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 299))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 836))]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 447))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 52))]));
      compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 447))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 836))]));
      compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 595))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 52))]));
      compute_local[(49)] = (compute_local[(49)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 595))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 836))]));
      compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 743))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 52))]));
      compute_local[(53)] = (compute_local[(53)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 743))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 836))]));
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 891))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 52))]));
      compute_local[(57)] = (compute_local[(57)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 891))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 836))]));
      compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1039))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 52))]));
      compute_local[(61)] = (compute_local[(61)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1039))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 836))]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 101))]));
      compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 885))]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 151))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 101))]));
      compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 151))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 885))]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 299))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 101))]));
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 299))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 885))]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 447))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 101))]));
      compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 447))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 885))]));
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 595))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 101))]));
      compute_local[(50)] = (compute_local[(50)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 595))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 885))]));
      compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 743))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 101))]));
      compute_local[(54)] = (compute_local[(54)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 743))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 885))]));
      compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 891))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 101))]));
      compute_local[(58)] = (compute_local[(58)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 891))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 885))]));
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1039))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 101))]));
      compute_local[(62)] = (compute_local[(62)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1039))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 885))]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 150))]));
      compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 3))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 934))]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 151))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 150))]));
      compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 151))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 934))]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 299))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 150))]));
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 299))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 934))]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 447))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 150))]));
      compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 447))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 934))]));
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 595))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 150))]));
      compute_local[(51)] = (compute_local[(51)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 595))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 934))]));
      compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 743))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 150))]));
      compute_local[(55)] = (compute_local[(55)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 743))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 934))]));
      compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 891))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 150))]));
      compute_local[(59)] = (compute_local[(59)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 891))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 934))]));
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1039))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 150))]));
      compute_local[(63)] = (compute_local[(63)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1039))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 934))]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 4))]));
      compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 788))]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 152))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 4))]));
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 152))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 788))]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 300))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 4))]));
      compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 300))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 788))]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 448))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 4))]));
      compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 448))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 788))]));
      compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 596))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 4))]));
      compute_local[(48)] = (compute_local[(48)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 596))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 788))]));
      compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 744))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 4))]));
      compute_local[(52)] = (compute_local[(52)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 744))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 788))]));
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 892))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 4))]));
      compute_local[(56)] = (compute_local[(56)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 892))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 788))]));
      compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1040))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 4))]));
      compute_local[(60)] = (compute_local[(60)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1040))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 788))]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 53))]));
      compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 837))]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 152))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 53))]));
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 152))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 837))]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 300))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 53))]));
      compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 300))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 837))]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 448))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 53))]));
      compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 448))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 837))]));
      compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 596))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 53))]));
      compute_local[(49)] = (compute_local[(49)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 596))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 837))]));
      compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 744))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 53))]));
      compute_local[(53)] = (compute_local[(53)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 744))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 837))]));
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 892))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 53))]));
      compute_local[(57)] = (compute_local[(57)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 892))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 837))]));
      compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1040))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 53))]));
      compute_local[(61)] = (compute_local[(61)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1040))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 837))]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 102))]));
      compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 886))]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 152))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 102))]));
      compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 152))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 886))]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 300))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 102))]));
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 300))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 886))]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 448))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 102))]));
      compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 448))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 886))]));
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 596))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 102))]));
      compute_local[(50)] = (compute_local[(50)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 596))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 886))]));
      compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 744))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 102))]));
      compute_local[(54)] = (compute_local[(54)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 744))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 886))]));
      compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 892))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 102))]));
      compute_local[(58)] = (compute_local[(58)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 892))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 886))]));
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1040))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 102))]));
      compute_local[(62)] = (compute_local[(62)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1040))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 886))]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 151))]));
      compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 4))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 935))]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 152))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 151))]));
      compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 152))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 935))]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 300))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 151))]));
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 300))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 935))]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 448))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 151))]));
      compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 448))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 935))]));
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 596))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 151))]));
      compute_local[(51)] = (compute_local[(51)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 596))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 935))]));
      compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 744))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 151))]));
      compute_local[(55)] = (compute_local[(55)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 744))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 935))]));
      compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 892))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 151))]));
      compute_local[(59)] = (compute_local[(59)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 892))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 935))]));
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1040))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 151))]));
      compute_local[(63)] = (compute_local[(63)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1040))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 935))]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 5))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 5))]));
      compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 5))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 789))]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 153))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 5))]));
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 153))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 789))]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 301))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 5))]));
      compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 301))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 789))]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 449))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 5))]));
      compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 449))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 789))]));
      compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 597))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 5))]));
      compute_local[(48)] = (compute_local[(48)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 597))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 789))]));
      compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 745))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 5))]));
      compute_local[(52)] = (compute_local[(52)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 745))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 789))]));
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 893))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 5))]));
      compute_local[(56)] = (compute_local[(56)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 893))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 789))]));
      compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1041))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 5))]));
      compute_local[(60)] = (compute_local[(60)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1041))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 789))]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 5))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 54))]));
      compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 5))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 838))]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 153))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 54))]));
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 153))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 838))]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 301))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 54))]));
      compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 301))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 838))]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 449))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 54))]));
      compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 449))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 838))]));
      compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 597))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 54))]));
      compute_local[(49)] = (compute_local[(49)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 597))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 838))]));
      compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 745))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 54))]));
      compute_local[(53)] = (compute_local[(53)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 745))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 838))]));
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 893))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 54))]));
      compute_local[(57)] = (compute_local[(57)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 893))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 838))]));
      compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1041))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 54))]));
      compute_local[(61)] = (compute_local[(61)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1041))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 838))]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 5))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 103))]));
      compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 5))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 887))]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 153))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 103))]));
      compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 153))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 887))]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 301))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 103))]));
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 301))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 887))]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 449))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 103))]));
      compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 449))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 887))]));
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 597))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 103))]));
      compute_local[(50)] = (compute_local[(50)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 597))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 887))]));
      compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 745))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 103))]));
      compute_local[(54)] = (compute_local[(54)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 745))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 887))]));
      compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 893))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 103))]));
      compute_local[(58)] = (compute_local[(58)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 893))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 887))]));
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1041))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 103))]));
      compute_local[(62)] = (compute_local[(62)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1041))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 887))]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 5))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 152))]));
      compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 5))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 936))]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 153))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 152))]));
      compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 153))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 936))]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 301))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 152))]));
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 301))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 936))]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 449))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 152))]));
      compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 449))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 936))]));
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 597))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 152))]));
      compute_local[(51)] = (compute_local[(51)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 597))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 936))]));
      compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 745))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 152))]));
      compute_local[(55)] = (compute_local[(55)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 745))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 936))]));
      compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 893))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 152))]));
      compute_local[(59)] = (compute_local[(59)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 893))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 936))]));
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1041))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 152))]));
      compute_local[(63)] = (compute_local[(63)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1041))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 936))]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 6))]));
      compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 790))]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 154))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 6))]));
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 154))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 790))]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 302))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 6))]));
      compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 302))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 790))]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 450))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 6))]));
      compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 450))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 790))]));
      compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 598))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 6))]));
      compute_local[(48)] = (compute_local[(48)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 598))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 790))]));
      compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 746))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 6))]));
      compute_local[(52)] = (compute_local[(52)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 746))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 790))]));
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 894))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 6))]));
      compute_local[(56)] = (compute_local[(56)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 894))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 790))]));
      compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1042))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 6))]));
      compute_local[(60)] = (compute_local[(60)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1042))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 790))]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 55))]));
      compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 839))]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 154))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 55))]));
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 154))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 839))]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 302))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 55))]));
      compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 302))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 839))]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 450))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 55))]));
      compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 450))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 839))]));
      compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 598))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 55))]));
      compute_local[(49)] = (compute_local[(49)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 598))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 839))]));
      compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 746))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 55))]));
      compute_local[(53)] = (compute_local[(53)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 746))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 839))]));
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 894))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 55))]));
      compute_local[(57)] = (compute_local[(57)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 894))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 839))]));
      compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1042))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 55))]));
      compute_local[(61)] = (compute_local[(61)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1042))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 839))]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 104))]));
      compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 888))]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 154))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 104))]));
      compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 154))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 888))]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 302))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 104))]));
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 302))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 888))]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 450))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 104))]));
      compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 450))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 888))]));
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 598))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 104))]));
      compute_local[(50)] = (compute_local[(50)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 598))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 888))]));
      compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 746))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 104))]));
      compute_local[(54)] = (compute_local[(54)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 746))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 888))]));
      compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 894))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 104))]));
      compute_local[(58)] = (compute_local[(58)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 894))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 888))]));
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1042))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 104))]));
      compute_local[(62)] = (compute_local[(62)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1042))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 888))]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 153))]));
      compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 6))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 937))]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 154))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 153))]));
      compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 154))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 937))]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 302))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 153))]));
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 302))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 937))]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 450))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 153))]));
      compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 450))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 937))]));
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 598))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 153))]));
      compute_local[(51)] = (compute_local[(51)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 598))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 937))]));
      compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 746))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 153))]));
      compute_local[(55)] = (compute_local[(55)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 746))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 937))]));
      compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 894))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 153))]));
      compute_local[(59)] = (compute_local[(59)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 894))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 937))]));
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1042))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 153))]));
      compute_local[(63)] = (compute_local[(63)] + (pad_temp_shared[(((((((int)threadIdx.y) * 74) + (ry_inner * 37)) + (((int)threadIdx.x) * 2)) + 1042))] * placeholder_shared[((((((int)threadIdx.z) * 196) + (ry_inner * 7)) + 937))]));
    }
  }
  compute[(((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 200704))] = compute_local[(32)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 224))] = compute_local[(4)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 200928))] = compute_local[(36)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 448))] = compute_local[(8)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 201152))] = compute_local[(40)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 672))] = compute_local[(12)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 201376))] = compute_local[(44)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 896))] = compute_local[(16)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 201600))] = compute_local[(48)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 1120))] = compute_local[(20)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 201824))] = compute_local[(52)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 1344))] = compute_local[(24)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 202048))] = compute_local[(56)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 1568))] = compute_local[(28)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 202272))] = compute_local[(60)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 12544))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 213248))] = compute_local[(33)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 12768))] = compute_local[(5)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 213472))] = compute_local[(37)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 12992))] = compute_local[(9)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 213696))] = compute_local[(41)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 13216))] = compute_local[(13)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 213920))] = compute_local[(45)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 13440))] = compute_local[(17)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 214144))] = compute_local[(49)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 13664))] = compute_local[(21)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 214368))] = compute_local[(53)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 13888))] = compute_local[(25)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 214592))] = compute_local[(57)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 14112))] = compute_local[(29)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 214816))] = compute_local[(61)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 25088))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 225792))] = compute_local[(34)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 25312))] = compute_local[(6)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 226016))] = compute_local[(38)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 25536))] = compute_local[(10)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 226240))] = compute_local[(42)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 25760))] = compute_local[(14)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 226464))] = compute_local[(46)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 25984))] = compute_local[(18)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 226688))] = compute_local[(50)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 26208))] = compute_local[(22)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 226912))] = compute_local[(54)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 26432))] = compute_local[(26)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 227136))] = compute_local[(58)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 26656))] = compute_local[(30)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 227360))] = compute_local[(62)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 37632))] = compute_local[(3)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 238336))] = compute_local[(35)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 37856))] = compute_local[(7)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 238560))] = compute_local[(39)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 38080))] = compute_local[(11)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 238784))] = compute_local[(43)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 38304))] = compute_local[(15)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 239008))] = compute_local[(47)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 38528))] = compute_local[(19)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 239232))] = compute_local[(51)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 38752))] = compute_local[(23)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 239456))] = compute_local[(55)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 38976))] = compute_local[(27)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 239680))] = compute_local[(59)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 39200))] = compute_local[(31)];
  compute[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 239904))] = compute_local[(63)];
}

