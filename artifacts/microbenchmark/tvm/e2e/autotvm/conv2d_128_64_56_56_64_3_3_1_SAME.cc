//3_10_256_27_1_4
//128_64_56_56_64_3_1_SAME
//dim3 grid(3, 10, 256);
//dim3 block(27, 1, 4);

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
extern "C" __global__ void __launch_bounds__(108) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[48];
  __shared__ float pad_temp_shared[216];
  __shared__ float placeholder_shared[96];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  compute_local[(32)] = 0.000000e+00f;
  compute_local[(33)] = 0.000000e+00f;
  compute_local[(34)] = 0.000000e+00f;
  compute_local[(35)] = 0.000000e+00f;
  compute_local[(36)] = 0.000000e+00f;
  compute_local[(37)] = 0.000000e+00f;
  compute_local[(38)] = 0.000000e+00f;
  compute_local[(39)] = 0.000000e+00f;
  compute_local[(40)] = 0.000000e+00f;
  compute_local[(41)] = 0.000000e+00f;
  compute_local[(42)] = 0.000000e+00f;
  compute_local[(43)] = 0.000000e+00f;
  compute_local[(44)] = 0.000000e+00f;
  compute_local[(45)] = 0.000000e+00f;
  compute_local[(46)] = 0.000000e+00f;
  compute_local[(47)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    if ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + ((((int)threadIdx.x) * 2) / 27)) < 58) {
      if (((((int)blockIdx.x) * 27) + ((((int)threadIdx.x) * 2) % 27)) < 58) {
        pad_temp_shared[(((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)))] = (((((1 <= (((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + ((((int)threadIdx.x) * 2) / 27))) && ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + ((((int)threadIdx.x) * 2) / 27)) < 57)) && (1 <= ((((int)blockIdx.x) * 27) + ((((int)threadIdx.x) * 2) % 27)))) && (((((int)blockIdx.x) * 27) + ((((int)threadIdx.x) * 2) % 27)) < 57)) ? placeholder[((((((((((((int)blockIdx.z) >> 1) * 200704) + (rc_outer * 3136)) + (((int)blockIdx.y) * 336)) + (((int)threadIdx.z) * 112)) + (((((int)threadIdx.x) * 2) / 27) * 56)) + (((int)blockIdx.x) * 27)) + ((((int)threadIdx.x) * 2) % 27)) - 57))] : 0.000000e+00f);
      }
    }
    if ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + (((((int)threadIdx.x) * 2) + 1) / 27)) < 58) {
      if (((((int)blockIdx.x) * 27) + (((((int)threadIdx.x) * 2) + 1) % 27)) < 58) {
        pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)) + 1))] = (((((1 <= (((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + (((((int)threadIdx.x) * 2) + 1) / 27))) && ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + (((((int)threadIdx.x) * 2) + 1) / 27)) < 57)) && (1 <= ((((int)blockIdx.x) * 27) + (((((int)threadIdx.x) * 2) + 1) % 27)))) && (((((int)blockIdx.x) * 27) + (((((int)threadIdx.x) * 2) + 1) % 27)) < 57)) ? placeholder[((((((((((((int)blockIdx.z) >> 1) * 200704) + (rc_outer * 3136)) + (((int)blockIdx.y) * 336)) + (((int)threadIdx.z) * 112)) + ((((((int)threadIdx.x) * 2) + 1) / 27) * 56)) + (((int)blockIdx.x) * 27)) + (((((int)threadIdx.x) * 2) + 1) % 27)) - 57))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 8) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 24) + ((int)threadIdx.x)) < 96) {
        if (((int)threadIdx.x) < 24) {
          placeholder_shared[(((((int)threadIdx.z) * 24) + ((int)threadIdx.x)))] = placeholder1[(((((((((int)blockIdx.z) & 1) * 18432) + (((int)threadIdx.z) * 4608)) + ((((int)threadIdx.x) / 3) * 576)) + (rc_outer * 9)) + ((((int)threadIdx.x) % 3) * 3)))];
        }
      }
    }
    __syncthreads();
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    __syncthreads();
    if ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + ((((int)threadIdx.x) * 2) / 27)) < 58) {
      if (((((int)blockIdx.x) * 27) + ((((int)threadIdx.x) * 2) % 27)) < 57) {
        pad_temp_shared[(((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)))] = ((((1 <= (((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + ((((int)threadIdx.x) * 2) / 27))) && ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + ((((int)threadIdx.x) * 2) / 27)) < 57)) && (((((int)blockIdx.x) * 27) + ((((int)threadIdx.x) * 2) % 27)) < 56)) ? placeholder[((((((((((((int)blockIdx.z) >> 1) * 200704) + (rc_outer * 3136)) + (((int)blockIdx.y) * 336)) + (((int)threadIdx.z) * 112)) + (((((int)threadIdx.x) * 2) / 27) * 56)) + (((int)blockIdx.x) * 27)) + ((((int)threadIdx.x) * 2) % 27)) - 56))] : 0.000000e+00f);
      }
    }
    if ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + (((((int)threadIdx.x) * 2) + 1) / 27)) < 58) {
      if (((((int)blockIdx.x) * 27) + (((((int)threadIdx.x) * 2) + 1) % 27)) < 57) {
        pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)) + 1))] = ((((1 <= (((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + (((((int)threadIdx.x) * 2) + 1) / 27))) && ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + (((((int)threadIdx.x) * 2) + 1) / 27)) < 57)) && (((((int)blockIdx.x) * 27) + (((((int)threadIdx.x) * 2) + 1) % 27)) < 56)) ? placeholder[((((((((((((int)blockIdx.z) >> 1) * 200704) + (rc_outer * 3136)) + (((int)blockIdx.y) * 336)) + (((int)threadIdx.z) * 112)) + ((((((int)threadIdx.x) * 2) + 1) / 27) * 56)) + (((int)blockIdx.x) * 27)) + (((((int)threadIdx.x) * 2) + 1) % 27)) - 56))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 8) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 24) + ((int)threadIdx.x)) < 96) {
        if (((int)threadIdx.x) < 24) {
          placeholder_shared[(((((int)threadIdx.z) * 24) + ((int)threadIdx.x)))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 18432) + (((int)threadIdx.z) * 4608)) + ((((int)threadIdx.x) / 3) * 576)) + (rc_outer * 9)) + ((((int)threadIdx.x) % 3) * 3)) + 1))];
        }
      }
    }
    __syncthreads();
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    __syncthreads();
    if ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + ((((int)threadIdx.x) * 2) / 27)) < 58) {
      if (((((int)blockIdx.x) * 27) + ((((int)threadIdx.x) * 2) % 27)) < 56) {
        pad_temp_shared[(((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)))] = ((((1 <= (((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + ((((int)threadIdx.x) * 2) / 27))) && ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + ((((int)threadIdx.x) * 2) / 27)) < 57)) && (((((int)blockIdx.x) * 27) + ((((int)threadIdx.x) * 2) % 27)) < 55)) ? placeholder[((((((((((((int)blockIdx.z) >> 1) * 200704) + (rc_outer * 3136)) + (((int)blockIdx.y) * 336)) + (((int)threadIdx.z) * 112)) + (((((int)threadIdx.x) * 2) / 27) * 56)) + (((int)blockIdx.x) * 27)) + ((((int)threadIdx.x) * 2) % 27)) - 55))] : 0.000000e+00f);
      }
    }
    if ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + (((((int)threadIdx.x) * 2) + 1) / 27)) < 58) {
      if (((((int)blockIdx.x) * 27) + (((((int)threadIdx.x) * 2) + 1) % 27)) < 56) {
        pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 2)) + 1))] = ((((1 <= (((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + (((((int)threadIdx.x) * 2) + 1) / 27))) && ((((((int)blockIdx.y) * 6) + (((int)threadIdx.z) * 2)) + (((((int)threadIdx.x) * 2) + 1) / 27)) < 57)) && (((((int)blockIdx.x) * 27) + (((((int)threadIdx.x) * 2) + 1) % 27)) < 55)) ? placeholder[((((((((((((int)blockIdx.z) >> 1) * 200704) + (rc_outer * 3136)) + (((int)blockIdx.y) * 336)) + (((int)threadIdx.z) * 112)) + ((((((int)threadIdx.x) * 2) + 1) / 27) * 56)) + (((int)blockIdx.x) * 27)) + (((((int)threadIdx.x) * 2) + 1) % 27)) - 55))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 8) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 24) + ((int)threadIdx.x)) < 96) {
        if (((int)threadIdx.x) < 24) {
          placeholder_shared[(((((int)threadIdx.z) * 24) + ((int)threadIdx.x)))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 18432) + (((int)threadIdx.z) * 4608)) + ((((int)threadIdx.x) / 3) * 576)) + (rc_outer * 9)) + ((((int)threadIdx.x) % 3) * 3)) + 2))];
        }
      }
    }
    __syncthreads();
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[((((int)threadIdx.z) * 24))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 3))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 6))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 9))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 12))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 15))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 18))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 21))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 1))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 4))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 7))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 10))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 13))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 16))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 19))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 27))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 22))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 2))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 5))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 8))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 11))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 14))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 17))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 20))]));
      }
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
    }
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 81))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 108))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 162))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
    if (((int)blockIdx.y) < 9) {
      if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
        compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 24) + 23))]));
      }
    }
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)))] = compute_local[(0)];
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 56))] = compute_local[(1)];
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 112))] = compute_local[(2)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 168))] = compute_local[(3)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 224))] = compute_local[(4)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 280))] = compute_local[(5)];
    }
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 3136))] = compute_local[(6)];
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 3192))] = compute_local[(7)];
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 3248))] = compute_local[(8)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 3304))] = compute_local[(9)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 3360))] = compute_local[(10)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 3416))] = compute_local[(11)];
    }
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 6272))] = compute_local[(12)];
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 6328))] = compute_local[(13)];
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 6384))] = compute_local[(14)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 6440))] = compute_local[(15)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 6496))] = compute_local[(16)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 6552))] = compute_local[(17)];
    }
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 9408))] = compute_local[(18)];
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 9464))] = compute_local[(19)];
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 9520))] = compute_local[(20)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 9576))] = compute_local[(21)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 9632))] = compute_local[(22)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 9688))] = compute_local[(23)];
    }
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 12544))] = compute_local[(24)];
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 12600))] = compute_local[(25)];
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 12656))] = compute_local[(26)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 12712))] = compute_local[(27)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 12768))] = compute_local[(28)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 12824))] = compute_local[(29)];
    }
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 15680))] = compute_local[(30)];
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 15736))] = compute_local[(31)];
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 15792))] = compute_local[(32)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 15848))] = compute_local[(33)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 15904))] = compute_local[(34)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 15960))] = compute_local[(35)];
    }
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 18816))] = compute_local[(36)];
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 18872))] = compute_local[(37)];
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 18928))] = compute_local[(38)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 18984))] = compute_local[(39)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 19040))] = compute_local[(40)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 19096))] = compute_local[(41)];
    }
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 21952))] = compute_local[(42)];
  }
  if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 22008))] = compute_local[(43)];
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 22064))] = compute_local[(44)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 22120))] = compute_local[(45)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 22176))] = compute_local[(46)];
    }
  }
  if (((int)blockIdx.y) < 9) {
    if (((((int)blockIdx.x) * 27) + ((int)threadIdx.x)) < 56) {
      compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 25088)) + (((int)blockIdx.y) * 336)) + (((int)blockIdx.x) * 27)) + ((int)threadIdx.x)) + 22232))] = compute_local[(47)];
    }
  }
}

