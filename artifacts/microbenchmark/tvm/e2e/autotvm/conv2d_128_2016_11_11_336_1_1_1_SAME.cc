//1_1_768_11_1_14
//128_2016_11_11_336_1_1_SAME
//dim3 grid(1, 1, 768);
//dim3 block(11, 1, 14);

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
extern "C" __global__ void __launch_bounds__(154) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[44];
  __shared__ float pad_temp_shared[847];
  __shared__ float placeholder_shared[392];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(33)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(34)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(35)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(36)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(37)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(38)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(39)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(40)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(41)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  compute_local[(42)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(32)] = 0.000000e+00f;
  compute_local[(43)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 288; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 847) {
      pad_temp_shared[(((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)))] = placeholder[((((((((int)blockIdx.z) / 6) * 243936) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)))];
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 846) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 1))] = placeholder[(((((((((int)blockIdx.z) / 6) * 243936) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 1))];
      }
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 845) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 2))] = placeholder[(((((((((int)blockIdx.z) / 6) * 243936) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 2))];
      }
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 844) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 3))] = placeholder[(((((((((int)blockIdx.z) / 6) * 243936) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 3))];
      }
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 843) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 4))] = placeholder[(((((((((int)blockIdx.z) / 6) * 243936) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 4))];
      }
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 842) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 5))] = placeholder[(((((((((int)blockIdx.z) / 6) * 243936) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 5))];
      }
    }
    if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 3) / 7)) < 56) {
      if (((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) < 392) {
        if (((int)threadIdx.x) < 10) {
          placeholder_shared[(((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)))] = placeholder1[(((((((((int)blockIdx.z) % 6) * 112896) + (((int)threadIdx.z) * 8064)) + (((((int)threadIdx.x) * 3) / 7) * 2016)) + (rc_outer * 7)) + ((((int)threadIdx.x) * 3) % 7)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + 1) / 7)) < 56) {
      if (((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) < 391) {
        if (((int)threadIdx.x) < 9) {
          placeholder_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[(((((((((int)blockIdx.z) % 6) * 112896) + (((int)threadIdx.z) * 8064)) + ((((((int)threadIdx.x) * 3) + 1) / 7) * 2016)) + (rc_outer * 7)) + (((((int)threadIdx.x) * 3) + 1) % 7)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + 2) / 7)) < 56) {
      if (((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) < 390) {
        if (((int)threadIdx.x) < 9) {
          placeholder_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[(((((((((int)blockIdx.z) % 6) * 112896) + (((int)threadIdx.z) * 8064)) + ((((((int)threadIdx.x) * 3) + 2) / 7) * 2016)) + (rc_outer * 7)) + (((((int)threadIdx.x) * 3) + 2) % 7)))];
        }
      }
    }
    __syncthreads();
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 11))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 22))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 33))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 55))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 66))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 88))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 99))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 110))] * placeholder_shared[((((int)threadIdx.z) * 7))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 98))]));
    compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 196))]));
    compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 294))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 132))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 132))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 132))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 132))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 187))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 187))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 187))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 187))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 198))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 198))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 198))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 198))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 209))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 209))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 209))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 209))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 1))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 99))]));
    compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 197))]));
    compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 295))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 341))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 341))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 341))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 341))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 352))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 2))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 352))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 100))]));
    compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 352))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 198))]));
    compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 352))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 296))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 363))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 363))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 363))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 363))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 374))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 374))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 374))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 374))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 385))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 385))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 385))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 385))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 396))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 396))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 396))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 396))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 418))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 418))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 418))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 418))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 451))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 451))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 451))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 451))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 473))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 3))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 473))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 101))]));
    compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 473))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 199))]));
    compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 473))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 297))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 506))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 506))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 506))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 506))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 517))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 517))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 517))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 517))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 528))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 528))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 528))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 528))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 539))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 539))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 539))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 539))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 550))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 550))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 550))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 550))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 561))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 561))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 561))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 561))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 572))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 572))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 572))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 572))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 583))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 583))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 583))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 583))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 594))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 4))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 594))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 102))]));
    compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 594))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 200))]));
    compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 594))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 298))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 627))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 627))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 627))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 627))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 638))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 638))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 638))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 638))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 649))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 649))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 649))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 649))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 660))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 660))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 660))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 660))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 671))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 671))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 671))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 671))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 682))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 682))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 682))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 682))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 693))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 693))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 693))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 693))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 704))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 704))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 704))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 704))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 715))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 5))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 715))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 103))]));
    compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 715))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 201))]));
    compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 715))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 299))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 726))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 726))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 726))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 726))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 737))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 737))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 737))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 737))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 748))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 748))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 748))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 748))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 759))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 759))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 759))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 759))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 781))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 781))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 781))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 781))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 792))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 792))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 792))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 792))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 803))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 803))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 803))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 803))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 814))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 814))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 814))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 814))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 825))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 825))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 825))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 825))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 836))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 6))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 836))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 104))]));
    compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 836))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 202))]));
    compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 836))] * placeholder_shared[(((((int)threadIdx.z) * 7) + 300))]));
  }
  compute[((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1694))] = compute_local[(11)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3388))] = compute_local[(22)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5082))] = compute_local[(33)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 11))] = compute_local[(1)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1705))] = compute_local[(12)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3399))] = compute_local[(23)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5093))] = compute_local[(34)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 22))] = compute_local[(2)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1716))] = compute_local[(13)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3410))] = compute_local[(24)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5104))] = compute_local[(35)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 33))] = compute_local[(3)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1727))] = compute_local[(14)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3421))] = compute_local[(25)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5115))] = compute_local[(36)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 44))] = compute_local[(4)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1738))] = compute_local[(15)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3432))] = compute_local[(26)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5126))] = compute_local[(37)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 55))] = compute_local[(5)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1749))] = compute_local[(16)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3443))] = compute_local[(27)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5137))] = compute_local[(38)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 66))] = compute_local[(6)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1760))] = compute_local[(17)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3454))] = compute_local[(28)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5148))] = compute_local[(39)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 77))] = compute_local[(7)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1771))] = compute_local[(18)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3465))] = compute_local[(29)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5159))] = compute_local[(40)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 88))] = compute_local[(8)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1782))] = compute_local[(19)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3476))] = compute_local[(30)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5170))] = compute_local[(41)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 99))] = compute_local[(9)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1793))] = compute_local[(20)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3487))] = compute_local[(31)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5181))] = compute_local[(42)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 110))] = compute_local[(10)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 1804))] = compute_local[(21)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 3498))] = compute_local[(32)];
  compute[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 5192))] = compute_local[(43)];
}

