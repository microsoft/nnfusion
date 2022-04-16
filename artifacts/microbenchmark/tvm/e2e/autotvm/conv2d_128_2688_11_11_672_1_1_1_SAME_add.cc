//1_1_1536_11_1_14
//128_2688_11_11_672_1_1_SAME
//dim3 grid(1, 1, 1536);
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
extern "C" __global__ void __launch_bounds__(154) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[44];
  __shared__ float pad_temp_shared[847];
  __shared__ float placeholder_shared[392];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  compute[(28)] = 0.000000e+00f;
  compute[(29)] = 0.000000e+00f;
  compute[(30)] = 0.000000e+00f;
  compute[(31)] = 0.000000e+00f;
  compute[(32)] = 0.000000e+00f;
  compute[(33)] = 0.000000e+00f;
  compute[(34)] = 0.000000e+00f;
  compute[(35)] = 0.000000e+00f;
  compute[(36)] = 0.000000e+00f;
  compute[(37)] = 0.000000e+00f;
  compute[(38)] = 0.000000e+00f;
  compute[(39)] = 0.000000e+00f;
  compute[(40)] = 0.000000e+00f;
  compute[(41)] = 0.000000e+00f;
  compute[(42)] = 0.000000e+00f;
  compute[(43)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 384; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 847) {
      pad_temp_shared[(((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)))] = placeholder[((((((((int)blockIdx.z) / 12) * 325248) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)))];
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 846) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 1))] = placeholder[(((((((((int)blockIdx.z) / 12) * 325248) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 1))];
      }
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 845) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 2))] = placeholder[(((((((((int)blockIdx.z) / 12) * 325248) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 2))];
      }
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 844) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 3))] = placeholder[(((((((((int)blockIdx.z) / 12) * 325248) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 3))];
      }
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 843) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 4))] = placeholder[(((((((((int)blockIdx.z) / 12) * 325248) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 4))];
      }
    }
    if (((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) < 842) {
      if (((int)threadIdx.x) < 10) {
        pad_temp_shared[((((((int)threadIdx.z) * 61) + (((int)threadIdx.x) * 6)) + 5))] = placeholder[(((((((((int)blockIdx.z) / 12) * 325248) + (rc_outer * 847)) + (((int)threadIdx.z) * 61)) + (((int)threadIdx.x) * 6)) + 5))];
      }
    }
    if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 3) / 7)) < 56) {
      if (((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) < 392) {
        if (((int)threadIdx.x) < 10) {
          placeholder_shared[(((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)))] = placeholder1[(((((((((int)blockIdx.z) % 12) * 150528) + (((int)threadIdx.z) * 10752)) + (((((int)threadIdx.x) * 3) / 7) * 2688)) + (rc_outer * 7)) + ((((int)threadIdx.x) * 3) % 7)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + 1) / 7)) < 56) {
      if (((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) < 391) {
        if (((int)threadIdx.x) < 9) {
          placeholder_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[(((((((((int)blockIdx.z) % 12) * 150528) + (((int)threadIdx.z) * 10752)) + ((((((int)threadIdx.x) * 3) + 1) / 7) * 2688)) + (rc_outer * 7)) + (((((int)threadIdx.x) * 3) + 1) % 7)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + 2) / 7)) < 56) {
      if (((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) < 390) {
        if (((int)threadIdx.x) < 9) {
          placeholder_shared[((((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[(((((((((int)blockIdx.z) % 12) * 150528) + (((int)threadIdx.z) * 10752)) + ((((((int)threadIdx.x) * 3) + 2) / 7) * 2688)) + (rc_outer * 7)) + (((((int)threadIdx.x) * 3) + 2) % 7)))];
        }
      }
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 11))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 22))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 33))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 55))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 66))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 88))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 99))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 110))] * placeholder_shared[((((int)threadIdx.z) * 28))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 7))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 14))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 11))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 22))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 33))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 66))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 88))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 99))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 21))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 132))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 187))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 198))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 209))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 1))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 132))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 187))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 198))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 209))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 8))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 132))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 187))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 198))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 209))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 15))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 132))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 143))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 176))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 187))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 198))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 209))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 231))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 22))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 341))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 352))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 2))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 341))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 352))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 9))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 341))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 352))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 16))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 242))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 253))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 264))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 286))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 297))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 319))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 341))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 352))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 23))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 363))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 374))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 385))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 396))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 418))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 451))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 473))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 3))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 363))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 374))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 385))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 396))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 418))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 451))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 473))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 10))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 363))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 374))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 385))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 396))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 418))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 451))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 473))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 17))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 363))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 374))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 385))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 396))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 407))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 418))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 429))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 451))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 473))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 24))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 506))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 517))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 528))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 539))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 550))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 561))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 572))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 583))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 594))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 4))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 506))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 517))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 528))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 539))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 550))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 561))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 572))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 583))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 594))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 11))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 506))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 517))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 528))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 539))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 550))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 561))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 572))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 583))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 594))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 18))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 484))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 506))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 517))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 528))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 539))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 550))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 561))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 572))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 583))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 594))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 25))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 627))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 638))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 649))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 660))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 671))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 682))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 693))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 704))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 715))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 5))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 627))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 638))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 649))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 660))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 671))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 682))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 693))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 704))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 715))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 12))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 627))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 638))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 649))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 660))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 671))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 682))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 693))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 704))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 715))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 19))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 627))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 638))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 649))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 660))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 671))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 682))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 693))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 704))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 715))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 26))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 726))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 737))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 748))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 759))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 781))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 792))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 803))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 814))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 825))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 836))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 6))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 726))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 737))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 748))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 759))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 781))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 792))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 803))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 814))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 825))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 836))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 13))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 726))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 737))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 748))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 759))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 781))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 792))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 803))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 814))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 825))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 836))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 20))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 726))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 737))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 748))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 759))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 781))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 792))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 803))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 814))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 825))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 836))] * placeholder_shared[(((((int)threadIdx.z) * 28) + 27))]));
  }
  T_add[((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)))] = (compute[(0)] + input2[((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 11))] = (compute[(1)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 11))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 22))] = (compute[(2)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 22))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 33))] = (compute[(3)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 33))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 44))] = (compute[(4)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 44))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 55))] = (compute[(5)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 55))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 66))] = (compute[(6)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 66))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 77))] = (compute[(7)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 77))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 88))] = (compute[(8)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 88))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 99))] = (compute[(9)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 99))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 110))] = (compute[(10)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 110))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 121))] = (compute[(11)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 121))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 132))] = (compute[(12)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 132))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 143))] = (compute[(13)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 143))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 154))] = (compute[(14)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 154))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 165))] = (compute[(15)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 165))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 176))] = (compute[(16)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 176))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 187))] = (compute[(17)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 187))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 198))] = (compute[(18)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 198))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 209))] = (compute[(19)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 209))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 220))] = (compute[(20)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 220))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 231))] = (compute[(21)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 231))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 242))] = (compute[(22)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 242))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 253))] = (compute[(23)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 253))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 264))] = (compute[(24)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 264))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 275))] = (compute[(25)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 275))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 286))] = (compute[(26)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 286))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 297))] = (compute[(27)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 297))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 308))] = (compute[(28)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 308))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 319))] = (compute[(29)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 319))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 330))] = (compute[(30)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 330))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 341))] = (compute[(31)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 341))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 352))] = (compute[(32)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 352))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 363))] = (compute[(33)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 363))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 374))] = (compute[(34)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 374))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 385))] = (compute[(35)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 385))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 396))] = (compute[(36)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 396))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 407))] = (compute[(37)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 407))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 418))] = (compute[(38)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 418))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 429))] = (compute[(39)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 429))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 440))] = (compute[(40)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 440))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 451))] = (compute[(41)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 451))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 462))] = (compute[(42)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 462))]);
  T_add[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 473))] = (compute[(43)] + input2[(((((((int)blockIdx.z) * 6776) + (((int)threadIdx.z) * 484)) + ((int)threadIdx.x)) + 473))]);
}

