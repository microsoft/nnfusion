//1232_1_1_176_1_1
//128_4032_11_11_672_1_1_SAME
//dim3 grid(1232, 1, 1);
//dim3 block(176, 1, 1);

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
extern "C" __global__ void __launch_bounds__(176) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[48];
  __shared__ float pad_temp_shared[176];
  __shared__ float input1_shared[192];
  compute[(0)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(28)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(29)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(30)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(31)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(32)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(33)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(34)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(35)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(36)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(37)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(38)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(39)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(40)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(41)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(42)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(43)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(44)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(45)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(46)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(47)] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 2016; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[(((int)threadIdx.x))] = input0[(((((((((int)blockIdx.x) / 77) * 3902976) + ((((int)threadIdx.x) / 22) * 487872)) + (rc_outer_outer * 242)) + ((((int)threadIdx.x) % 22) * 11)) + (((int)blockIdx.x) % 11)))];
    if (((int)threadIdx.x) < 96) {
      ((float2*)(input1_shared + ((((int)threadIdx.x) * 2))))[0] = ((float2*)(input1 + ((((((((int)blockIdx.x) % 77) / 11) * 387072) + (((int)threadIdx.x) * 4032)) + (rc_outer_outer * 2)))))[0];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[((((((int)threadIdx.x) % 88) / 11) * 24))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[((((((int)threadIdx.x) % 88) / 11) * 24))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 2))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 2))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 4))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 4))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 6))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 6))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 8))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 8))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 10))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 10))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 12))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 12))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 14))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 14))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 16))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 16))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 18))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 18))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 20))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 20))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 22))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 88))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 22))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 1))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 1))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 3))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 3))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 5))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 5))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 7))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 7))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 9))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 9))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 11))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 11))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 13))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 13))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 15))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 15))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 17))]));
    compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 17))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 19))]));
    compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 19))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 21))]));
    compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 21))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 11))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 23))]));
    compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 99))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 23))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[((((((int)threadIdx.x) % 88) / 11) * 24))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[((((((int)threadIdx.x) % 88) / 11) * 24))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 2))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 2))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 4))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 4))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 6))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 6))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 8))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 8))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 10))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 10))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 12))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 12))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 14))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 14))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 16))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 16))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 18))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 18))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 20))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 20))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 22))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 22))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 110))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 22))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 1))]));
    compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 1))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 3))]));
    compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 3))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 5))]));
    compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 5))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 7))]));
    compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 7))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 9))]));
    compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 9))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 11))]));
    compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 11))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 13))]));
    compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 13))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 15))]));
    compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 15))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 17))]));
    compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 17))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 19))]));
    compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 19))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 21))]));
    compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 21))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 33))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 23))]));
    compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((((int)threadIdx.x) / 88) * 44) + (((int)threadIdx.x) % 11)) + 121))] * input1_shared[(((((((int)threadIdx.x) % 88) / 11) * 24) + 23))]));
  }
  for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
    for (int ax1_inner = 0; ax1_inner < 12; ++ax1_inner) {
      T_add[((((((((((((int)blockIdx.x) / 77) * 650496) + ((((int)threadIdx.x) / 88) * 162624)) + (ax0_inner * 81312)) + (((((int)blockIdx.x) % 77) / 11) * 11616)) + (((((int)threadIdx.x) % 88) / 11) * 1452)) + (ax1_inner * 121)) + ((((int)threadIdx.x) % 11) * 11)) + (((int)blockIdx.x) % 11)))] = (compute[(((ax0_inner * 12) + ax1_inner))] + input2[((((((((((((int)blockIdx.x) / 77) * 650496) + ((((int)threadIdx.x) / 88) * 162624)) + (ax0_inner * 81312)) + (((((int)blockIdx.x) % 77) / 11) * 11616)) + (((((int)threadIdx.x) % 88) / 11) * 1452)) + (ax1_inner * 121)) + ((((int)threadIdx.x) % 11) * 11)) + (((int)blockIdx.x) % 11)))]);
      T_add[(((((((((((((int)blockIdx.x) / 77) * 650496) + ((((int)threadIdx.x) / 88) * 162624)) + (ax0_inner * 81312)) + (((((int)blockIdx.x) % 77) / 11) * 11616)) + (((((int)threadIdx.x) % 88) / 11) * 1452)) + (ax1_inner * 121)) + ((((int)threadIdx.x) % 11) * 11)) + (((int)blockIdx.x) % 11)) + 325248))] = (compute[((((ax0_inner * 12) + ax1_inner) + 24))] + input2[(((((((((((((int)blockIdx.x) / 77) * 650496) + ((((int)threadIdx.x) / 88) * 162624)) + (ax0_inner * 81312)) + (((((int)blockIdx.x) % 77) / 11) * 11616)) + (((((int)threadIdx.x) % 88) / 11) * 1452)) + (ax1_inner * 121)) + ((((int)threadIdx.x) % 11) * 11)) + (((int)blockIdx.x) % 11)) + 325248))]);
    }
  }
}

