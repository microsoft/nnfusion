//2656_1_1_249_1_1
//128_168_83_83_84_1_1_SAME
//dim3 grid(2656, 1, 1);
//dim3 block(249, 1, 1);

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
extern "C" __global__ void __launch_bounds__(249) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[112];
  __shared__ float pad_temp_shared[7968];
  __shared__ float input1_shared[2016];
  for (int nn_inner_init = 0; nn_inner_init < 2; ++nn_inner_init) {
    compute[((nn_inner_init * 4))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 8))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 16))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 24))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 32))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 40))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 48))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 56))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 64))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 72))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 80))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 88))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 96))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 104))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 1))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 9))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 17))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 25))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 33))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 41))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 49))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 57))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 65))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 73))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 81))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 89))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 97))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 105))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 2))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 10))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 18))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 26))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 34))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 42))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 50))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 58))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 66))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 74))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 82))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 90))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 98))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 106))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 3))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 11))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 19))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 27))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 35))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 43))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 51))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 59))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 67))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 75))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 83))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 91))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 99))] = 0.000000e+00f;
    compute[(((nn_inner_init * 4) + 107))] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 7; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[(((int)threadIdx.x))] = input0[(((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 249))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 20667))];
    pad_temp_shared[((((int)threadIdx.x) + 498))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 41334))];
    pad_temp_shared[((((int)threadIdx.x) + 747))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 62001))];
    pad_temp_shared[((((int)threadIdx.x) + 996))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 82668))];
    pad_temp_shared[((((int)threadIdx.x) + 1245))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 103335))];
    pad_temp_shared[((((int)threadIdx.x) + 1494))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 124002))];
    pad_temp_shared[((((int)threadIdx.x) + 1743))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 144669))];
    pad_temp_shared[((((int)threadIdx.x) + 1992))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1157352))];
    pad_temp_shared[((((int)threadIdx.x) + 2241))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 2241) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 3) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 2490))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 2490) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 6) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 2739))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 2739) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 9) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 2988))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 2988) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 12) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 3237))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 3237) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 15) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 3486))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 3486) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 18) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 3735))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 3735) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 21) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 3984))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 2314704))];
    pad_temp_shared[((((int)threadIdx.x) + 4233))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 4233) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 3) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 4482))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 4482) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 6) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 4731))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 4731) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 9) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 4980))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 4980) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 12) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 5229))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 5229) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 15) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 5478))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 5478) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 18) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 5727))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 5727) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 21) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 5976))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (rc_outer_outer * 165336)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 3472056))];
    pad_temp_shared[((((int)threadIdx.x) + 6225))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 6225) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 3) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 6474))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 6474) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 6) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 6723))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 6723) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 9) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 6972))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 6972) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 12) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 7221))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 7221) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 15) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 7470))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 7470) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 18) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 7719))] = input0[((((((((((int)blockIdx.x) / 83) * 4629408) + (((((int)threadIdx.x) + 7719) / 1992) * 1157352)) + (rc_outer_outer * 165336)) + (((((int)threadIdx.x) / 83) + 21) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    input1_shared[((((int)threadIdx.x) * 2))] = input1[(((((((int)threadIdx.x) / 12) * 168) + (rc_outer_outer * 24)) + ((((int)threadIdx.x) % 12) * 2)))];
    input1_shared[(((((int)threadIdx.x) * 2) + 1))] = input1[(((((((((int)threadIdx.x) * 2) + 1) / 24) * 168) + (rc_outer_outer * 24)) + (((((int)threadIdx.x) * 2) + 1) % 24)))];
    input1_shared[(((((int)threadIdx.x) * 2) + 498))] = input1[(((((((((int)threadIdx.x) * 2) + 498) / 24) * 168) + (rc_outer_outer * 24)) + (((((int)threadIdx.x) * 2) + 18) % 24)))];
    input1_shared[(((((int)threadIdx.x) * 2) + 499))] = input1[(((((((((int)threadIdx.x) * 2) + 499) / 24) * 168) + (rc_outer_outer * 24)) + (((((int)threadIdx.x) * 2) + 19) % 24)))];
    input1_shared[(((((int)threadIdx.x) * 2) + 996))] = input1[(((((((((int)threadIdx.x) * 2) + 996) / 24) * 168) + (rc_outer_outer * 24)) + (((((int)threadIdx.x) * 2) + 12) % 24)))];
    input1_shared[(((((int)threadIdx.x) * 2) + 997))] = input1[(((((((((int)threadIdx.x) * 2) + 997) / 24) * 168) + (rc_outer_outer * 24)) + (((((int)threadIdx.x) * 2) + 13) % 24)))];
    input1_shared[(((((int)threadIdx.x) * 2) + 1494))] = input1[(((((((((int)threadIdx.x) * 2) + 1494) / 24) * 168) + (rc_outer_outer * 24)) + (((((int)threadIdx.x) * 2) + 6) % 24)))];
    input1_shared[(((((int)threadIdx.x) * 2) + 1495))] = input1[(((((((((int)threadIdx.x) * 2) + 1495) / 24) * 168) + (rc_outer_outer * 24)) + (((((int)threadIdx.x) * 2) + 7) % 24)))];
    if (((int)threadIdx.x) < 12) {
      input1_shared[(((((int)threadIdx.x) * 2) + 1992))] = input1[((((rc_outer_outer * 24) + (((int)threadIdx.x) * 2)) + 13944))];
    }
    if (((int)threadIdx.x) < 12) {
      input1_shared[(((((int)threadIdx.x) * 2) + 1993))] = input1[(((((((((int)threadIdx.x) * 2) + 1993) / 24) * 168) + (rc_outer_outer * 24)) + ((((int)threadIdx.x) * 2) + 1)))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 6; ++rc_inner) {
        for (int nn_inner = 0; nn_inner < 2; ++nn_inner) {
          compute[((nn_inner * 4))] = (compute[((nn_inner * 4))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[(((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner))]));
          compute[(((nn_inner * 4) + 8))] = (compute[(((nn_inner * 4) + 8))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 288))]));
          compute[(((nn_inner * 4) + 16))] = (compute[(((nn_inner * 4) + 16))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 576))]));
          compute[(((nn_inner * 4) + 24))] = (compute[(((nn_inner * 4) + 24))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 864))]));
          compute[(((nn_inner * 4) + 32))] = (compute[(((nn_inner * 4) + 32))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1152))]));
          compute[(((nn_inner * 4) + 40))] = (compute[(((nn_inner * 4) + 40))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1440))]));
          compute[(((nn_inner * 4) + 48))] = (compute[(((nn_inner * 4) + 48))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1728))]));
          compute[(((nn_inner * 4) + 56))] = (compute[(((nn_inner * 4) + 56))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[(((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner))]));
          compute[(((nn_inner * 4) + 64))] = (compute[(((nn_inner * 4) + 64))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 288))]));
          compute[(((nn_inner * 4) + 72))] = (compute[(((nn_inner * 4) + 72))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 576))]));
          compute[(((nn_inner * 4) + 80))] = (compute[(((nn_inner * 4) + 80))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 864))]));
          compute[(((nn_inner * 4) + 88))] = (compute[(((nn_inner * 4) + 88))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1152))]));
          compute[(((nn_inner * 4) + 96))] = (compute[(((nn_inner * 4) + 96))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1440))]));
          compute[(((nn_inner * 4) + 104))] = (compute[(((nn_inner * 4) + 104))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1728))]));
          compute[(((nn_inner * 4) + 1))] = (compute[(((nn_inner * 4) + 1))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 24))]));
          compute[(((nn_inner * 4) + 9))] = (compute[(((nn_inner * 4) + 9))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 312))]));
          compute[(((nn_inner * 4) + 17))] = (compute[(((nn_inner * 4) + 17))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 600))]));
          compute[(((nn_inner * 4) + 25))] = (compute[(((nn_inner * 4) + 25))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 888))]));
          compute[(((nn_inner * 4) + 33))] = (compute[(((nn_inner * 4) + 33))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1176))]));
          compute[(((nn_inner * 4) + 41))] = (compute[(((nn_inner * 4) + 41))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1464))]));
          compute[(((nn_inner * 4) + 49))] = (compute[(((nn_inner * 4) + 49))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1752))]));
          compute[(((nn_inner * 4) + 57))] = (compute[(((nn_inner * 4) + 57))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 24))]));
          compute[(((nn_inner * 4) + 65))] = (compute[(((nn_inner * 4) + 65))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 312))]));
          compute[(((nn_inner * 4) + 73))] = (compute[(((nn_inner * 4) + 73))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 600))]));
          compute[(((nn_inner * 4) + 81))] = (compute[(((nn_inner * 4) + 81))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 888))]));
          compute[(((nn_inner * 4) + 89))] = (compute[(((nn_inner * 4) + 89))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1176))]));
          compute[(((nn_inner * 4) + 97))] = (compute[(((nn_inner * 4) + 97))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1464))]));
          compute[(((nn_inner * 4) + 105))] = (compute[(((nn_inner * 4) + 105))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1752))]));
          compute[(((nn_inner * 4) + 2))] = (compute[(((nn_inner * 4) + 2))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 48))]));
          compute[(((nn_inner * 4) + 10))] = (compute[(((nn_inner * 4) + 10))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 336))]));
          compute[(((nn_inner * 4) + 18))] = (compute[(((nn_inner * 4) + 18))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 624))]));
          compute[(((nn_inner * 4) + 26))] = (compute[(((nn_inner * 4) + 26))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 912))]));
          compute[(((nn_inner * 4) + 34))] = (compute[(((nn_inner * 4) + 34))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1200))]));
          compute[(((nn_inner * 4) + 42))] = (compute[(((nn_inner * 4) + 42))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1488))]));
          compute[(((nn_inner * 4) + 50))] = (compute[(((nn_inner * 4) + 50))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1776))]));
          compute[(((nn_inner * 4) + 58))] = (compute[(((nn_inner * 4) + 58))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 48))]));
          compute[(((nn_inner * 4) + 66))] = (compute[(((nn_inner * 4) + 66))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 336))]));
          compute[(((nn_inner * 4) + 74))] = (compute[(((nn_inner * 4) + 74))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 624))]));
          compute[(((nn_inner * 4) + 82))] = (compute[(((nn_inner * 4) + 82))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 912))]));
          compute[(((nn_inner * 4) + 90))] = (compute[(((nn_inner * 4) + 90))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1200))]));
          compute[(((nn_inner * 4) + 98))] = (compute[(((nn_inner * 4) + 98))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1488))]));
          compute[(((nn_inner * 4) + 106))] = (compute[(((nn_inner * 4) + 106))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1776))]));
          compute[(((nn_inner * 4) + 3))] = (compute[(((nn_inner * 4) + 3))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 72))]));
          compute[(((nn_inner * 4) + 11))] = (compute[(((nn_inner * 4) + 11))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 360))]));
          compute[(((nn_inner * 4) + 19))] = (compute[(((nn_inner * 4) + 19))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 648))]));
          compute[(((nn_inner * 4) + 27))] = (compute[(((nn_inner * 4) + 27))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 936))]));
          compute[(((nn_inner * 4) + 35))] = (compute[(((nn_inner * 4) + 35))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1224))]));
          compute[(((nn_inner * 4) + 43))] = (compute[(((nn_inner * 4) + 43))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1512))]));
          compute[(((nn_inner * 4) + 51))] = (compute[(((nn_inner * 4) + 51))] + (pad_temp_shared[(((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1800))]));
          compute[(((nn_inner * 4) + 59))] = (compute[(((nn_inner * 4) + 59))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 72))]));
          compute[(((nn_inner * 4) + 67))] = (compute[(((nn_inner * 4) + 67))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 360))]));
          compute[(((nn_inner * 4) + 75))] = (compute[(((nn_inner * 4) + 75))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 648))]));
          compute[(((nn_inner * 4) + 83))] = (compute[(((nn_inner * 4) + 83))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 936))]));
          compute[(((nn_inner * 4) + 91))] = (compute[(((nn_inner * 4) + 91))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1224))]));
          compute[(((nn_inner * 4) + 99))] = (compute[(((nn_inner * 4) + 99))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1512))]));
          compute[(((nn_inner * 4) + 107))] = (compute[(((nn_inner * 4) + 107))] + (pad_temp_shared[((((((nn_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 3984))] * input1_shared[((((((((int)threadIdx.x) / 83) * 96) + (rc_outer_inner * 6)) + rc_inner) + 1800))]));
        }
      }
    }
  }
  for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
    for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      T_add[((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))] = (compute[(((ax0_inner * 4) + ax1_inner))] + input2[((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 82668))] = (compute[((((ax0_inner * 4) + ax1_inner) + 8))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 82668))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 165336))] = (compute[((((ax0_inner * 4) + ax1_inner) + 16))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 165336))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 248004))] = (compute[((((ax0_inner * 4) + ax1_inner) + 24))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 248004))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 330672))] = (compute[((((ax0_inner * 4) + ax1_inner) + 32))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 330672))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 413340))] = (compute[((((ax0_inner * 4) + ax1_inner) + 40))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 413340))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 496008))] = (compute[((((ax0_inner * 4) + ax1_inner) + 48))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 496008))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1157352))] = (compute[((((ax0_inner * 4) + ax1_inner) + 56))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1157352))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1240020))] = (compute[((((ax0_inner * 4) + ax1_inner) + 64))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1240020))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1322688))] = (compute[((((ax0_inner * 4) + ax1_inner) + 72))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1322688))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1405356))] = (compute[((((ax0_inner * 4) + ax1_inner) + 80))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1405356))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1488024))] = (compute[((((ax0_inner * 4) + ax1_inner) + 88))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1488024))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1570692))] = (compute[((((ax0_inner * 4) + ax1_inner) + 96))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1570692))]);
      T_add[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1653360))] = (compute[((((ax0_inner * 4) + ax1_inner) + 104))] + input2[(((((((((((int)blockIdx.x) / 83) * 2314704) + (ax0_inner * 578676)) + ((((int)threadIdx.x) / 83) * 27556)) + (ax1_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1653360))]);
    }
  }
}

