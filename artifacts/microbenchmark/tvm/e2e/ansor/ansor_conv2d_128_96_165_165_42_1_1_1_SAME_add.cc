//31680_1_1_44_1_1
//128_96_165_165_42_1_1_SAME
//dim3 grid(31680, 1, 1);
//dim3 block(44, 1, 1);

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
extern "C" __global__ void __launch_bounds__(44) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ T_add, float* __restrict__ input2) {
  float conv2d_nchw[105];
  __shared__ float pad_temp_shared[1760];
  __shared__ float input1_shared[672];
  for (int ff_inner_init = 0; ff_inner_init < 21; ++ff_inner_init) {
    conv2d_nchw[(ff_inner_init * 5)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 5) + 1)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 5) + 2)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 5) + 3)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 5) + 4)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 6; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = input0[(((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + (((int)threadIdx.x) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 44)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 44) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 44) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 4) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 88)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 88) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 33) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 3) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 132)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 132) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 22) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 2) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 176)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 176) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 11) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 1) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 220)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + (((int)threadIdx.x) % 5)) + 108900)];
    pad_temp_shared[(((int)threadIdx.x) + 264)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 264) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 44) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 4) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 308)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 308) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 33) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 3) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 352)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 352) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 22) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 2) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 396)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 396) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 11) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 1) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 440)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + (((int)threadIdx.x) % 5)) + 217800)];
    pad_temp_shared[(((int)threadIdx.x) + 484)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 484) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 44) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 4) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 528)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 528) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 33) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 3) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 572)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 572) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 22) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 2) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 616)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 616) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 11) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 1) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 660)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + (((int)threadIdx.x) % 5)) + 326700)];
    pad_temp_shared[(((int)threadIdx.x) + 704)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 704) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 44) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 4) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 748)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 748) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 33) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 3) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 792)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 792) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 22) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 2) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 836)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)threadIdx.x) + 836) / 55) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 11) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 1) % 5))];
    pad_temp_shared[(((int)threadIdx.x) + 880)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + (((int)threadIdx.x) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 924)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 4) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 44) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 4) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 968)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 8) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 33) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 3) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1012)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 12) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 22) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 2) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1056)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 16) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 11) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 1) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1100)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + (((int)threadIdx.x) % 5)) + 2722500)];
    pad_temp_shared[(((int)threadIdx.x) + 1144)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 24) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 44) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 4) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1188)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 28) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 33) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 3) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1232)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 32) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 22) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 2) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1276)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 36) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 11) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 1) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1320)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + (((int)threadIdx.x) % 5)) + 2831400)];
    pad_temp_shared[(((int)threadIdx.x) + 1364)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 44) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 44) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 4) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1408)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 48) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 33) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 3) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1452)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 52) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 22) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 2) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1496)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 56) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 11) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 1) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1540)] = input0[((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + (((int)threadIdx.x) % 5)) + 2940300)];
    pad_temp_shared[(((int)threadIdx.x) + 1584)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 64) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 44) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 4) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1628)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 68) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 33) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 3) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1672)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 72) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 22) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 2) % 5)) + 2613600)];
    pad_temp_shared[(((int)threadIdx.x) + 1716)] = input0[(((((((((((int)blockIdx.x) / 495) * 5227200) + (rc_outer_outer * 435600)) + (((((((int)threadIdx.x) / 11) + 76) % 80) / 5) * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((((int)threadIdx.x) + 11) % 55) / 5) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ((((int)threadIdx.x) + 1) % 5)) + 2613600)];
    input1_shared[(((int)threadIdx.x) * 4)] = input1[((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4))];
    input1_shared[((((int)threadIdx.x) * 4) + 1)] = input1[((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 1) & 15))];
    input1_shared[((((int)threadIdx.x) * 4) + 2)] = input1[((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 2) & 15))];
    input1_shared[((((int)threadIdx.x) * 4) + 3)] = input1[((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 3) & 15))];
    input1_shared[((((int)threadIdx.x) * 4) + 176)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1056)];
    input1_shared[((((int)threadIdx.x) * 4) + 177)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 1) & 15)) + 1056)];
    input1_shared[((((int)threadIdx.x) * 4) + 178)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 2) & 15)) + 1056)];
    input1_shared[((((int)threadIdx.x) * 4) + 179)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 3) & 15)) + 1056)];
    input1_shared[((((int)threadIdx.x) * 4) + 352)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2112)];
    input1_shared[((((int)threadIdx.x) * 4) + 353)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 1) & 15)) + 2112)];
    input1_shared[((((int)threadIdx.x) * 4) + 354)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 2) & 15)) + 2112)];
    input1_shared[((((int)threadIdx.x) * 4) + 355)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 3) & 15)) + 2112)];
    if (((int)threadIdx.x) < 36) {
      input1_shared[((((int)threadIdx.x) * 4) + 528)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 3168)];
    }
    if (((int)threadIdx.x) < 36) {
      input1_shared[((((int)threadIdx.x) * 4) + 529)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 1) & 15)) + 3168)];
    }
    if (((int)threadIdx.x) < 36) {
      input1_shared[((((int)threadIdx.x) * 4) + 530)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 2) & 15)) + 3168)];
    }
    if (((int)threadIdx.x) < 36) {
      input1_shared[((((int)threadIdx.x) * 4) + 531)] = input1[(((((((int)threadIdx.x) >> 2) * 96) + (rc_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 3) & 15)) + 3168)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
        for (int ff_inner = 0; ff_inner < 21; ++ff_inner) {
          conv2d_nchw[(ff_inner * 5)] = (conv2d_nchw[(ff_inner * 5)] + (pad_temp_shared[(((((((int)threadIdx.x) / 22) * 880) + (rc_outer_inner * 440)) + (rc_inner * 55)) + ((((int)threadIdx.x) % 11) * 5))] * input1_shared[((((((((int)threadIdx.x) % 22) / 11) * 336) + (ff_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw[((ff_inner * 5) + 1)] = (conv2d_nchw[((ff_inner * 5) + 1)] + (pad_temp_shared[((((((((int)threadIdx.x) / 22) * 880) + (rc_outer_inner * 440)) + (rc_inner * 55)) + ((((int)threadIdx.x) % 11) * 5)) + 1)] * input1_shared[((((((((int)threadIdx.x) % 22) / 11) * 336) + (ff_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw[((ff_inner * 5) + 2)] = (conv2d_nchw[((ff_inner * 5) + 2)] + (pad_temp_shared[((((((((int)threadIdx.x) / 22) * 880) + (rc_outer_inner * 440)) + (rc_inner * 55)) + ((((int)threadIdx.x) % 11) * 5)) + 2)] * input1_shared[((((((((int)threadIdx.x) % 22) / 11) * 336) + (ff_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw[((ff_inner * 5) + 3)] = (conv2d_nchw[((ff_inner * 5) + 3)] + (pad_temp_shared[((((((((int)threadIdx.x) / 22) * 880) + (rc_outer_inner * 440)) + (rc_inner * 55)) + ((((int)threadIdx.x) % 11) * 5)) + 3)] * input1_shared[((((((((int)threadIdx.x) % 22) / 11) * 336) + (ff_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
          conv2d_nchw[((ff_inner * 5) + 4)] = (conv2d_nchw[((ff_inner * 5) + 4)] + (pad_temp_shared[((((((((int)threadIdx.x) / 22) * 880) + (rc_outer_inner * 440)) + (rc_inner * 55)) + ((((int)threadIdx.x) % 11) * 5)) + 4)] * input1_shared[((((((((int)threadIdx.x) % 22) / 11) * 336) + (ff_inner * 16)) + (rc_outer_inner * 8)) + rc_inner)]));
        }
      }
    }
  }
  for (int ax1_inner = 0; ax1_inner < 21; ++ax1_inner) {
    for (int ax3_inner = 0; ax3_inner < 5; ++ax3_inner) {
      T_add[((((((((((int)blockIdx.x) / 495) * 2286900) + ((((int)threadIdx.x) / 11) * 571725)) + (ax1_inner * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) % 11) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ax3_inner)] = (conv2d_nchw[((ax1_inner * 5) + ax3_inner)] + input2[((((((((((int)blockIdx.x) / 495) * 2286900) + ((((int)threadIdx.x) / 11) * 571725)) + (ax1_inner * 27225)) + (((((int)blockIdx.x) % 495) / 33) * 1815)) + ((((int)threadIdx.x) % 11) * 165)) + ((((int)blockIdx.x) % 33) * 5)) + ax3_inner)]);
    }
  }
}

