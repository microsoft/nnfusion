//896_1_1_448_1_1
//128_512_28_28_1024_1_2_VALID
//dim3 grid(896, 1, 1);
//dim3 block(448, 1, 1);

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
extern "C" __global__ void __launch_bounds__(448) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ T_add, float* __restrict__ input2) {
  float conv2d_nchw[64];
  __shared__ float pad_temp_shared[3456];
  __shared__ float input1_shared[4096];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[16] = 0.000000e+00f;
  conv2d_nchw[17] = 0.000000e+00f;
  conv2d_nchw[24] = 0.000000e+00f;
  conv2d_nchw[25] = 0.000000e+00f;
  conv2d_nchw[32] = 0.000000e+00f;
  conv2d_nchw[33] = 0.000000e+00f;
  conv2d_nchw[40] = 0.000000e+00f;
  conv2d_nchw[41] = 0.000000e+00f;
  conv2d_nchw[48] = 0.000000e+00f;
  conv2d_nchw[49] = 0.000000e+00f;
  conv2d_nchw[56] = 0.000000e+00f;
  conv2d_nchw[57] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[18] = 0.000000e+00f;
  conv2d_nchw[19] = 0.000000e+00f;
  conv2d_nchw[26] = 0.000000e+00f;
  conv2d_nchw[27] = 0.000000e+00f;
  conv2d_nchw[34] = 0.000000e+00f;
  conv2d_nchw[35] = 0.000000e+00f;
  conv2d_nchw[42] = 0.000000e+00f;
  conv2d_nchw[43] = 0.000000e+00f;
  conv2d_nchw[50] = 0.000000e+00f;
  conv2d_nchw[51] = 0.000000e+00f;
  conv2d_nchw[58] = 0.000000e+00f;
  conv2d_nchw[59] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[20] = 0.000000e+00f;
  conv2d_nchw[21] = 0.000000e+00f;
  conv2d_nchw[28] = 0.000000e+00f;
  conv2d_nchw[29] = 0.000000e+00f;
  conv2d_nchw[36] = 0.000000e+00f;
  conv2d_nchw[37] = 0.000000e+00f;
  conv2d_nchw[44] = 0.000000e+00f;
  conv2d_nchw[45] = 0.000000e+00f;
  conv2d_nchw[52] = 0.000000e+00f;
  conv2d_nchw[53] = 0.000000e+00f;
  conv2d_nchw[60] = 0.000000e+00f;
  conv2d_nchw[61] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  conv2d_nchw[22] = 0.000000e+00f;
  conv2d_nchw[23] = 0.000000e+00f;
  conv2d_nchw[30] = 0.000000e+00f;
  conv2d_nchw[31] = 0.000000e+00f;
  conv2d_nchw[38] = 0.000000e+00f;
  conv2d_nchw[39] = 0.000000e+00f;
  conv2d_nchw[46] = 0.000000e+00f;
  conv2d_nchw[47] = 0.000000e+00f;
  conv2d_nchw[54] = 0.000000e+00f;
  conv2d_nchw[55] = 0.000000e+00f;
  conv2d_nchw[62] = 0.000000e+00f;
  conv2d_nchw[63] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = input0[(((((((((int)blockIdx.x) / 56) * 3211264) + ((((int)threadIdx.x) / 432) * 401408)) + (rc_outer_outer * 12544)) + (((((int)threadIdx.x) % 432) / 27) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + (((int)threadIdx.x) % 27))];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = input0[(((((((((int)blockIdx.x) / 56) * 3211264) + (((((int)threadIdx.x) + 448) / 432) * 401408)) + (rc_outer_outer * 12544)) + ((((((int)threadIdx.x) + 16) % 432) / 27) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 16) % 27))];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = input0[(((((((((int)blockIdx.x) / 56) * 3211264) + (((((int)threadIdx.x) + 896) / 432) * 401408)) + (rc_outer_outer * 12544)) + ((((((int)threadIdx.x) + 32) % 432) / 27) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 5) % 27))];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = input0[(((((((((int)blockIdx.x) / 56) * 3211264) + (((((int)threadIdx.x) + 1344) / 432) * 401408)) + (rc_outer_outer * 12544)) + ((((((int)threadIdx.x) + 48) % 432) / 27) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 21) % 27))];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = input0[(((((((((int)blockIdx.x) / 56) * 3211264) + (((((int)threadIdx.x) + 1792) / 432) * 401408)) + (rc_outer_outer * 12544)) + ((((((int)threadIdx.x) + 64) % 432) / 27) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 10) % 27))];
    pad_temp_shared[(((int)threadIdx.x) + 2240)] = input0[(((((((((int)blockIdx.x) / 56) * 3211264) + (((((int)threadIdx.x) + 2240) / 432) * 401408)) + (rc_outer_outer * 12544)) + ((((((int)threadIdx.x) + 80) % 432) / 27) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 26) % 27))];
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = input0[(((((((((int)blockIdx.x) / 56) * 3211264) + (((((int)threadIdx.x) + 2688) / 432) * 401408)) + (rc_outer_outer * 12544)) + ((((((int)threadIdx.x) + 96) % 432) / 27) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 15) % 27))];
    if (((int)threadIdx.x) < 320) {
      pad_temp_shared[(((int)threadIdx.x) + 3136)] = input0[(((((((((int)blockIdx.x) / 56) * 3211264) + (((((int)threadIdx.x) + 3136) / 432) * 401408)) + (rc_outer_outer * 12544)) + ((((((int)threadIdx.x) + 112) % 432) / 27) * 784)) + ((((int)blockIdx.x) % 14) * 56)) + ((((int)threadIdx.x) + 4) % 27))];
    }
    input1_shared[((int)threadIdx.x)] = input1[((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    input1_shared[(((int)threadIdx.x) + 448)] = input1[(((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336)];
    input1_shared[(((int)threadIdx.x) + 896)] = input1[(((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 28672)];
    input1_shared[(((int)threadIdx.x) + 1344)] = input1[(((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 43008)];
    input1_shared[(((int)threadIdx.x) + 1792)] = input1[(((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 57344)];
    input1_shared[(((int)threadIdx.x) + 2240)] = input1[(((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 71680)];
    input1_shared[(((int)threadIdx.x) + 2688)] = input1[(((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 86016)];
    input1_shared[(((int)threadIdx.x) + 3136)] = input1[(((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 100352)];
    input1_shared[(((int)threadIdx.x) + 3584)] = input1[(((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 114688)];
    if (((int)threadIdx.x) < 64) {
      input1_shared[(((int)threadIdx.x) + 4032)] = input1[(((((((((int)blockIdx.x) % 56) / 14) * 131072) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 129024)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2))] * input1_shared[(((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2))] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 432)] * input1_shared[(((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 432)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 864)] * input1_shared[(((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 864)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1296)] * input1_shared[(((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1296)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[32] = (conv2d_nchw[32] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1728)] * input1_shared[(((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw[33] = (conv2d_nchw[33] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1728)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[40] = (conv2d_nchw[40] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2160)] * input1_shared[(((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw[41] = (conv2d_nchw[41] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2160)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[48] = (conv2d_nchw[48] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2592)] * input1_shared[(((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw[49] = (conv2d_nchw[49] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2592)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[56] = (conv2d_nchw[56] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3024)] * input1_shared[(((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2))]));
      conv2d_nchw[57] = (conv2d_nchw[57] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3024)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 27)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 27)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 459)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 459)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 891)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 891)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1323)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1323)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[32] = (conv2d_nchw[32] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1755)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[33] = (conv2d_nchw[33] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1755)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[40] = (conv2d_nchw[40] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2187)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[41] = (conv2d_nchw[41] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2187)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[48] = (conv2d_nchw[48] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2619)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[49] = (conv2d_nchw[49] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2619)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[56] = (conv2d_nchw[56] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3051)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[57] = (conv2d_nchw[57] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3051)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2))] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2))] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 432)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 432)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 864)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 864)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1296)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1296)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw[34] = (conv2d_nchw[34] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1728)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw[35] = (conv2d_nchw[35] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1728)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw[42] = (conv2d_nchw[42] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2160)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw[43] = (conv2d_nchw[43] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2160)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw[50] = (conv2d_nchw[50] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2592)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw[51] = (conv2d_nchw[51] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2592)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw[58] = (conv2d_nchw[58] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3024)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 32)]));
      conv2d_nchw[59] = (conv2d_nchw[59] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3024)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 48)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 27)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 27)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 459)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 459)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 891)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 891)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1323)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1323)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw[34] = (conv2d_nchw[34] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1755)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw[35] = (conv2d_nchw[35] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1755)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw[42] = (conv2d_nchw[42] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2187)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw[43] = (conv2d_nchw[43] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2187)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw[50] = (conv2d_nchw[50] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2619)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw[51] = (conv2d_nchw[51] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2619)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw[58] = (conv2d_nchw[58] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3051)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 33)]));
      conv2d_nchw[59] = (conv2d_nchw[59] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3051)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 49)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2))] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2))] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 432)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 432)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 864)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 864)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1296)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1296)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw[36] = (conv2d_nchw[36] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1728)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[37] = (conv2d_nchw[37] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1728)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw[44] = (conv2d_nchw[44] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2160)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[45] = (conv2d_nchw[45] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2160)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw[52] = (conv2d_nchw[52] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2592)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[53] = (conv2d_nchw[53] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2592)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw[60] = (conv2d_nchw[60] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3024)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 64)]));
      conv2d_nchw[61] = (conv2d_nchw[61] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3024)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 80)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 27)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 27)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 459)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 459)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 891)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 891)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1323)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1323)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw[36] = (conv2d_nchw[36] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1755)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[37] = (conv2d_nchw[37] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1755)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw[44] = (conv2d_nchw[44] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2187)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[45] = (conv2d_nchw[45] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2187)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw[52] = (conv2d_nchw[52] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2619)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[53] = (conv2d_nchw[53] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2619)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw[60] = (conv2d_nchw[60] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3051)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 65)]));
      conv2d_nchw[61] = (conv2d_nchw[61] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3051)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 81)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2))] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2))] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 432)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 432)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 864)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 864)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1296)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1296)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw[38] = (conv2d_nchw[38] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1728)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw[39] = (conv2d_nchw[39] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1728)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw[46] = (conv2d_nchw[46] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2160)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw[47] = (conv2d_nchw[47] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2160)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw[54] = (conv2d_nchw[54] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2592)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw[55] = (conv2d_nchw[55] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2592)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw[62] = (conv2d_nchw[62] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3024)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 96)]));
      conv2d_nchw[63] = (conv2d_nchw[63] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3024)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 112)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 27)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 27)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 459)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 459)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 891)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 891)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1323)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1323)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw[38] = (conv2d_nchw[38] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1755)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw[39] = (conv2d_nchw[39] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 1755)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw[46] = (conv2d_nchw[46] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2187)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw[47] = (conv2d_nchw[47] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2187)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw[54] = (conv2d_nchw[54] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2619)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw[55] = (conv2d_nchw[55] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 2619)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 113)]));
      conv2d_nchw[62] = (conv2d_nchw[62] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3051)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 97)]));
      conv2d_nchw[63] = (conv2d_nchw[63] + (pad_temp_shared[(((rc_outer_inner * 54) + ((((int)threadIdx.x) % 14) * 2)) + 3051)] * input1_shared[((((((int)threadIdx.x) / 14) * 128) + (rc_outer_inner * 2)) + 113)]));
    }
  }
  for (int ax0_inner = 0; ax0_inner < 8; ++ax0_inner) {
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      T_add[((((((((((int)blockIdx.x) / 56) * 1605632) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 56) / 14) * 50176)) + ((((int)threadIdx.x) / 14) * 1568)) + (ax1_inner * 196)) + ((((int)blockIdx.x) % 14) * 14)) + (((int)threadIdx.x) % 14))] = (conv2d_nchw[((ax0_inner * 8) + ax1_inner)] + input2[((((((((((int)blockIdx.x) / 56) * 1605632) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 56) / 14) * 50176)) + ((((int)threadIdx.x) / 14) * 1568)) + (ax1_inner * 196)) + ((((int)blockIdx.x) % 14) * 14)) + (((int)threadIdx.x) % 14))]);
    }
  }
}

