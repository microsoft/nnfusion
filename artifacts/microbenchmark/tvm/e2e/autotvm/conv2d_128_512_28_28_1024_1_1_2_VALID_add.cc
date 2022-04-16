//2_2_1024_7_1_32
//128_512_28_28_1024_1_2_VALID
//dim3 grid(2, 2, 1024);
//dim3 block(7, 1, 32);

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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[28];
  __shared__ float pad_temp_shared[676];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) < 676) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[(((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)))] = placeholder[(((((((((((int)blockIdx.z) >> 3) * 401408) + (rc_outer * 3136)) + ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) / 169) * 784)) + (((int)blockIdx.y) * 392)) + (((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) % 169) / 13) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) % 13)))];
      }
    }
    if (((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) < 675) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 1))] = placeholder[(((((((((((int)blockIdx.z) >> 3) * 401408) + (rc_outer * 3136)) + (((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 1) / 169) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 1) % 169) / 13) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 1) % 13)))];
      }
    }
    if (((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) < 674) {
      if (((int)threadIdx.x) < 5) {
        pad_temp_shared[((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 2))] = placeholder[(((((((((((int)blockIdx.z) >> 3) * 401408) + (rc_outer * 3136)) + (((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 2) / 169) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 2) % 169) / 13) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 2) % 13)))];
      }
    }
    if (((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) < 673) {
      if (((int)threadIdx.x) < 5) {
        pad_temp_shared[((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 3))] = placeholder[(((((((((((int)blockIdx.z) >> 3) * 401408) + (rc_outer * 3136)) + (((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 3) / 169) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 3) % 169) / 13) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 4)) + 3) % 13)))];
      }
    }
    if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 3) >> 2)) < 128) {
      if (((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) < 512) {
        if (((int)threadIdx.x) < 6) {
          placeholder_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 65536) + (((int)threadIdx.z) * 2048)) + (((((int)threadIdx.x) * 3) >> 2) * 512)) + (rc_outer * 4)) + ((((int)threadIdx.x) * 3) & 3)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + 1) >> 2)) < 128) {
      if (((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) < 511) {
        if (((int)threadIdx.x) < 5) {
          placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 65536) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.x) * 3) + 1) >> 2) * 512)) + (rc_outer * 4)) + (((((int)threadIdx.x) * 3) + 1) & 3)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + 2) >> 2)) < 128) {
      if (((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) < 510) {
        if (((int)threadIdx.x) < 5) {
          placeholder_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[(((((((((int)blockIdx.z) & 7) * 65536) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.x) * 3) + 2) >> 2) * 512)) + (rc_outer * 4)) + (((((int)threadIdx.x) * 3) + 2) & 3)))];
        }
      }
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
  }
  T_add[((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = (compute[(0)] + input2[((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6272))] = (compute[(7)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6272))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12544))] = (compute[(14)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12544))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18816))] = (compute[(21)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18816))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 14))] = (compute[(1)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 14))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6286))] = (compute[(8)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6286))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12558))] = (compute[(15)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12558))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18830))] = (compute[(22)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18830))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 28))] = (compute[(2)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 28))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6300))] = (compute[(9)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6300))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12572))] = (compute[(16)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12572))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18844))] = (compute[(23)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18844))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 42))] = (compute[(3)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 42))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6314))] = (compute[(10)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6314))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12586))] = (compute[(17)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12586))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18858))] = (compute[(24)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18858))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 56))] = (compute[(4)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 56))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6328))] = (compute[(11)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6328))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12600))] = (compute[(18)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12600))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18872))] = (compute[(25)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18872))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 70))] = (compute[(5)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 70))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6342))] = (compute[(12)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6342))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12614))] = (compute[(19)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12614))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18886))] = (compute[(26)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18886))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 84))] = (compute[(6)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 84))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6356))] = (compute[(13)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6356))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12628))] = (compute[(20)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12628))]);
  T_add[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18900))] = (compute[(27)] + input2[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18900))]);
}

