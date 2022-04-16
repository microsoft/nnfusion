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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[28];
  __shared__ float pad_temp_shared[676];
  __shared__ float placeholder_shared[512];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
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
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 256))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 384))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 257))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 385))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 258))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 386))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 259))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 387))]));
  }
  compute[((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6272))] = compute_local[(7)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12544))] = compute_local[(14)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18816))] = compute_local[(21)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 14))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6286))] = compute_local[(8)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12558))] = compute_local[(15)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18830))] = compute_local[(22)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 28))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6300))] = compute_local[(9)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12572))] = compute_local[(16)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18844))] = compute_local[(23)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 42))] = compute_local[(3)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6314))] = compute_local[(10)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12586))] = compute_local[(17)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18858))] = compute_local[(24)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 56))] = compute_local[(4)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6328))] = compute_local[(11)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12600))] = compute_local[(18)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18872))] = compute_local[(25)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 70))] = compute_local[(5)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6342))] = compute_local[(12)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12614))] = compute_local[(19)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18886))] = compute_local[(26)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 84))] = compute_local[(6)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 6356))] = compute_local[(13)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 12628))] = compute_local[(20)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 18900))] = compute_local[(27)];
}

