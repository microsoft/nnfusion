//1_14_512_14_1_16
//128_256_56_56_512_1_2_VALID
//dim3 grid(1, 14, 512);
//dim3 block(14, 1, 16);

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
  float compute_local[32];
  __shared__ float pad_temp_shared[660];
  __shared__ float placeholder_shared[512];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) < 660) {
      pad_temp_shared[(((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)))] = placeholder[((((((((((int)blockIdx.z) >> 2) * 802816) + (rc_outer * 12544)) + ((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) / 165) * 3136)) + (((int)blockIdx.y) * 224)) + (((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) % 165) / 55) * 56)) + (((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) % 55)))];
    }
    if (((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) < 659) {
      pad_temp_shared[((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[((((((((((int)blockIdx.z) >> 2) * 802816) + (rc_outer * 12544)) + (((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) + 1) / 165) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) + 1) % 165) / 55) * 56)) + ((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) + 1) % 55)))];
    }
    if (((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) < 658) {
      pad_temp_shared[((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[((((((((((int)blockIdx.z) >> 2) * 802816) + (rc_outer * 12544)) + (((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) + 2) / 165) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) + 2) % 165) / 55) * 56)) + ((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 3)) + 2) % 55)))];
    }
    if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 3) >> 2)) < 128) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) < 512) {
        if (((int)threadIdx.x) < 11) {
          placeholder_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)))] = placeholder1[(((((((((int)blockIdx.z) & 3) * 32768) + (((int)threadIdx.z) * 2048)) + (((((int)threadIdx.x) * 3) >> 2) * 256)) + (rc_outer * 4)) + ((((int)threadIdx.x) * 3) & 3)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 3) + 1) >> 2)) < 128) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) < 511) {
        if (((int)threadIdx.x) < 11) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[(((((((((int)blockIdx.z) & 3) * 32768) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.x) * 3) + 1) >> 2) * 256)) + (rc_outer * 4)) + (((((int)threadIdx.x) * 3) + 1) & 3)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 3) + 2) >> 2)) < 128) {
      if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) < 510) {
        if (((int)threadIdx.x) < 10) {
          placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[(((((((((int)blockIdx.z) & 3) * 32768) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.x) * 3) + 2) >> 2) * 256)) + (rc_outer * 4)) + (((((int)threadIdx.x) * 3) + 2) & 3)))];
        }
      }
    }
    __syncthreads();
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
  }
  compute[(((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50176))] = compute_local[(16)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 14))] = compute_local[(8)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50190))] = compute_local[(24)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50204))] = compute_local[(17)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 42))] = compute_local[(9)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50218))] = compute_local[(25)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 784))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50960))] = compute_local[(18)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 798))] = compute_local[(10)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50974))] = compute_local[(26)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 812))] = compute_local[(3)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50988))] = compute_local[(19)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 826))] = compute_local[(11)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51002))] = compute_local[(27)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1568))] = compute_local[(4)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51744))] = compute_local[(20)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1582))] = compute_local[(12)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51758))] = compute_local[(28)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1596))] = compute_local[(5)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51772))] = compute_local[(21)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1610))] = compute_local[(13)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51786))] = compute_local[(29)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2352))] = compute_local[(6)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52528))] = compute_local[(22)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2366))] = compute_local[(14)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52542))] = compute_local[(30)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2380))] = compute_local[(7)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52556))] = compute_local[(23)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2394))] = compute_local[(15)];
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52570))] = compute_local[(31)];
}

