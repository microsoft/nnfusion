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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[32];
  __shared__ float pad_temp_shared[660];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(28)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(29)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(30)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(31)] = 0.000000e+00f;
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
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 110))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 138))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 193))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 275))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 303))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 330))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 358))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 440))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
    compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 495))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 523))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 605))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
    compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 633))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
  }
  T_add[(((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)))] = (compute[(0)] + input2[(((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50176))] = (compute[(16)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50176))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 14))] = (compute[(8)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 14))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50190))] = (compute[(24)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50190))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = (compute[(1)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50204))] = (compute[(17)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50204))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 42))] = (compute[(9)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 42))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50218))] = (compute[(25)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50218))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 784))] = (compute[(2)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 784))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50960))] = (compute[(18)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50960))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 798))] = (compute[(10)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 798))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50974))] = (compute[(26)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50974))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 812))] = (compute[(3)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 812))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50988))] = (compute[(19)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50988))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 826))] = (compute[(11)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 826))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51002))] = (compute[(27)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51002))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1568))] = (compute[(4)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1568))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51744))] = (compute[(20)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51744))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1582))] = (compute[(12)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1582))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51758))] = (compute[(28)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51758))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1596))] = (compute[(5)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1596))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51772))] = (compute[(21)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51772))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1610))] = (compute[(13)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 1610))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51786))] = (compute[(29)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 51786))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2352))] = (compute[(6)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2352))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52528))] = (compute[(22)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52528))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2366))] = (compute[(14)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2366))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52542))] = (compute[(30)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52542))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2380))] = (compute[(7)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2380))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52556))] = (compute[(23)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52556))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2394))] = (compute[(15)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 2394))]);
  T_add[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52570))] = (compute[(31)] + input2[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 52570))]);
}

