//1_1_21504_11_1_4
//128_672_11_11_5_1_SAME
//dim3 grid(1, 1, 21504);
//dim3 block(11, 1, 4);

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
extern "C" __global__ void __launch_bounds__(44) depthwise_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ DepthwiseConv2d) {
  __shared__ float PaddedInput_shared[900];
  __shared__ float placeholder_shared[100];
  float PaddedInput_shared_local[75];
  float placeholder_shared_local[25];
  float DepthwiseConv2d_local[11];
  PaddedInput_shared[(((((int)threadIdx.z) * 11) + ((int)threadIdx.x)))] = ((((30 <= ((((int)threadIdx.z) * 11) + ((int)threadIdx.x))) && (2 <= (((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) % 15))) && ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) % 15) < 13)) ? placeholder[(((((((int)blockIdx.z) * 484) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) / 15) * 11)) + (((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 44))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 14) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 14) % 15) < 13)) ? placeholder[(((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 44) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 14) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 88))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 13) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 13) % 15) < 13)) ? placeholder[(((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 88) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 13) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 132))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 12) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 12) % 15) < 13)) ? placeholder[(((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 132) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 12) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 176))] = ((((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) < 19) && (2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 11) % 15))) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 11) % 15) < 13)) ? placeholder[(((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 176) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 11) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 220))] = (((((30 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 220) % 225)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 220) % 225) < 195)) && (2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 10) % 15))) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 10) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 220) / 225) * 121)) + ((((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 220) % 225) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 10) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 264))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 9) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 9) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 264) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 39) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 9) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 308))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 8) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 8) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 308) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 83) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 8) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 352))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 7) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 7) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 352) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 127) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 7) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 396))] = ((((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) < 24) && (2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 6) % 15))) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 6) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 396) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 171) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 6) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 440))] = (((((30 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 215) % 225)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 215) % 225) < 195)) && (2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 5) % 15))) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 5) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 440) / 225) * 121)) + ((((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 215) % 225) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 5) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 484))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 4) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 4) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 484) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 34) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 4) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 528))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 3) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 3) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 528) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 78) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 3) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 572))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 2) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 2) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 572) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 122) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 2) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 616))] = ((((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) < 29) && (2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 1) % 15))) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 1) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 616) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 166) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 1) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 660))] = (((((30 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 210) % 225)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 210) % 225) < 195)) && (2 <= (((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) % 15))) && ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 660) / 225) * 121)) + ((((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 210) % 225) / 15) * 11)) + (((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 704))] = ((((1 <= ((((int)threadIdx.z) * 11) + ((int)threadIdx.x))) && (2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 14) % 15))) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 14) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 704) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 29) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 14) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 748))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 13) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 13) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 748) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 73) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 13) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 792))] = (((2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 12) % 15)) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 12) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 792) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 117) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 12) % 15)) - 24))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 836))] = ((((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) < 34) && (2 <= ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 11) % 15))) && (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 11) % 15) < 13)) ? placeholder[((((((((int)blockIdx.z) * 484) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 836) / 225) * 121)) + (((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 161) / 15) * 11)) + ((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 11) % 15)) - 24))] : 0.000000e+00f);
  if (((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) < 20) {
    if (((int)threadIdx.z) < 2) {
      PaddedInput_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 880))] = 0.000000e+00f;
    }
  }
  placeholder_shared[(((((int)threadIdx.z) * 11) + ((int)threadIdx.x)))] = placeholder1[(((((((int)blockIdx.z) % 168) * 100) + (((int)threadIdx.z) * 11)) + ((int)threadIdx.x)))];
  placeholder_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 44))] = placeholder1[((((((((int)blockIdx.z) % 168) * 100) + (((int)threadIdx.z) * 11)) + ((int)threadIdx.x)) + 44))];
  if (((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) < 12) {
    if (((int)threadIdx.z) < 2) {
      placeholder_shared[((((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) + 88))] = placeholder1[((((((((int)blockIdx.z) % 168) * 100) + (((int)threadIdx.z) * 11)) + ((int)threadIdx.x)) + 88))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.z) * 225) + ((int)threadIdx.x)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 3))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 4))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 15))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 16))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 17))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 18))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 19))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 30))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 31))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 32))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 33))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 34))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 45))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 46))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 47))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 48))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 49))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 60))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 61))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 62))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 63))];
  PaddedInput_shared_local[(24)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 64))];
  PaddedInput_shared_local[(25)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 75))];
  PaddedInput_shared_local[(26)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 76))];
  PaddedInput_shared_local[(27)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 77))];
  PaddedInput_shared_local[(28)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 78))];
  PaddedInput_shared_local[(29)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 79))];
  PaddedInput_shared_local[(30)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 90))];
  PaddedInput_shared_local[(31)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 91))];
  PaddedInput_shared_local[(32)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 92))];
  PaddedInput_shared_local[(33)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 93))];
  PaddedInput_shared_local[(34)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 94))];
  PaddedInput_shared_local[(35)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 105))];
  PaddedInput_shared_local[(36)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 106))];
  PaddedInput_shared_local[(37)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 107))];
  PaddedInput_shared_local[(38)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 108))];
  PaddedInput_shared_local[(39)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 109))];
  PaddedInput_shared_local[(40)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 120))];
  PaddedInput_shared_local[(41)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 121))];
  PaddedInput_shared_local[(42)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 122))];
  PaddedInput_shared_local[(43)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 123))];
  PaddedInput_shared_local[(44)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 124))];
  PaddedInput_shared_local[(45)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 135))];
  PaddedInput_shared_local[(46)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 136))];
  PaddedInput_shared_local[(47)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 137))];
  PaddedInput_shared_local[(48)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 138))];
  PaddedInput_shared_local[(49)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 139))];
  PaddedInput_shared_local[(50)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 150))];
  PaddedInput_shared_local[(51)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 151))];
  PaddedInput_shared_local[(52)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 152))];
  PaddedInput_shared_local[(53)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 153))];
  PaddedInput_shared_local[(54)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 154))];
  PaddedInput_shared_local[(55)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 165))];
  PaddedInput_shared_local[(56)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 166))];
  PaddedInput_shared_local[(57)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 167))];
  PaddedInput_shared_local[(58)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 168))];
  PaddedInput_shared_local[(59)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 169))];
  PaddedInput_shared_local[(60)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 180))];
  PaddedInput_shared_local[(61)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 181))];
  PaddedInput_shared_local[(62)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 182))];
  PaddedInput_shared_local[(63)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 183))];
  PaddedInput_shared_local[(64)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 184))];
  PaddedInput_shared_local[(65)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 195))];
  PaddedInput_shared_local[(66)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 196))];
  PaddedInput_shared_local[(67)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 197))];
  PaddedInput_shared_local[(68)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 198))];
  PaddedInput_shared_local[(69)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 199))];
  PaddedInput_shared_local[(70)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 210))];
  PaddedInput_shared_local[(71)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 211))];
  PaddedInput_shared_local[(72)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 212))];
  PaddedInput_shared_local[(73)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 213))];
  PaddedInput_shared_local[(74)] = PaddedInput_shared[((((((int)threadIdx.z) * 225) + ((int)threadIdx.x)) + 214))];
  placeholder_shared_local[(0)] = placeholder_shared[((((int)threadIdx.z) * 25))];
  placeholder_shared_local[(1)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 1))];
  placeholder_shared_local[(2)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 2))];
  placeholder_shared_local[(3)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 3))];
  placeholder_shared_local[(4)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 4))];
  placeholder_shared_local[(5)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 5))];
  placeholder_shared_local[(6)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 6))];
  placeholder_shared_local[(7)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 7))];
  placeholder_shared_local[(8)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 8))];
  placeholder_shared_local[(9)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 9))];
  placeholder_shared_local[(10)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 10))];
  placeholder_shared_local[(11)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 11))];
  placeholder_shared_local[(12)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 12))];
  placeholder_shared_local[(13)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 13))];
  placeholder_shared_local[(14)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 14))];
  placeholder_shared_local[(15)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 15))];
  placeholder_shared_local[(16)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 16))];
  placeholder_shared_local[(17)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 17))];
  placeholder_shared_local[(18)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 18))];
  placeholder_shared_local[(19)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 19))];
  placeholder_shared_local[(20)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 20))];
  placeholder_shared_local[(21)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 21))];
  placeholder_shared_local[(22)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 22))];
  placeholder_shared_local[(23)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 23))];
  placeholder_shared_local[(24)] = placeholder_shared[(((((int)threadIdx.z) * 25) + 24))];
  for (int i_c = 0; i_c < 11; ++i_c) {
    DepthwiseConv2d_local[(i_c)] = 0.000000e+00f;
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[((i_c * 5))] * placeholder_shared_local[(0)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 1))] * placeholder_shared_local[(1)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 2))] * placeholder_shared_local[(2)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 3))] * placeholder_shared_local[(3)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 4))] * placeholder_shared_local[(4)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 5))] * placeholder_shared_local[(5)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 6))] * placeholder_shared_local[(6)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 7))] * placeholder_shared_local[(7)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 8))] * placeholder_shared_local[(8)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 9))] * placeholder_shared_local[(9)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 10))] * placeholder_shared_local[(10)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 11))] * placeholder_shared_local[(11)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 12))] * placeholder_shared_local[(12)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 13))] * placeholder_shared_local[(13)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 14))] * placeholder_shared_local[(14)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 15))] * placeholder_shared_local[(15)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 16))] * placeholder_shared_local[(16)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 17))] * placeholder_shared_local[(17)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 18))] * placeholder_shared_local[(18)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 19))] * placeholder_shared_local[(19)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 20))] * placeholder_shared_local[(20)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 21))] * placeholder_shared_local[(21)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 22))] * placeholder_shared_local[(22)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 23))] * placeholder_shared_local[(23)]));
    DepthwiseConv2d_local[(i_c)] = (DepthwiseConv2d_local[(i_c)] + (PaddedInput_shared_local[(((i_c * 5) + 24))] * placeholder_shared_local[(24)]));
  }
  DepthwiseConv2d[((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 11))] = DepthwiseConv2d_local[(1)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 22))] = DepthwiseConv2d_local[(2)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 33))] = DepthwiseConv2d_local[(3)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 44))] = DepthwiseConv2d_local[(4)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 55))] = DepthwiseConv2d_local[(5)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 66))] = DepthwiseConv2d_local[(6)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 77))] = DepthwiseConv2d_local[(7)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 88))] = DepthwiseConv2d_local[(8)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 99))] = DepthwiseConv2d_local[(9)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 484) + (((int)threadIdx.z) * 121)) + ((int)threadIdx.x)) + 110))] = DepthwiseConv2d_local[(10)];
}

