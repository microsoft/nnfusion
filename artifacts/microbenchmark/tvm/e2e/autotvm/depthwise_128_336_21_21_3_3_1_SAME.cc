//1_1_43008_21_3_1
//128_336_21_21_3_1_SAME
//dim3 grid(1, 1, 43008);
//dim3 block(21, 3, 1);

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
extern "C" __global__ void __launch_bounds__(63) depthwise_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ DepthwiseConv2d) {
  __shared__ float PaddedInput_shared[529];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[27];
  float placeholder_shared_local[9];
  float DepthwiseConv2d_local[7];
  PaddedInput_shared[(((((int)threadIdx.y) * 21) + ((int)threadIdx.x)))] = ((((23 <= ((((int)threadIdx.y) * 21) + ((int)threadIdx.x))) && (1 <= (((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) % 23))) && ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) % 23) < 22)) ? placeholder[(((((((int)blockIdx.z) * 441) + ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) / 23) * 21)) + (((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) % 23)) - 22))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 63))] = (((1 <= ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 17) % 23)) && (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 17) % 23) < 22)) ? placeholder[(((((((int)blockIdx.z) * 441) + (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 63) / 23) * 21)) + ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 17) % 23)) - 22))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 126))] = (((1 <= ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 11) % 23)) && (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 11) % 23) < 22)) ? placeholder[(((((((int)blockIdx.z) * 441) + (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 126) / 23) * 21)) + ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 11) % 23)) - 22))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 189))] = (((1 <= ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 5) % 23)) && (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 5) % 23) < 22)) ? placeholder[(((((((int)blockIdx.z) * 441) + (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 189) / 23) * 21)) + ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 5) % 23)) - 22))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 252))] = (((1 <= ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 22) % 23)) && (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 22) % 23) < 22)) ? placeholder[(((((((int)blockIdx.z) * 441) + (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 252) / 23) * 21)) + ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 22) % 23)) - 22))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 315))] = (((1 <= ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 16) % 23)) && (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 16) % 23) < 22)) ? placeholder[(((((((int)blockIdx.z) * 441) + (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 315) / 23) * 21)) + ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 16) % 23)) - 22))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 378))] = (((1 <= ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 10) % 23)) && (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 10) % 23) < 22)) ? placeholder[(((((((int)blockIdx.z) * 441) + (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 378) / 23) * 21)) + ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 10) % 23)) - 22))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 441))] = (((1 <= ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 4) % 23)) && (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 4) % 23) < 22)) ? placeholder[(((((((int)blockIdx.z) * 441) + (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 441) / 23) * 21)) + ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 4) % 23)) - 22))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) < 25) {
    if (((int)threadIdx.y) < 2) {
      PaddedInput_shared[((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 504))] = ((((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) < 2) && (1 <= ((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 21) % 23))) && (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 21) % 23) < 22)) ? placeholder[(((((((int)blockIdx.z) * 441) + (((((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 504) / 23) * 21)) + (((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) + 21)) - 22))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 7) + (((int)threadIdx.x) / 3)) < 3) {
    if (((((int)threadIdx.y) * 21) + ((int)threadIdx.x)) < 9) {
      if (((int)threadIdx.y) < 1) {
        placeholder_shared[(((((int)threadIdx.y) * 21) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 21) + ((((int)blockIdx.z) % 336) * 9)) + ((int)threadIdx.x)))];
      }
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 161) + ((int)threadIdx.x)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 23))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 24))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 25))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 46))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 47))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 48))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 69))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 70))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 71))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 92))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 93))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 94))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 115))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 116))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 117))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 138))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 139))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 140))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 161))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 162))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 163))];
  PaddedInput_shared_local[(24)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 184))];
  PaddedInput_shared_local[(25)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 185))];
  PaddedInput_shared_local[(26)] = PaddedInput_shared[((((((int)threadIdx.y) * 161) + ((int)threadIdx.x)) + 186))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared_local[(24)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared_local[(25)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared_local[(26)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[((((((int)blockIdx.z) * 441) + (((int)threadIdx.y) * 147)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 441) + (((int)threadIdx.y) * 147)) + ((int)threadIdx.x)) + 21))] = DepthwiseConv2d_local[(1)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 441) + (((int)threadIdx.y) * 147)) + ((int)threadIdx.x)) + 42))] = DepthwiseConv2d_local[(2)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 441) + (((int)threadIdx.y) * 147)) + ((int)threadIdx.x)) + 63))] = DepthwiseConv2d_local[(3)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 441) + (((int)threadIdx.y) * 147)) + ((int)threadIdx.x)) + 84))] = DepthwiseConv2d_local[(4)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 441) + (((int)threadIdx.y) * 147)) + ((int)threadIdx.x)) + 105))] = DepthwiseConv2d_local[(5)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 441) + (((int)threadIdx.y) * 147)) + ((int)threadIdx.x)) + 126))] = DepthwiseConv2d_local[(6)];
}

