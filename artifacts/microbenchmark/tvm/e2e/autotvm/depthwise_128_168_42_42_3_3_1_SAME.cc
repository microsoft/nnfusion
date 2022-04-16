//1_2_21504_42_3_1
//128_168_42_42_3_1_SAME
//dim3 grid(1, 2, 21504);
//dim3 block(42, 3, 1);

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
extern "C" __global__ void __launch_bounds__(126) depthwise_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ DepthwiseConv2d) {
  __shared__ float PaddedInput_shared[1012];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[27];
  float placeholder_shared_local[9];
  float DepthwiseConv2d_local[7];
  PaddedInput_shared[(((((int)threadIdx.y) * 42) + ((int)threadIdx.x)))] = ((((1 <= ((((int)blockIdx.y) * 21) + (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) / 44))) && (1 <= (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) % 44))) && ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) / 44) * 42)) + (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 126))] = (((1 <= ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 38) % 44)) && (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 38) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 126) / 44) * 42)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 38) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 252))] = (((1 <= ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 32) % 44)) && (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 32) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 252) / 44) * 42)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 32) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 378))] = (((1 <= ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 26) % 44)) && (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 26) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 378) / 44) * 42)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 26) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 504))] = (((1 <= ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 20) % 44)) && (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 20) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 504) / 44) * 42)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 20) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 630))] = (((1 <= ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 14) % 44)) && (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 14) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 630) / 44) * 42)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 14) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 756))] = (((1 <= ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 8) % 44)) && (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 8) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 756) / 44) * 42)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 8) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 882))] = ((((((((int)blockIdx.y) * 21) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 882) / 44)) < 43) && (1 <= ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 2) % 44))) && (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 2) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 882) / 44) * 42)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 2) % 44)) - 43))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) < 4) {
    if (((int)threadIdx.y) < 1) {
      PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 1008))] = (((((((int)blockIdx.y) * 21) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 1008) / 44)) < 43) && (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) < 3)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 1008) / 44) * 42)) + (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 40)) - 43))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) / 3)) < 3) {
    if (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) < 9) {
      if (((int)threadIdx.y) < 1) {
        placeholder_shared[(((((int)threadIdx.y) * 42) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 42) + ((((int)blockIdx.z) % 168) * 9)) + ((int)threadIdx.x)))];
      }
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 308) + ((int)threadIdx.x)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 44))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 45))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 46))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 88))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 89))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 90))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 132))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 133))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 134))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 176))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 177))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 178))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 220))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 221))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 222))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 264))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 265))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 266))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 308))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 309))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 310))];
  PaddedInput_shared_local[(24)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 352))];
  PaddedInput_shared_local[(25)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 353))];
  PaddedInput_shared_local[(26)] = PaddedInput_shared[((((((int)threadIdx.y) * 308) + ((int)threadIdx.x)) + 354))];
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
  DepthwiseConv2d[(((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((int)threadIdx.y) * 294)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
  DepthwiseConv2d[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((int)threadIdx.y) * 294)) + ((int)threadIdx.x)) + 42))] = DepthwiseConv2d_local[(1)];
  DepthwiseConv2d[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((int)threadIdx.y) * 294)) + ((int)threadIdx.x)) + 84))] = DepthwiseConv2d_local[(2)];
  DepthwiseConv2d[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((int)threadIdx.y) * 294)) + ((int)threadIdx.x)) + 126))] = DepthwiseConv2d_local[(3)];
  DepthwiseConv2d[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((int)threadIdx.y) * 294)) + ((int)threadIdx.x)) + 168))] = DepthwiseConv2d_local[(4)];
  DepthwiseConv2d[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((int)threadIdx.y) * 294)) + ((int)threadIdx.x)) + 210))] = DepthwiseConv2d_local[(5)];
  DepthwiseConv2d[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 882)) + (((int)threadIdx.y) * 294)) + ((int)threadIdx.x)) + 252))] = DepthwiseConv2d_local[(6)];
}

