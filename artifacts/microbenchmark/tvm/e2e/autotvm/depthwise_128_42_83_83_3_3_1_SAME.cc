//1_83_1792_83_1_1
//128_42_83_83_3_1_SAME
//dim3 grid(1, 83, 1792);
//dim3 block(83, 1, 1);

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
extern "C" __global__ void __launch_bounds__(83) depthwise_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ DepthwiseConv2d) {
  __shared__ float PaddedInput_shared[765];
  __shared__ float placeholder_shared[27];
  float PaddedInput_shared_local[27];
  float placeholder_shared_local[27];
  float DepthwiseConv2d_local[3];
  PaddedInput_shared[(((int)threadIdx.x))] = (((1 <= ((int)blockIdx.y)) && (1 <= ((int)threadIdx.x))) ? placeholder[(((((((int)blockIdx.z) * 20667) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) - 84))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 83))] = ((((1 <= (((((int)threadIdx.x) + 83) / 85) + ((int)blockIdx.y))) && (1 <= ((((int)threadIdx.x) + 83) % 85))) && (((((int)threadIdx.x) + 83) % 85) < 84)) ? placeholder[((((((((int)blockIdx.z) * 20667) + (((((int)threadIdx.x) + 83) / 85) * 83)) + (((int)blockIdx.y) * 83)) + ((((int)threadIdx.x) + 83) % 85)) - 84))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 166))] = (((((((((int)threadIdx.x) + 166) / 85) + ((int)blockIdx.y)) < 84) && (1 <= ((((int)threadIdx.x) + 81) % 85))) && (((((int)threadIdx.x) + 81) % 85) < 84)) ? placeholder[((((((((int)blockIdx.z) * 20667) + (((((int)threadIdx.x) + 166) / 85) * 83)) + (((int)blockIdx.y) * 83)) + ((((int)threadIdx.x) + 81) % 85)) - 84))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 249))] = (((((1 <= ((((((int)threadIdx.x) + 249) % 255) / 85) + ((int)blockIdx.y))) && (((((((int)threadIdx.x) + 249) % 255) / 85) + ((int)blockIdx.y)) < 84)) && (1 <= ((((int)threadIdx.x) + 79) % 85))) && (((((int)threadIdx.x) + 79) % 85) < 84)) ? placeholder[(((((((((int)blockIdx.z) * 20667) + (((((int)threadIdx.x) + 249) / 255) * 6889)) + ((((((int)threadIdx.x) + 249) % 255) / 85) * 83)) + (((int)blockIdx.y) * 83)) + ((((int)threadIdx.x) + 79) % 85)) - 84))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 332))] = ((((1 <= (((((int)threadIdx.x) + 77) / 85) + ((int)blockIdx.y))) && (1 <= ((((int)threadIdx.x) + 77) % 85))) && (((((int)threadIdx.x) + 77) % 85) < 84)) ? placeholder[(((((((((int)blockIdx.z) * 20667) + (((((int)threadIdx.x) + 332) / 255) * 6889)) + (((((int)threadIdx.x) + 77) / 85) * 83)) + (((int)blockIdx.y) * 83)) + ((((int)threadIdx.x) + 77) % 85)) - 84))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 415))] = (((((((((int)threadIdx.x) + 160) / 85) + ((int)blockIdx.y)) < 84) && (1 <= ((((int)threadIdx.x) + 75) % 85))) && (((((int)threadIdx.x) + 75) % 85) < 84)) ? placeholder[(((((((((int)blockIdx.z) * 20667) + (((((int)threadIdx.x) + 415) / 255) * 6889)) + (((((int)threadIdx.x) + 160) / 85) * 83)) + (((int)blockIdx.y) * 83)) + ((((int)threadIdx.x) + 75) % 85)) - 84))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 498))] = (((((1 <= ((((((int)threadIdx.x) + 243) % 255) / 85) + ((int)blockIdx.y))) && (((((((int)threadIdx.x) + 243) % 255) / 85) + ((int)blockIdx.y)) < 84)) && (1 <= ((((int)threadIdx.x) + 73) % 85))) && (((((int)threadIdx.x) + 73) % 85) < 84)) ? placeholder[(((((((((int)blockIdx.z) * 20667) + (((((int)threadIdx.x) + 498) / 255) * 6889)) + ((((((int)threadIdx.x) + 243) % 255) / 85) * 83)) + (((int)blockIdx.y) * 83)) + ((((int)threadIdx.x) + 73) % 85)) - 84))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 581))] = ((((1 <= (((((int)threadIdx.x) + 71) / 85) + ((int)blockIdx.y))) && (1 <= ((((int)threadIdx.x) + 71) % 85))) && (((((int)threadIdx.x) + 71) % 85) < 84)) ? placeholder[(((((((((int)blockIdx.z) * 20667) + (((((int)threadIdx.x) + 581) / 255) * 6889)) + (((((int)threadIdx.x) + 71) / 85) * 83)) + (((int)blockIdx.y) * 83)) + ((((int)threadIdx.x) + 71) % 85)) - 84))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 664))] = (((((((((int)threadIdx.x) + 154) / 85) + ((int)blockIdx.y)) < 84) && (1 <= ((((int)threadIdx.x) + 69) % 85))) && (((((int)threadIdx.x) + 69) % 85) < 84)) ? placeholder[(((((((((int)blockIdx.z) * 20667) + (((((int)threadIdx.x) + 664) / 255) * 6889)) + (((((int)threadIdx.x) + 154) / 85) * 83)) + (((int)blockIdx.y) * 83)) + ((((int)threadIdx.x) + 69) % 85)) - 84))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 18) {
    PaddedInput_shared[((((int)threadIdx.x) + 747))] = ((((((((int)threadIdx.x) + 237) / 85) + ((int)blockIdx.y)) < 84) && (((int)threadIdx.x) < 17)) ? placeholder[(((((((((int)blockIdx.z) * 20667) + (((((int)threadIdx.x) + 747) / 255) * 6889)) + (((((int)threadIdx.x) + 237) / 85) * 83)) + (((int)blockIdx.y) * 83)) + (((int)threadIdx.x) + 67)) - 84))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 27) {
    placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((((int)blockIdx.z) % 14) * 27) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((int)threadIdx.x))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((int)threadIdx.x) + 255))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[((((int)threadIdx.x) + 510))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((int)threadIdx.x) + 1))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((int)threadIdx.x) + 256))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[((((int)threadIdx.x) + 511))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((int)threadIdx.x) + 2))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((int)threadIdx.x) + 257))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[((((int)threadIdx.x) + 512))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((int)threadIdx.x) + 85))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((int)threadIdx.x) + 340))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[((((int)threadIdx.x) + 595))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((int)threadIdx.x) + 86))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((int)threadIdx.x) + 341))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[((((int)threadIdx.x) + 596))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((int)threadIdx.x) + 87))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((int)threadIdx.x) + 342))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[((((int)threadIdx.x) + 597))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((int)threadIdx.x) + 170))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((int)threadIdx.x) + 425))];
  PaddedInput_shared_local[(24)] = PaddedInput_shared[((((int)threadIdx.x) + 680))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((int)threadIdx.x) + 171))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((int)threadIdx.x) + 426))];
  PaddedInput_shared_local[(25)] = PaddedInput_shared[((((int)threadIdx.x) + 681))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((int)threadIdx.x) + 172))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((int)threadIdx.x) + 427))];
  PaddedInput_shared_local[(26)] = PaddedInput_shared[((((int)threadIdx.x) + 682))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(9)] = placeholder_shared[(9)];
  placeholder_shared_local[(18)] = placeholder_shared[(18)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(10)] = placeholder_shared[(10)];
  placeholder_shared_local[(19)] = placeholder_shared[(19)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(11)] = placeholder_shared[(11)];
  placeholder_shared_local[(20)] = placeholder_shared[(20)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(12)] = placeholder_shared[(12)];
  placeholder_shared_local[(21)] = placeholder_shared[(21)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(13)] = placeholder_shared[(13)];
  placeholder_shared_local[(22)] = placeholder_shared[(22)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(14)] = placeholder_shared[(14)];
  placeholder_shared_local[(23)] = placeholder_shared[(23)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(15)] = placeholder_shared[(15)];
  placeholder_shared_local[(24)] = placeholder_shared[(24)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(16)] = placeholder_shared[(16)];
  placeholder_shared_local[(25)] = placeholder_shared[(25)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  placeholder_shared_local[(17)] = placeholder_shared[(17)];
  placeholder_shared_local[(26)] = placeholder_shared[(26)];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(9)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(18)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(10)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(19)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(11)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(20)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(12)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(21)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(13)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(22)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(14)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(23)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(15)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(24)] * placeholder_shared_local[(24)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(16)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(25)] * placeholder_shared_local[(25)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(17)]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared_local[(26)] * placeholder_shared_local[(26)]));
  DepthwiseConv2d[((((((int)blockIdx.z) * 20667) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 20667) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 6889))] = DepthwiseConv2d_local[(1)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 20667) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)) + 13778))] = DepthwiseConv2d_local[(2)];
}

