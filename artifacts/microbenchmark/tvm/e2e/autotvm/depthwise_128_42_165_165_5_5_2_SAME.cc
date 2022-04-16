//1_83_5376_83_1_1
//128_42_165_165_5_2_SAME
//dim3 grid(1, 83, 5376);
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
  __shared__ float PaddedInput_shared[845];
  __shared__ float placeholder_shared[25];
  float PaddedInput_shared_local[25];
  float placeholder_shared_local[25];
  float DepthwiseConv2d_local[1];
  PaddedInput_shared[(((int)threadIdx.x))] = (((1 <= ((int)blockIdx.y)) && (2 <= ((int)threadIdx.x))) ? placeholder[(((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + ((int)threadIdx.x)) - 332))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 83))] = ((1 <= ((int)blockIdx.y)) ? placeholder[(((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + ((int)threadIdx.x)) - 249))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 166))] = ((((2 <= ((((int)blockIdx.y) * 2) + ((((int)threadIdx.x) + 166) / 169))) && (2 <= ((((int)threadIdx.x) + 166) % 169))) && (((((int)threadIdx.x) + 166) % 169) < 167)) ? placeholder[((((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + (((((int)threadIdx.x) + 166) / 169) * 165)) + ((((int)threadIdx.x) + 166) % 169)) - 332))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 249))] = ((2 <= ((((int)blockIdx.y) * 2) + ((((int)threadIdx.x) + 249) / 169))) ? placeholder[((((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + (((((int)threadIdx.x) + 249) / 169) * 165)) + (((int)threadIdx.x) + 80)) - 332))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 332))] = ((((2 <= ((((int)blockIdx.y) * 2) + ((((int)threadIdx.x) + 332) / 169))) && (2 <= ((((int)threadIdx.x) + 163) % 169))) && (((((int)threadIdx.x) + 163) % 169) < 167)) ? placeholder[((((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + (((((int)threadIdx.x) + 332) / 169) * 165)) + ((((int)threadIdx.x) + 163) % 169)) - 332))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 415))] = placeholder[((((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + (((((int)threadIdx.x) + 415) / 169) * 165)) + (((int)threadIdx.x) + 77)) - 332))];
  PaddedInput_shared[((((int)threadIdx.x) + 498))] = ((((((((int)blockIdx.y) * 2) + ((((int)threadIdx.x) + 498) / 169)) < 167) && (2 <= ((((int)threadIdx.x) + 160) % 169))) && (((((int)threadIdx.x) + 160) % 169) < 167)) ? placeholder[((((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + (((((int)threadIdx.x) + 498) / 169) * 165)) + ((((int)threadIdx.x) + 160) % 169)) - 332))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 581))] = ((((((int)blockIdx.y) * 2) + ((((int)threadIdx.x) + 581) / 169)) < 167) ? placeholder[((((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + (((((int)threadIdx.x) + 581) / 169) * 165)) + (((int)threadIdx.x) + 74)) - 332))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 664))] = ((((((((int)blockIdx.y) * 2) + ((((int)threadIdx.x) + 664) / 169)) < 167) && (2 <= ((((int)threadIdx.x) + 157) % 169))) && (((((int)threadIdx.x) + 157) % 169) < 167)) ? placeholder[((((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + (((((int)threadIdx.x) + 664) / 169) * 165)) + ((((int)threadIdx.x) + 157) % 169)) - 332))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 747))] = ((((((int)blockIdx.y) * 2) + ((((int)threadIdx.x) + 747) / 169)) < 167) ? placeholder[((((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + (((((int)threadIdx.x) + 747) / 169) * 165)) + (((int)threadIdx.x) + 71)) - 332))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 15) {
    PaddedInput_shared[((((int)threadIdx.x) + 830))] = (((((((int)blockIdx.y) * 2) + ((((int)threadIdx.x) + 830) / 169)) < 167) && (((int)threadIdx.x) < 13)) ? placeholder[((((((((int)blockIdx.z) * 27225) + (((int)blockIdx.y) * 330)) + (((((int)threadIdx.x) + 830) / 169) * 165)) + (((int)threadIdx.x) + 154)) - 332))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 25) {
    placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((((int)blockIdx.z) % 42) * 25) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[((((int)threadIdx.x) * 2))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 3))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 4))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 169))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 170))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 171))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 172))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 173))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 338))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 339))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 340))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 341))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 342))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 507))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 508))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 509))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 510))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 511))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 676))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 677))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 678))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 679))];
  PaddedInput_shared_local[(24)] = PaddedInput_shared[(((((int)threadIdx.x) * 2) + 680))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  placeholder_shared_local[(9)] = placeholder_shared[(9)];
  placeholder_shared_local[(10)] = placeholder_shared[(10)];
  placeholder_shared_local[(11)] = placeholder_shared[(11)];
  placeholder_shared_local[(12)] = placeholder_shared[(12)];
  placeholder_shared_local[(13)] = placeholder_shared[(13)];
  placeholder_shared_local[(14)] = placeholder_shared[(14)];
  placeholder_shared_local[(15)] = placeholder_shared[(15)];
  placeholder_shared_local[(16)] = placeholder_shared[(16)];
  placeholder_shared_local[(17)] = placeholder_shared[(17)];
  placeholder_shared_local[(18)] = placeholder_shared[(18)];
  placeholder_shared_local[(19)] = placeholder_shared[(19)];
  placeholder_shared_local[(20)] = placeholder_shared[(20)];
  placeholder_shared_local[(21)] = placeholder_shared[(21)];
  placeholder_shared_local[(22)] = placeholder_shared[(22)];
  placeholder_shared_local[(23)] = placeholder_shared[(23)];
  placeholder_shared_local[(24)] = placeholder_shared[(24)];
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
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(9)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(10)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(11)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(12)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(13)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(14)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(15)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(16)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(17)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(18)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(19)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(20)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(21)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(22)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(23)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared_local[(24)] * placeholder_shared_local[(24)]));
  DepthwiseConv2d[((((((int)blockIdx.z) * 6889) + (((int)blockIdx.y) * 83)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
}

