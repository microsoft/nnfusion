//1_1_43008_11_11_1
//128_672_21_21_5_2_SAME
//dim3 grid(1, 1, 43008);
//dim3 block(11, 11, 1);

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
extern "C" __global__ void __launch_bounds__(121) depthwise_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ DepthwiseConv2d) {
  __shared__ float PaddedInput_shared[1250];
  __shared__ float placeholder_shared[50];
  float PaddedInput_shared_local[50];
  float placeholder_shared_local[50];
  float DepthwiseConv2d_local[2];
  PaddedInput_shared[(((((int)threadIdx.y) * 11) + ((int)threadIdx.x)))] = ((((50 <= ((((int)threadIdx.y) * 11) + ((int)threadIdx.x))) && (2 <= (((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) % 25))) && ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) % 25) < 23)) ? placeholder[(((((((int)blockIdx.z) * 882) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) / 25) * 21)) + (((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 121))] = (((2 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 21) % 25)) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 21) % 25) < 23)) ? placeholder[(((((((int)blockIdx.z) * 882) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 121) / 25) * 21)) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 21) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 242))] = (((2 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 17) % 25)) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 17) % 25) < 23)) ? placeholder[(((((((int)blockIdx.z) * 882) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 242) / 25) * 21)) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 17) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 363))] = (((2 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 13) % 25)) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 13) % 25) < 23)) ? placeholder[(((((((int)blockIdx.z) * 882) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 363) / 25) * 21)) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 13) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 484))] = ((((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) < 91) && (2 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 9) % 25))) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 9) % 25) < 23)) ? placeholder[(((((((int)blockIdx.z) * 882) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 484) / 25) * 21)) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 9) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 605))] = (((((50 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 605) % 625)) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 605) % 625) < 575)) && (2 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 5) % 25))) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 5) % 25) < 23)) ? placeholder[((((((((int)blockIdx.z) * 882) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 605) / 625) * 441)) + ((((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 605) % 625) / 25) * 21)) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 5) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 726))] = (((2 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 1) % 25)) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 1) % 25) < 23)) ? placeholder[((((((((int)blockIdx.z) * 882) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 726) / 625) * 441)) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 101) / 25) * 21)) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 1) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 847))] = (((2 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 22) % 25)) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 22) % 25) < 23)) ? placeholder[((((((((int)blockIdx.z) * 882) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 847) / 625) * 441)) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 222) / 25) * 21)) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 22) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 968))] = (((2 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 18) % 25)) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 18) % 25) < 23)) ? placeholder[((((((((int)blockIdx.z) * 882) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 968) / 625) * 441)) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 343) / 25) * 21)) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 18) % 25)) - 44))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 1089))] = ((((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) < 111) && (2 <= ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 14) % 25))) && (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 14) % 25) < 23)) ? placeholder[((((((((int)blockIdx.z) * 882) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 1089) / 625) * 441)) + (((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 464) / 25) * 21)) + ((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 14) % 25)) - 44))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) < 40) {
    if (((int)threadIdx.y) < 4) {
      PaddedInput_shared[((((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) + 1210))] = 0.000000e+00f;
    }
  }
  if (((((int)threadIdx.y) * 11) + ((int)threadIdx.x)) < 50) {
    if (((int)threadIdx.y) < 5) {
      placeholder_shared[(((((int)threadIdx.y) * 11) + ((int)threadIdx.x)))] = placeholder1[(((((((int)blockIdx.z) % 336) * 50) + (((int)threadIdx.y) * 11)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 3))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 4))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 25))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 26))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 27))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 28))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 29))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 50))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 51))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 52))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 53))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 54))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 75))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 76))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 77))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 78))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 79))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 100))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 101))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 102))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 103))];
  PaddedInput_shared_local[(24)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 104))];
  PaddedInput_shared_local[(25)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 625))];
  PaddedInput_shared_local[(26)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 626))];
  PaddedInput_shared_local[(27)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 627))];
  PaddedInput_shared_local[(28)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 628))];
  PaddedInput_shared_local[(29)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 629))];
  PaddedInput_shared_local[(30)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 650))];
  PaddedInput_shared_local[(31)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 651))];
  PaddedInput_shared_local[(32)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 652))];
  PaddedInput_shared_local[(33)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 653))];
  PaddedInput_shared_local[(34)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 654))];
  PaddedInput_shared_local[(35)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 675))];
  PaddedInput_shared_local[(36)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 676))];
  PaddedInput_shared_local[(37)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 677))];
  PaddedInput_shared_local[(38)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 678))];
  PaddedInput_shared_local[(39)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 679))];
  PaddedInput_shared_local[(40)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 700))];
  PaddedInput_shared_local[(41)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 701))];
  PaddedInput_shared_local[(42)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 702))];
  PaddedInput_shared_local[(43)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 703))];
  PaddedInput_shared_local[(44)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 704))];
  PaddedInput_shared_local[(45)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 725))];
  PaddedInput_shared_local[(46)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 726))];
  PaddedInput_shared_local[(47)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 727))];
  PaddedInput_shared_local[(48)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 728))];
  PaddedInput_shared_local[(49)] = PaddedInput_shared[((((((int)threadIdx.y) * 50) + (((int)threadIdx.x) * 2)) + 729))];
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
  placeholder_shared_local[(25)] = placeholder_shared[(25)];
  placeholder_shared_local[(26)] = placeholder_shared[(26)];
  placeholder_shared_local[(27)] = placeholder_shared[(27)];
  placeholder_shared_local[(28)] = placeholder_shared[(28)];
  placeholder_shared_local[(29)] = placeholder_shared[(29)];
  placeholder_shared_local[(30)] = placeholder_shared[(30)];
  placeholder_shared_local[(31)] = placeholder_shared[(31)];
  placeholder_shared_local[(32)] = placeholder_shared[(32)];
  placeholder_shared_local[(33)] = placeholder_shared[(33)];
  placeholder_shared_local[(34)] = placeholder_shared[(34)];
  placeholder_shared_local[(35)] = placeholder_shared[(35)];
  placeholder_shared_local[(36)] = placeholder_shared[(36)];
  placeholder_shared_local[(37)] = placeholder_shared[(37)];
  placeholder_shared_local[(38)] = placeholder_shared[(38)];
  placeholder_shared_local[(39)] = placeholder_shared[(39)];
  placeholder_shared_local[(40)] = placeholder_shared[(40)];
  placeholder_shared_local[(41)] = placeholder_shared[(41)];
  placeholder_shared_local[(42)] = placeholder_shared[(42)];
  placeholder_shared_local[(43)] = placeholder_shared[(43)];
  placeholder_shared_local[(44)] = placeholder_shared[(44)];
  placeholder_shared_local[(45)] = placeholder_shared[(45)];
  placeholder_shared_local[(46)] = placeholder_shared[(46)];
  placeholder_shared_local[(47)] = placeholder_shared[(47)];
  placeholder_shared_local[(48)] = placeholder_shared[(48)];
  placeholder_shared_local[(49)] = placeholder_shared[(49)];
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
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(25)] * placeholder_shared_local[(25)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(26)] * placeholder_shared_local[(26)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(27)] * placeholder_shared_local[(27)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(28)] * placeholder_shared_local[(28)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(29)] * placeholder_shared_local[(29)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(30)] * placeholder_shared_local[(30)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(31)] * placeholder_shared_local[(31)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(32)] * placeholder_shared_local[(32)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(33)] * placeholder_shared_local[(33)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(34)] * placeholder_shared_local[(34)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(35)] * placeholder_shared_local[(35)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(36)] * placeholder_shared_local[(36)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(37)] * placeholder_shared_local[(37)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(38)] * placeholder_shared_local[(38)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(39)] * placeholder_shared_local[(39)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(40)] * placeholder_shared_local[(40)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(41)] * placeholder_shared_local[(41)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(42)] * placeholder_shared_local[(42)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(43)] * placeholder_shared_local[(43)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(44)] * placeholder_shared_local[(44)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(45)] * placeholder_shared_local[(45)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(46)] * placeholder_shared_local[(46)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(47)] * placeholder_shared_local[(47)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(48)] * placeholder_shared_local[(48)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared_local[(49)] * placeholder_shared_local[(49)]));
  DepthwiseConv2d[((((((int)blockIdx.z) * 242) + (((int)threadIdx.y) * 11)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
  DepthwiseConv2d[(((((((int)blockIdx.z) * 242) + (((int)threadIdx.y) * 11)) + ((int)threadIdx.x)) + 121))] = DepthwiseConv2d_local[(1)];
}

