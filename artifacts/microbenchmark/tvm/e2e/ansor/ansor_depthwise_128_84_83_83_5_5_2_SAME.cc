//37632_1_1_252_1_1
//128_84_83_83_5_2_SAME
//dim3 grid(37632, 1, 1);
//dim3 block(252, 1, 1);

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
extern "C" __global__ void __launch_bounds__(252) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[2];
  __shared__ float PaddedInput_shared[2610];
  __shared__ float kernel_shared[50];
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  PaddedInput_shared[((((int)threadIdx.x) * 2))] = ((((2 <= (((((int)blockIdx.x) % 7) * 12) + ((((int)threadIdx.x) * 2) / 87))) && (2 <= ((((int)threadIdx.x) * 2) % 87))) && (((((int)threadIdx.x) * 2) % 87) < 85)) ? data[(((((((((int)blockIdx.x) / 7) * 13778) + ((((int)blockIdx.x) % 7) * 996)) + (((((int)threadIdx.x) * 2) / 87) * 83)) + ((((int)threadIdx.x) * 2) % 87)) - 168))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1))] = ((((2 <= (((((int)blockIdx.x) % 7) * 12) + (((((int)threadIdx.x) * 2) + 1) / 87))) && (2 <= (((((int)threadIdx.x) * 2) + 1) % 87))) && ((((((int)threadIdx.x) * 2) + 1) % 87) < 85)) ? data[(((((((((int)blockIdx.x) / 7) * 13778) + ((((int)blockIdx.x) % 7) * 996)) + ((((((int)threadIdx.x) * 2) + 1) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 1) % 87)) - 168))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 504))] = (((2 <= (((((int)threadIdx.x) * 2) + 69) % 87)) && ((((((int)threadIdx.x) * 2) + 69) % 87) < 85)) ? data[(((((((((int)blockIdx.x) / 7) * 13778) + ((((int)blockIdx.x) % 7) * 996)) + ((((((int)threadIdx.x) * 2) + 504) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 69) % 87)) - 168))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 505))] = (((2 <= (((((int)threadIdx.x) * 2) + 70) % 87)) && ((((((int)threadIdx.x) * 2) + 70) % 87) < 85)) ? data[(((((((((int)blockIdx.x) / 7) * 13778) + ((((int)blockIdx.x) % 7) * 996)) + ((((((int)threadIdx.x) * 2) + 505) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 70) % 87)) - 168))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1008))] = (((((2 <= (((((int)blockIdx.x) % 7) * 12) + ((((((int)threadIdx.x) * 2) + 1008) % 1305) / 87))) && ((((((int)blockIdx.x) % 7) * 12) + ((((((int)threadIdx.x) * 2) + 1008) % 1305) / 87)) < 85)) && (2 <= (((((int)threadIdx.x) * 2) + 51) % 87))) && ((((((int)threadIdx.x) * 2) + 51) % 87) < 85)) ? data[((((((((((int)blockIdx.x) / 7) * 13778) + ((((((int)threadIdx.x) * 2) + 1008) / 1305) * 6889)) + ((((int)blockIdx.x) % 7) * 996)) + (((((((int)threadIdx.x) * 2) + 1008) % 1305) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 51) % 87)) - 168))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1009))] = (((((2 <= (((((int)blockIdx.x) % 7) * 12) + ((((((int)threadIdx.x) * 2) + 1009) % 1305) / 87))) && ((((((int)blockIdx.x) % 7) * 12) + ((((((int)threadIdx.x) * 2) + 1009) % 1305) / 87)) < 85)) && (2 <= (((((int)threadIdx.x) * 2) + 52) % 87))) && ((((((int)threadIdx.x) * 2) + 52) % 87) < 85)) ? data[((((((((((int)blockIdx.x) / 7) * 13778) + ((((((int)threadIdx.x) * 2) + 1009) / 1305) * 6889)) + ((((int)blockIdx.x) % 7) * 996)) + (((((((int)threadIdx.x) * 2) + 1009) % 1305) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 52) % 87)) - 168))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1512))] = (((2 <= (((((int)threadIdx.x) * 2) + 33) % 87)) && ((((((int)threadIdx.x) * 2) + 33) % 87) < 85)) ? data[((((((((((int)blockIdx.x) / 7) * 13778) + ((((((int)threadIdx.x) * 2) + 1512) / 1305) * 6889)) + ((((int)blockIdx.x) % 7) * 996)) + ((((((int)threadIdx.x) * 2) + 207) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 33) % 87)) - 168))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 1513))] = (((2 <= (((((int)threadIdx.x) * 2) + 34) % 87)) && ((((((int)threadIdx.x) * 2) + 34) % 87) < 85)) ? data[((((((((((int)blockIdx.x) / 7) * 13778) + ((((((int)threadIdx.x) * 2) + 1513) / 1305) * 6889)) + ((((int)blockIdx.x) % 7) * 996)) + ((((((int)threadIdx.x) * 2) + 208) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 34) % 87)) - 168))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 2016))] = (((((((((int)blockIdx.x) % 7) * 12) + (((((int)threadIdx.x) * 2) + 711) / 87)) < 85) && (2 <= (((((int)threadIdx.x) * 2) + 15) % 87))) && ((((((int)threadIdx.x) * 2) + 15) % 87) < 85)) ? data[((((((((((int)blockIdx.x) / 7) * 13778) + ((((((int)threadIdx.x) * 2) + 2016) / 1305) * 6889)) + ((((int)blockIdx.x) % 7) * 996)) + ((((((int)threadIdx.x) * 2) + 711) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 15) % 87)) - 168))] : 0.000000e+00f);
  PaddedInput_shared[(((((int)threadIdx.x) * 2) + 2017))] = (((((((((int)blockIdx.x) % 7) * 12) + (((((int)threadIdx.x) * 2) + 712) / 87)) < 85) && (2 <= (((((int)threadIdx.x) * 2) + 16) % 87))) && ((((((int)threadIdx.x) * 2) + 16) % 87) < 85)) ? data[((((((((((int)blockIdx.x) / 7) * 13778) + ((((((int)threadIdx.x) * 2) + 2017) / 1305) * 6889)) + ((((int)blockIdx.x) % 7) * 996)) + ((((((int)threadIdx.x) * 2) + 712) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 16) % 87)) - 168))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 45) {
    PaddedInput_shared[(((((int)threadIdx.x) * 2) + 2520))] = (((((((((int)blockIdx.x) % 7) * 12) + (((((int)threadIdx.x) * 2) + 1215) / 87)) < 85) && (2 <= (((((int)threadIdx.x) * 2) + 84) % 87))) && ((((((int)threadIdx.x) * 2) + 84) % 87) < 85)) ? data[((((((((((int)blockIdx.x) / 7) * 13778) + ((((((int)threadIdx.x) * 2) + 2520) / 1305) * 6889)) + ((((int)blockIdx.x) % 7) * 996)) + ((((((int)threadIdx.x) * 2) + 1215) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 84) % 87)) - 168))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 45) {
    PaddedInput_shared[(((((int)threadIdx.x) * 2) + 2521))] = (((((((((int)blockIdx.x) % 7) * 12) + (((((int)threadIdx.x) * 2) + 1216) / 87)) < 85) && (2 <= (((((int)threadIdx.x) * 2) + 85) % 87))) && ((((((int)threadIdx.x) * 2) + 85) % 87) < 85)) ? data[((((((((((int)blockIdx.x) / 7) * 13778) + ((((((int)threadIdx.x) * 2) + 2521) / 1305) * 6889)) + ((((int)blockIdx.x) % 7) * 996)) + ((((((int)threadIdx.x) * 2) + 1216) / 87) * 83)) + (((((int)threadIdx.x) * 2) + 85) % 87)) - 168))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 50) {
    kernel_shared[(((int)threadIdx.x))] = kernel[(((((((int)blockIdx.x) % 294) / 7) * 50) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)))] * kernel_shared[(0)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1305))] * kernel_shared[(25)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1))] * kernel_shared[(1)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1306))] * kernel_shared[(26)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 2))] * kernel_shared[(2)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1307))] * kernel_shared[(27)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 3))] * kernel_shared[(3)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1308))] * kernel_shared[(28)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 4))] * kernel_shared[(4)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1309))] * kernel_shared[(29)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 87))] * kernel_shared[(5)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1392))] * kernel_shared[(30)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 88))] * kernel_shared[(6)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1393))] * kernel_shared[(31)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 89))] * kernel_shared[(7)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1394))] * kernel_shared[(32)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 90))] * kernel_shared[(8)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1395))] * kernel_shared[(33)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 91))] * kernel_shared[(9)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1396))] * kernel_shared[(34)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 174))] * kernel_shared[(10)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1479))] * kernel_shared[(35)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 175))] * kernel_shared[(11)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1480))] * kernel_shared[(36)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 176))] * kernel_shared[(12)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1481))] * kernel_shared[(37)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 177))] * kernel_shared[(13)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1482))] * kernel_shared[(38)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 178))] * kernel_shared[(14)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1483))] * kernel_shared[(39)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 261))] * kernel_shared[(15)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1566))] * kernel_shared[(40)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 262))] * kernel_shared[(16)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1567))] * kernel_shared[(41)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 263))] * kernel_shared[(17)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1568))] * kernel_shared[(42)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 264))] * kernel_shared[(18)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1569))] * kernel_shared[(43)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 265))] * kernel_shared[(19)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1570))] * kernel_shared[(44)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 348))] * kernel_shared[(20)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1653))] * kernel_shared[(45)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 349))] * kernel_shared[(21)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1654))] * kernel_shared[(46)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 350))] * kernel_shared[(22)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1655))] * kernel_shared[(47)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 351))] * kernel_shared[(23)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1656))] * kernel_shared[(48)]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 352))] * kernel_shared[(24)]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[(((((((int)threadIdx.x) / 42) * 174) + ((((int)threadIdx.x) % 42) * 2)) + 1657))] * kernel_shared[(49)]));
  for (int c_inner = 0; c_inner < 2; ++c_inner) {
    DepthwiseConv2d[((((((((int)blockIdx.x) / 7) * 3528) + (c_inner * 1764)) + ((((int)blockIdx.x) % 7) * 252)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(c_inner)];
  }
}

