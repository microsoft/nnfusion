//1_7_10752_42_3_1
//128_84_42_42_3_1_SAME
//dim3 grid(1, 7, 10752);
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
  __shared__ float PaddedInput_shared[352];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[12];
  float placeholder_shared_local[9];
  float DepthwiseConv2d_local[2];
  PaddedInput_shared[(((((int)threadIdx.y) * 42) + ((int)threadIdx.x)))] = ((((1 <= ((((int)blockIdx.y) * 6) + (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) / 44))) && (1 <= (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) % 44))) && ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 252)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) / 44) * 42)) + (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) % 44)) - 43))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 126))] = (((1 <= ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 38) % 44)) && (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 38) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 252)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 126) / 44) * 42)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 38) % 44)) - 43))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) < 100) {
    PaddedInput_shared[((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 252))] = ((((((((int)blockIdx.y) * 6) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 252) / 44)) < 43) && (1 <= ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 32) % 44))) && (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 32) % 44) < 43)) ? placeholder[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 252)) + (((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 252) / 44) * 42)) + ((((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) + 32) % 44)) - 43))] : 0.000000e+00f);
  }
  if (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) / 3)) < 3) {
    if (((((int)threadIdx.y) * 42) + ((int)threadIdx.x)) < 9) {
      if (((int)threadIdx.y) < 1) {
        placeholder_shared[(((((int)threadIdx.y) * 42) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 42) + ((((int)blockIdx.z) % 84) * 9)) + ((int)threadIdx.x)))];
      }
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 88) + ((int)threadIdx.x)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 44))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 45))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 46))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 88))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 89))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 90))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 132))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 133))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 88) + ((int)threadIdx.x)) + 134))];
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
  DepthwiseConv2d[(((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 252)) + (((int)threadIdx.y) * 84)) + ((int)threadIdx.x)))] = DepthwiseConv2d_local[(0)];
  DepthwiseConv2d[((((((((int)blockIdx.z) * 1764) + (((int)blockIdx.y) * 252)) + (((int)threadIdx.y) * 84)) + ((int)threadIdx.x)) + 42))] = DepthwiseConv2d_local[(1)];
}

