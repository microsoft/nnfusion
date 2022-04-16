//3584_1_1_448_1_1
//128_256_14_14_1024_1_1_SAME
//dim3 grid(3584, 1, 1);
//dim3 block(448, 1, 1);

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
extern "C" __global__ void __launch_bounds__(448) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ T_add, float* __restrict__ input2) {
  float compute[16];
  __shared__ float pad_temp_shared[1792];
  __shared__ float input1_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[(((int)threadIdx.x))] = input0[((((((((((((int)blockIdx.x) / 224) * 401408) + ((((int)threadIdx.x) / 224) * 50176)) + (rc_outer_outer * 3136)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)))];
    pad_temp_shared[((((int)threadIdx.x) + 448))] = input0[(((((((((((((int)blockIdx.x) / 224) * 401408) + ((((int)threadIdx.x) / 224) * 50176)) + (rc_outer_outer * 3136)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 100352))];
    pad_temp_shared[((((int)threadIdx.x) + 896))] = input0[(((((((((((((int)blockIdx.x) / 224) * 401408) + ((((int)threadIdx.x) / 224) * 50176)) + (rc_outer_outer * 3136)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 200704))];
    pad_temp_shared[((((int)threadIdx.x) + 1344))] = input0[(((((((((((((int)blockIdx.x) / 224) * 401408) + ((((int)threadIdx.x) / 224) * 50176)) + (rc_outer_outer * 3136)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 301056))];
    input1_shared[(((int)threadIdx.x))] = input1[(((((((((int)blockIdx.x) % 224) / 14) * 16384) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)))];
    input1_shared[((((int)threadIdx.x) + 448))] = input1[((((((((((int)blockIdx.x) % 224) / 14) * 16384) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 7168))];
    if (((int)threadIdx.x) < 128) {
      input1_shared[((((int)threadIdx.x) + 896))] = input1[((((((((((int)blockIdx.x) % 224) / 14) * 16384) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)))] * input1_shared[(((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 256))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 512))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 768))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 224))] * input1_shared[(((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 224))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 256))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 224))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 512))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 224))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 768))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 14))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 1))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 14))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 257))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 14))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 513))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 14))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 769))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 238))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 1))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 238))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 257))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 238))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 513))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 238))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 769))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 448))] * input1_shared[(((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 448))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 256))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 448))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 512))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 448))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 768))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 672))] * input1_shared[(((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 672))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 256))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 672))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 512))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 672))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 768))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 462))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 1))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 462))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 257))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 462))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 513))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 462))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 769))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 686))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 1))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 686))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 257))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 686))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 513))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((((((int)threadIdx.x) / 224) * 896) + (rc_outer_inner * 28)) + (((int)threadIdx.x) % 14)) + 686))] * input1_shared[((((((((int)threadIdx.x) % 224) / 14) * 16) + (rc_outer_inner * 2)) + 769))]));
    }
  }
  for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
    T_add[(((((((((((((int)blockIdx.x) / 224) * 1605632) + ((((int)threadIdx.x) / 224) * 802816)) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 224) / 14) * 12544)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)))] = (compute[(ax0_inner)] + input2[(((((((((((((int)blockIdx.x) / 224) * 1605632) + ((((int)threadIdx.x) / 224) * 802816)) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 224) / 14) * 12544)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)))]);
    T_add[((((((((((((((int)blockIdx.x) / 224) * 1605632) + ((((int)threadIdx.x) / 224) * 802816)) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 224) / 14) * 12544)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 3136))] = (compute[((ax0_inner + 4))] + input2[((((((((((((((int)blockIdx.x) / 224) * 1605632) + ((((int)threadIdx.x) / 224) * 802816)) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 224) / 14) * 12544)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 3136))]);
    T_add[((((((((((((((int)blockIdx.x) / 224) * 1605632) + ((((int)threadIdx.x) / 224) * 802816)) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 224) / 14) * 12544)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 6272))] = (compute[((ax0_inner + 8))] + input2[((((((((((((((int)blockIdx.x) / 224) * 1605632) + ((((int)threadIdx.x) / 224) * 802816)) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 224) / 14) * 12544)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 6272))]);
    T_add[((((((((((((((int)blockIdx.x) / 224) * 1605632) + ((((int)threadIdx.x) / 224) * 802816)) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 224) / 14) * 12544)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 9408))] = (compute[((ax0_inner + 12))] + input2[((((((((((((((int)blockIdx.x) / 224) * 1605632) + ((((int)threadIdx.x) / 224) * 802816)) + (ax0_inner * 200704)) + (((((int)blockIdx.x) % 224) / 14) * 12544)) + (((((int)threadIdx.x) % 224) / 14) * 196)) + (((((int)blockIdx.x) % 14) / 7) * 98)) + (((((int)threadIdx.x) % 14) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 9408))]);
  }
}

