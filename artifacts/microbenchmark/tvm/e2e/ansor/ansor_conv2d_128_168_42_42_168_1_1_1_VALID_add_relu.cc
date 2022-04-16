//4704_1_1_84_1_1
//128_168_42_42_168_1_1_VALID
//dim3 grid(4704, 1, 1);
//dim3 block(84, 1, 1);

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
extern "C" __global__ void __launch_bounds__(84) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[96];
  __shared__ float pad_temp_shared[1344];
  __shared__ float input1_shared[96];
  for (int nn_inner_init = 0; nn_inner_init < 4; ++nn_inner_init) {
    for (int ff_inner_init = 0; ff_inner_init < 2; ++ff_inner_init) {
      compute1[(((nn_inner_init * 8) + (ff_inner_init * 4)))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 32))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 64))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 1))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 33))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 65))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 2))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 34))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 66))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 3))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 35))] = 0.000000e+00f;
      compute1[((((nn_inner_init * 8) + (ff_inner_init * 4)) + 67))] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 42; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[(((int)threadIdx.x))] = input0[((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)))];
    pad_temp_shared[((((int)threadIdx.x) + 84))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 1764))];
    pad_temp_shared[((((int)threadIdx.x) + 168))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 3528))];
    pad_temp_shared[((((int)threadIdx.x) + 252))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 5292))];
    pad_temp_shared[((((int)threadIdx.x) + 336))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 296352))];
    pad_temp_shared[((((int)threadIdx.x) + 420))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 298116))];
    pad_temp_shared[((((int)threadIdx.x) + 504))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 299880))];
    pad_temp_shared[((((int)threadIdx.x) + 588))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 301644))];
    pad_temp_shared[((((int)threadIdx.x) + 672))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 592704))];
    pad_temp_shared[((((int)threadIdx.x) + 756))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 594468))];
    pad_temp_shared[((((int)threadIdx.x) + 840))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 596232))];
    pad_temp_shared[((((int)threadIdx.x) + 924))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 597996))];
    pad_temp_shared[((((int)threadIdx.x) + 1008))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 889056))];
    pad_temp_shared[((((int)threadIdx.x) + 1092))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 890820))];
    pad_temp_shared[((((int)threadIdx.x) + 1176))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 892584))];
    pad_temp_shared[((((int)threadIdx.x) + 1260))] = input0[(((((((((int)blockIdx.x) / 147) * 1185408) + (rc_outer_outer * 7056)) + ((((int)blockIdx.x) % 21) * 84)) + ((int)threadIdx.x)) + 894348))];
    input1_shared[(((int)threadIdx.x))] = input1[(((((((((int)blockIdx.x) % 147) / 21) * 4032) + ((((int)threadIdx.x) >> 2) * 168)) + (rc_outer_outer * 4)) + (((int)threadIdx.x) & 3)))];
    if (((int)threadIdx.x) < 12) {
      input1_shared[((((int)threadIdx.x) + 84))] = input1[((((((((((int)blockIdx.x) % 147) / 21) * 4032) + ((((int)threadIdx.x) >> 2) * 168)) + (rc_outer_outer * 4)) + (((int)threadIdx.x) & 3)) + 3528))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int nn_inner = 0; nn_inner < 4; ++nn_inner) {
        for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
          compute1[(((nn_inner * 8) + (ff_inner * 4)))] = (compute1[(((nn_inner * 8) + (ff_inner * 4)))] + (pad_temp_shared[((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)))] * input1_shared[(((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 32))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 32))] + (pad_temp_shared[((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)))] * input1_shared[((((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner) + 32))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 64))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 64))] + (pad_temp_shared[((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)))] * input1_shared[((((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner) + 64))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 1))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 1))] + (pad_temp_shared[(((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)) + 1))] * input1_shared[(((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 33))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 33))] + (pad_temp_shared[(((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)) + 1))] * input1_shared[((((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner) + 32))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 65))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 65))] + (pad_temp_shared[(((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)) + 1))] * input1_shared[((((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner) + 64))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 2))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 2))] + (pad_temp_shared[(((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)) + 42))] * input1_shared[(((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 34))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 34))] + (pad_temp_shared[(((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)) + 42))] * input1_shared[((((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner) + 32))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 66))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 66))] + (pad_temp_shared[(((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)) + 42))] * input1_shared[((((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner) + 64))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 3))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 3))] + (pad_temp_shared[(((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)) + 43))] * input1_shared[(((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 35))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 35))] + (pad_temp_shared[(((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)) + 43))] * input1_shared[((((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner) + 32))]));
          compute1[((((nn_inner * 8) + (ff_inner * 4)) + 67))] = (compute1[((((nn_inner * 8) + (ff_inner * 4)) + 67))] + (pad_temp_shared[(((((nn_inner * 336) + (rc_outer_inner * 84)) + ((((int)threadIdx.x) % 21) * 2)) + 43))] * input1_shared[((((((((int)threadIdx.x) / 21) * 8) + (ff_inner * 4)) + rc_outer_inner) + 64))]));
        }
      }
    }
  }
  for (int i0_inner = 0; i0_inner < 4; ++i0_inner) {
    for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
      for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
        for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
          compute[(((((((((((((int)blockIdx.x) / 147) * 1185408) + (i0_inner * 296352)) + (((((int)blockIdx.x) % 147) / 21) * 42336)) + ((((int)threadIdx.x) / 21) * 3528)) + (i1_inner * 1764)) + ((((int)blockIdx.x) % 21) * 84)) + (i2_inner * 42)) + ((((int)threadIdx.x) % 21) * 2)) + i3_inner))] = max((compute1[(((((i0_inner * 8) + (i1_inner * 4)) + (i2_inner * 2)) + i3_inner))] + input2[(((((((((((((int)blockIdx.x) / 147) * 1185408) + (i0_inner * 296352)) + (((((int)blockIdx.x) % 147) / 21) * 42336)) + ((((int)threadIdx.x) / 21) * 3528)) + (i1_inner * 1764)) + ((((int)blockIdx.x) % 21) * 84)) + (i2_inner * 42)) + ((((int)threadIdx.x) % 21) * 2)) + i3_inner))]), 0.000000e+00f);
          compute[((((((((((((((int)blockIdx.x) / 147) * 1185408) + (i0_inner * 296352)) + (((((int)blockIdx.x) % 147) / 21) * 42336)) + ((((int)threadIdx.x) / 21) * 3528)) + (i1_inner * 1764)) + ((((int)blockIdx.x) % 21) * 84)) + (i2_inner * 42)) + ((((int)threadIdx.x) % 21) * 2)) + i3_inner) + 14112))] = max((compute1[((((((i0_inner * 8) + (i1_inner * 4)) + (i2_inner * 2)) + i3_inner) + 32))] + input2[((((((((((((((int)blockIdx.x) / 147) * 1185408) + (i0_inner * 296352)) + (((((int)blockIdx.x) % 147) / 21) * 42336)) + ((((int)threadIdx.x) / 21) * 3528)) + (i1_inner * 1764)) + ((((int)blockIdx.x) % 21) * 84)) + (i2_inner * 42)) + ((((int)threadIdx.x) % 21) * 2)) + i3_inner) + 14112))]), 0.000000e+00f);
          compute[((((((((((((((int)blockIdx.x) / 147) * 1185408) + (i0_inner * 296352)) + (((((int)blockIdx.x) % 147) / 21) * 42336)) + ((((int)threadIdx.x) / 21) * 3528)) + (i1_inner * 1764)) + ((((int)blockIdx.x) % 21) * 84)) + (i2_inner * 42)) + ((((int)threadIdx.x) % 21) * 2)) + i3_inner) + 28224))] = max((compute1[((((((i0_inner * 8) + (i1_inner * 4)) + (i2_inner * 2)) + i3_inner) + 64))] + input2[((((((((((((((int)blockIdx.x) / 147) * 1185408) + (i0_inner * 296352)) + (((((int)blockIdx.x) % 147) / 21) * 42336)) + ((((int)threadIdx.x) / 21) * 3528)) + (i1_inner * 1764)) + ((((int)blockIdx.x) % 21) * 84)) + (i2_inner * 42)) + ((((int)threadIdx.x) % 21) * 2)) + i3_inner) + 28224))]), 0.000000e+00f);
        }
      }
    }
  }
}

