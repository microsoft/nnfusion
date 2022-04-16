//224_1_1_896_1_1
//128_256_14_14_256_3_1_SAME
//dim3 grid(224, 1, 1);
//dim3 block(896, 1, 1);

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
extern "C" __global__ void __launch_bounds__(896) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[32];
  __shared__ float pad_temp_shared[1024];
  __shared__ float input1_shared[9216];
  for (int nn_outer_inner_init = 0; nn_outer_inner_init < 4; ++nn_outer_inner_init) {
    compute1[((nn_outer_inner_init * 4))] = 0.000000e+00f;
    compute1[(((nn_outer_inner_init * 4) + 16))] = 0.000000e+00f;
    compute1[(((nn_outer_inner_init * 4) + 1))] = 0.000000e+00f;
    compute1[(((nn_outer_inner_init * 4) + 17))] = 0.000000e+00f;
    compute1[(((nn_outer_inner_init * 4) + 2))] = 0.000000e+00f;
    compute1[(((nn_outer_inner_init * 4) + 18))] = 0.000000e+00f;
    compute1[(((nn_outer_inner_init * 4) + 3))] = 0.000000e+00f;
    compute1[(((nn_outer_inner_init * 4) + 19))] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[(((int)threadIdx.x))] = (((((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 63) >> 4))) && ((((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 63) >> 4)) < 15)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? input0[((((((((((((int)blockIdx.x) / 7) * 200704) + ((((int)threadIdx.x) >> 8) * 50176)) + (rc_outer_outer * 784)) + (((((int)threadIdx.x) & 255) >> 6) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((int)threadIdx.x) & 63) >> 4) * 14)) + (((int)threadIdx.x) & 15)) - 15))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 128) {
      pad_temp_shared[((((int)threadIdx.x) + 896))] = (((((1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 63) >> 4))) && ((((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 63) >> 4)) < 15)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? input0[((((((((((((int)blockIdx.x) / 7) * 200704) + (((((int)threadIdx.x) + 896) >> 8) * 50176)) + (rc_outer_outer * 784)) + (((((int)threadIdx.x) >> 6) + 2) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((((int)threadIdx.x) & 63) >> 4) * 14)) + (((int)threadIdx.x) & 15)) - 15))] : 0.000000e+00f);
    }
    input1_shared[((((int)threadIdx.x) * 3))] = input1[(((((((int)threadIdx.x) / 12) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) % 12) * 3)))];
    input1_shared[(((((int)threadIdx.x) * 3) + 1))] = input1[((((((((int)threadIdx.x) / 12) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) % 12) * 3)) + 1))];
    input1_shared[(((((int)threadIdx.x) * 3) + 2))] = input1[((((((((int)threadIdx.x) / 12) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) % 12) * 3)) + 2))];
    input1_shared[(((((int)threadIdx.x) * 3) + 2688))] = input1[((((((((int)threadIdx.x) + 896) / 12) * 2304) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) + 8) % 12) * 3)))];
    input1_shared[(((((int)threadIdx.x) * 3) + 2689))] = input1[(((((((((int)threadIdx.x) + 896) / 12) * 2304) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) + 8) % 12) * 3)) + 1))];
    input1_shared[(((((int)threadIdx.x) * 3) + 2690))] = input1[(((((((((int)threadIdx.x) + 896) / 12) * 2304) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) + 8) % 12) * 3)) + 2))];
    input1_shared[(((((int)threadIdx.x) * 3) + 5376))] = input1[((((((((int)threadIdx.x) + 1792) / 12) * 2304) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) + 4) % 12) * 3)))];
    input1_shared[(((((int)threadIdx.x) * 3) + 5377))] = input1[(((((((((int)threadIdx.x) + 1792) / 12) * 2304) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) + 4) % 12) * 3)) + 1))];
    input1_shared[(((((int)threadIdx.x) * 3) + 5378))] = input1[(((((((((int)threadIdx.x) + 1792) / 12) * 2304) + (rc_outer_outer * 36)) + (((((int)threadIdx.x) + 4) % 12) * 3)) + 2))];
    if (((int)threadIdx.x) < 384) {
      input1_shared[(((((int)threadIdx.x) * 3) + 8064))] = input1[((((((((int)threadIdx.x) / 12) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) % 12) * 3)) + 516096))];
      input1_shared[(((((int)threadIdx.x) * 3) + 8065))] = input1[((((((((int)threadIdx.x) / 12) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) % 12) * 3)) + 516097))];
      input1_shared[(((((int)threadIdx.x) * 3) + 8066))] = input1[((((((((int)threadIdx.x) / 12) * 2304) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) % 12) * 3)) + 516098))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
        for (int nn_outer_inner = 0; nn_outer_inner < 4; ++nn_outer_inner) {
          for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
            compute1[((nn_outer_inner * 4))] = (compute1[((nn_outer_inner * 4))] + (pad_temp_shared[((((((nn_outer_inner * 256) + (rc_outer_inner * 64)) + (ry_inner * 16)) + rx_outer_inner) + (((int)threadIdx.x) % 14)))] * input1_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_outer_inner))]));
            compute1[(((nn_outer_inner * 4) + 16))] = (compute1[(((nn_outer_inner * 4) + 16))] + (pad_temp_shared[((((((nn_outer_inner * 256) + (rc_outer_inner * 64)) + (ry_inner * 16)) + rx_outer_inner) + (((int)threadIdx.x) % 14)))] * input1_shared[(((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 4608))]));
            compute1[(((nn_outer_inner * 4) + 1))] = (compute1[(((nn_outer_inner * 4) + 1))] + (pad_temp_shared[(((((((nn_outer_inner * 256) + (rc_outer_inner * 64)) + (ry_inner * 16)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 16))] * input1_shared[((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_outer_inner))]));
            compute1[(((nn_outer_inner * 4) + 17))] = (compute1[(((nn_outer_inner * 4) + 17))] + (pad_temp_shared[(((((((nn_outer_inner * 256) + (rc_outer_inner * 64)) + (ry_inner * 16)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 16))] * input1_shared[(((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 4608))]));
            compute1[(((nn_outer_inner * 4) + 2))] = (compute1[(((nn_outer_inner * 4) + 2))] + (pad_temp_shared[((((((nn_outer_inner * 256) + (rc_outer_inner * 64)) + (ry_inner * 16)) + rx_outer_inner) + (((int)threadIdx.x) % 14)))] * input1_shared[(((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 36))]));
            compute1[(((nn_outer_inner * 4) + 18))] = (compute1[(((nn_outer_inner * 4) + 18))] + (pad_temp_shared[((((((nn_outer_inner * 256) + (rc_outer_inner * 64)) + (ry_inner * 16)) + rx_outer_inner) + (((int)threadIdx.x) % 14)))] * input1_shared[(((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 4644))]));
            compute1[(((nn_outer_inner * 4) + 3))] = (compute1[(((nn_outer_inner * 4) + 3))] + (pad_temp_shared[(((((((nn_outer_inner * 256) + (rc_outer_inner * 64)) + (ry_inner * 16)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 16))] * input1_shared[(((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 36))]));
            compute1[(((nn_outer_inner * 4) + 19))] = (compute1[(((nn_outer_inner * 4) + 19))] + (pad_temp_shared[(((((((nn_outer_inner * 256) + (rc_outer_inner * 64)) + (ry_inner * 16)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 16))] * input1_shared[(((((((((int)threadIdx.x) / 14) * 72) + (rc_outer_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 4644))]));
          }
        }
      }
    }
  }
  for (int i0_inner = 0; i0_inner < 4; ++i0_inner) {
    for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
      for (int i2_inner = 0; i2_inner < 2; ++i2_inner) {
        compute[(((((((((((int)blockIdx.x) / 7) * 200704) + (i0_inner * 50176)) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (i2_inner * 14)) + (((int)threadIdx.x) % 14)))] = max((compute1[((((i0_inner * 4) + (i1_inner * 2)) + i2_inner))] + input2[(((((((((((int)blockIdx.x) / 7) * 200704) + (i0_inner * 50176)) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (i2_inner * 14)) + (((int)threadIdx.x) % 14)))]), 0.000000e+00f);
        compute[((((((((((((int)blockIdx.x) / 7) * 200704) + (i0_inner * 50176)) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (i2_inner * 14)) + (((int)threadIdx.x) % 14)) + 25088))] = max((compute1[(((((i0_inner * 4) + (i1_inner * 2)) + i2_inner) + 16))] + input2[((((((((((((int)blockIdx.x) / 7) * 200704) + (i0_inner * 50176)) + ((((int)threadIdx.x) / 14) * 392)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (i2_inner * 14)) + (((int)threadIdx.x) % 14)) + 25088))]), 0.000000e+00f);
      }
    }
  }
}

