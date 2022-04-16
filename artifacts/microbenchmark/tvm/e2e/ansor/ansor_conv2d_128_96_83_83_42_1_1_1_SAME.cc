//2656_1_1_249_1_1
//128_96_83_83_42_1_1_SAME
//dim3 grid(2656, 1, 1);
//dim3 block(249, 1, 1);

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
extern "C" __global__ void __launch_bounds__(249) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ compute) {
  float compute_local[56];
  __shared__ float pad_temp_shared[3984];
  __shared__ float input1_shared[504];
  for (int nn_c_outer_inner_init = 0; nn_c_outer_inner_init < 2; ++nn_c_outer_inner_init) {
    for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 7; ++ff_c_outer_inner_init) {
      compute_local[(((nn_c_outer_inner_init * 14) + ff_c_outer_inner_init))] = 0.000000e+00f;
      compute_local[((((nn_c_outer_inner_init * 14) + ff_c_outer_inner_init) + 28))] = 0.000000e+00f;
      compute_local[((((nn_c_outer_inner_init * 14) + ff_c_outer_inner_init) + 7))] = 0.000000e+00f;
      compute_local[((((nn_c_outer_inner_init * 14) + ff_c_outer_inner_init) + 35))] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[(((int)threadIdx.x))] = input0[(((((((((int)blockIdx.x) / 83) * 2645376) + (rc_outer_outer * 82668)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 249))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (rc_outer_outer * 82668)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 20667))];
    pad_temp_shared[((((int)threadIdx.x) + 498))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (rc_outer_outer * 82668)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 41334))];
    pad_temp_shared[((((int)threadIdx.x) + 747))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (rc_outer_outer * 82668)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 62001))];
    pad_temp_shared[((((int)threadIdx.x) + 996))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (rc_outer_outer * 82668)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 661344))];
    pad_temp_shared[((((int)threadIdx.x) + 1245))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (((((int)threadIdx.x) + 1245) / 996) * 661344)) + (rc_outer_outer * 82668)) + (((((int)threadIdx.x) / 83) + 3) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 1494))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (((((int)threadIdx.x) + 1494) / 996) * 661344)) + (rc_outer_outer * 82668)) + (((((int)threadIdx.x) / 83) + 6) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 1743))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (((((int)threadIdx.x) + 1743) / 996) * 661344)) + (rc_outer_outer * 82668)) + (((((int)threadIdx.x) / 83) + 9) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 1992))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (rc_outer_outer * 82668)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1322688))];
    pad_temp_shared[((((int)threadIdx.x) + 2241))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (((((int)threadIdx.x) + 2241) / 996) * 661344)) + (rc_outer_outer * 82668)) + (((((int)threadIdx.x) / 83) + 3) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 2490))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (((((int)threadIdx.x) + 2490) / 996) * 661344)) + (rc_outer_outer * 82668)) + (((((int)threadIdx.x) / 83) + 6) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 2739))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (((((int)threadIdx.x) + 2739) / 996) * 661344)) + (rc_outer_outer * 82668)) + (((((int)threadIdx.x) / 83) + 9) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 2988))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (rc_outer_outer * 82668)) + ((((int)threadIdx.x) / 83) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 1984032))];
    pad_temp_shared[((((int)threadIdx.x) + 3237))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (((((int)threadIdx.x) + 3237) / 996) * 661344)) + (rc_outer_outer * 82668)) + (((((int)threadIdx.x) / 83) + 3) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 3486))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (((((int)threadIdx.x) + 3486) / 996) * 661344)) + (rc_outer_outer * 82668)) + (((((int)threadIdx.x) / 83) + 6) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    pad_temp_shared[((((int)threadIdx.x) + 3735))] = input0[((((((((((int)blockIdx.x) / 83) * 2645376) + (((((int)threadIdx.x) + 3735) / 996) * 661344)) + (rc_outer_outer * 82668)) + (((((int)threadIdx.x) / 83) + 9) * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))];
    input1_shared[(((int)threadIdx.x))] = input1[(((((((int)threadIdx.x) / 12) * 96) + (rc_outer_outer * 12)) + (((int)threadIdx.x) % 12)))];
    input1_shared[((((int)threadIdx.x) + 249))] = input1[((((((((int)threadIdx.x) + 249) / 12) * 96) + (rc_outer_outer * 12)) + ((((int)threadIdx.x) + 9) % 12)))];
    if (((int)threadIdx.x) < 6) {
      input1_shared[((((int)threadIdx.x) + 498))] = input1[((((((((int)threadIdx.x) + 498) / 12) * 96) + (rc_outer_outer * 12)) + (((int)threadIdx.x) + 6)))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int nn_c_outer_inner = 0; nn_c_outer_inner < 2; ++nn_c_outer_inner) {
        for (int ff_c_outer_inner = 0; ff_c_outer_inner < 7; ++ff_c_outer_inner) {
          for (int rc_inner = 0; rc_inner < 6; ++rc_inner) {
            compute_local[(((nn_c_outer_inner * 14) + ff_c_outer_inner))] = (compute_local[(((nn_c_outer_inner * 14) + ff_c_outer_inner))] + (pad_temp_shared[(((((nn_c_outer_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[((((((((int)threadIdx.x) / 83) * 84) + (ff_c_outer_inner * 12)) + (rc_outer_inner * 6)) + rc_inner))]));
            compute_local[((((nn_c_outer_inner * 14) + ff_c_outer_inner) + 28))] = (compute_local[((((nn_c_outer_inner * 14) + ff_c_outer_inner) + 28))] + (pad_temp_shared[(((((nn_c_outer_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)))] * input1_shared[(((((((((int)threadIdx.x) / 83) * 84) + (ff_c_outer_inner * 12)) + (rc_outer_inner * 6)) + rc_inner) + 252))]));
            compute_local[((((nn_c_outer_inner * 14) + ff_c_outer_inner) + 7))] = (compute_local[((((nn_c_outer_inner * 14) + ff_c_outer_inner) + 7))] + (pad_temp_shared[((((((nn_c_outer_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 996))] * input1_shared[((((((((int)threadIdx.x) / 83) * 84) + (ff_c_outer_inner * 12)) + (rc_outer_inner * 6)) + rc_inner))]));
            compute_local[((((nn_c_outer_inner * 14) + ff_c_outer_inner) + 35))] = (compute_local[((((nn_c_outer_inner * 14) + ff_c_outer_inner) + 35))] + (pad_temp_shared[((((((nn_c_outer_inner * 1992) + (rc_outer_inner * 498)) + (rc_inner * 83)) + (((int)threadIdx.x) % 83)) + 996))] * input1_shared[(((((((((int)threadIdx.x) / 83) * 84) + (ff_c_outer_inner * 12)) + (rc_outer_inner * 6)) + rc_inner) + 252))]));
          }
        }
      }
    }
  }
  for (int nn_inner = 0; nn_inner < 4; ++nn_inner) {
    for (int ff_inner = 0; ff_inner < 7; ++ff_inner) {
      compute[((((((((((int)blockIdx.x) / 83) * 1157352) + (nn_inner * 289338)) + ((((int)threadIdx.x) / 83) * 48223)) + (ff_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)))] = compute_local[(((nn_inner * 7) + ff_inner))];
      compute[(((((((((((int)blockIdx.x) / 83) * 1157352) + (nn_inner * 289338)) + ((((int)threadIdx.x) / 83) * 48223)) + (ff_inner * 6889)) + ((((int)blockIdx.x) % 83) * 83)) + (((int)threadIdx.x) % 83)) + 144669))] = compute_local[((((nn_inner * 7) + ff_inner) + 28))];
    }
  }
}

