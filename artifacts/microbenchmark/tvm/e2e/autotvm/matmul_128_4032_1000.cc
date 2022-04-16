//8_50_1_4_4_1
//128_4032_1000
//dim3 grid(8, 50, 1);
//dim3 block(4, 4, 1);

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
extern "C" __global__ void __launch_bounds__(16) matmul_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[20];
  __shared__ float placeholder_shared[576];
  __shared__ float placeholder_d_shared[720];
  float placeholder_shared_local[24];
  float placeholder_d_shared_local[30];
  float placeholder_shared_local1[24];
  float placeholder_d_shared_local1[30];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    T_matmul_NN_local[(i_c_init)] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_init + 4))] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_init + 8))] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_init + 12))] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_init + 16))] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_init + 2))] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_init + 6))] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_init + 10))] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_init + 14))] = 0.000000e+00f;
    T_matmul_NN_local[((i_c_init + 18))] = 0.000000e+00f;
  }
  for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
        if ((((ax1_outer * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner) < 18) {
          placeholder_shared[((((((((int)threadIdx.y) * 72) + (ax0_inner * 18)) + (ax1_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[(((((((((int)blockIdx.x) * 64512) + (((int)threadIdx.y) * 16128)) + (ax0_inner * 4032)) + (ax1_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
        }
      }
    }
  }
  for (int ax0_inner1 = 0; ax0_inner1 < 5; ++ax0_inner1) {
    for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
      for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
        if (((((int)threadIdx.y) * 5) + ax0_inner1) < 18) {
          if ((((ax1_outer1 * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1) < 20) {
            if (((((((int)blockIdx.y) * 20) + (ax1_outer1 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1) < 1000) {
              placeholder_d_shared[((((((((int)threadIdx.y) * 100) + (ax0_inner1 * 20)) + (ax1_outer1 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[(((((((((int)threadIdx.y) * 5000) + (ax0_inner1 * 1000)) + (((int)blockIdx.y) * 20)) + (ax1_outer1 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
            }
          }
        }
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 223; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner2 = 0; ax0_inner2 < 4; ++ax0_inner2) {
      for (int ax1_outer2 = 0; ax1_outer2 < 2; ++ax1_outer2) {
        for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
          if ((((ax1_outer2 * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) < 18) {
            if (((((k_outer_outer * 18) + (ax1_outer2 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) < 4014) {
              placeholder_shared[(((((((((k_outer_outer + 1) & 1) * 288) + (((int)threadIdx.y) * 72)) + (ax0_inner2 * 18)) + (ax1_outer2 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[(((((((((((int)blockIdx.x) * 64512) + (((int)threadIdx.y) * 16128)) + (ax0_inner2 * 4032)) + (k_outer_outer * 18)) + (ax1_outer2 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) + 18))];
            }
          }
        }
      }
    }
    for (int ax0_inner3 = 0; ax0_inner3 < 5; ++ax0_inner3) {
      for (int ax1_outer3 = 0; ax1_outer3 < 2; ++ax1_outer3) {
        for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
          if (((((int)threadIdx.y) * 5) + ax0_inner3) < 18) {
            if ((((ax1_outer3 * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) < 20) {
              if ((((k_outer_outer * 18) + (((int)threadIdx.y) * 5)) + ax0_inner3) < 4014) {
                if (((((((int)blockIdx.y) * 20) + (ax1_outer3 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) < 1000) {
                  placeholder_d_shared[(((((((((k_outer_outer + 1) & 1) * 360) + (((int)threadIdx.y) * 100)) + (ax0_inner3 * 20)) + (ax1_outer3 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[(((((((((k_outer_outer * 18000) + (((int)threadIdx.y) * 5000)) + (ax0_inner3 * 1000)) + (((int)blockIdx.y) * 20)) + (ax1_outer3 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 18000))];
                }
              }
            }
          }
        }
      }
    }
    for (int ax0 = 0; ax0 < 2; ++ax0) {
      for (int ax1 = 0; ax1 < 6; ++ax1) {
        placeholder_shared_local[(((ax0 * 6) + ax1))] = placeholder_shared[((((((k_outer_outer & 1) * 288) + (((int)threadIdx.x) * 36)) + (ax0 * 18)) + ax1))];
        placeholder_shared_local[((((ax0 * 6) + ax1) + 12))] = placeholder_shared[(((((((k_outer_outer & 1) * 288) + (((int)threadIdx.x) * 36)) + (ax0 * 18)) + ax1) + 144))];
      }
    }
    for (int ax01 = 0; ax01 < 6; ++ax01) {
      placeholder_d_shared_local[(ax01)] = placeholder_d_shared[(((((k_outer_outer & 1) * 360) + (ax01 * 20)) + ((int)threadIdx.y)))];
      placeholder_d_shared_local[((ax01 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax01 * 20)) + ((int)threadIdx.y)) + 4))];
      placeholder_d_shared_local[((ax01 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax01 * 20)) + ((int)threadIdx.y)) + 8))];
      placeholder_d_shared_local[((ax01 + 18))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax01 * 20)) + ((int)threadIdx.y)) + 12))];
      placeholder_d_shared_local[((ax01 + 24))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax01 * 20)) + ((int)threadIdx.y)) + 16))];
    }
    for (int k_inner_inner = 0; k_inner_inner < 6; ++k_inner_inner) {
      for (int i_c = 0; i_c < 2; ++i_c) {
        T_matmul_NN_local[(i_c)] = (T_matmul_NN_local[(i_c)] + (placeholder_shared_local[(((i_c * 6) + k_inner_inner))] * placeholder_d_shared_local[(k_inner_inner)]));
        T_matmul_NN_local[((i_c + 4))] = (T_matmul_NN_local[((i_c + 4))] + (placeholder_shared_local[(((i_c * 6) + k_inner_inner))] * placeholder_d_shared_local[((k_inner_inner + 6))]));
        T_matmul_NN_local[((i_c + 8))] = (T_matmul_NN_local[((i_c + 8))] + (placeholder_shared_local[(((i_c * 6) + k_inner_inner))] * placeholder_d_shared_local[((k_inner_inner + 12))]));
        T_matmul_NN_local[((i_c + 12))] = (T_matmul_NN_local[((i_c + 12))] + (placeholder_shared_local[(((i_c * 6) + k_inner_inner))] * placeholder_d_shared_local[((k_inner_inner + 18))]));
        T_matmul_NN_local[((i_c + 16))] = (T_matmul_NN_local[((i_c + 16))] + (placeholder_shared_local[(((i_c * 6) + k_inner_inner))] * placeholder_d_shared_local[((k_inner_inner + 24))]));
        T_matmul_NN_local[((i_c + 2))] = (T_matmul_NN_local[((i_c + 2))] + (placeholder_shared_local[((((i_c * 6) + k_inner_inner) + 12))] * placeholder_d_shared_local[(k_inner_inner)]));
        T_matmul_NN_local[((i_c + 6))] = (T_matmul_NN_local[((i_c + 6))] + (placeholder_shared_local[((((i_c * 6) + k_inner_inner) + 12))] * placeholder_d_shared_local[((k_inner_inner + 6))]));
        T_matmul_NN_local[((i_c + 10))] = (T_matmul_NN_local[((i_c + 10))] + (placeholder_shared_local[((((i_c * 6) + k_inner_inner) + 12))] * placeholder_d_shared_local[((k_inner_inner + 12))]));
        T_matmul_NN_local[((i_c + 14))] = (T_matmul_NN_local[((i_c + 14))] + (placeholder_shared_local[((((i_c * 6) + k_inner_inner) + 12))] * placeholder_d_shared_local[((k_inner_inner + 18))]));
        T_matmul_NN_local[((i_c + 18))] = (T_matmul_NN_local[((i_c + 18))] + (placeholder_shared_local[((((i_c * 6) + k_inner_inner) + 12))] * placeholder_d_shared_local[((k_inner_inner + 24))]));
      }
    }
    for (int ax02 = 0; ax02 < 2; ++ax02) {
      for (int ax11 = 0; ax11 < 6; ++ax11) {
        placeholder_shared_local[(((ax02 * 6) + ax11))] = placeholder_shared[(((((((k_outer_outer & 1) * 288) + (((int)threadIdx.x) * 36)) + (ax02 * 18)) + ax11) + 6))];
        placeholder_shared_local[((((ax02 * 6) + ax11) + 12))] = placeholder_shared[(((((((k_outer_outer & 1) * 288) + (((int)threadIdx.x) * 36)) + (ax02 * 18)) + ax11) + 150))];
      }
    }
    for (int ax03 = 0; ax03 < 6; ++ax03) {
      placeholder_d_shared_local[(ax03)] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax03 * 20)) + ((int)threadIdx.y)) + 120))];
      placeholder_d_shared_local[((ax03 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax03 * 20)) + ((int)threadIdx.y)) + 124))];
      placeholder_d_shared_local[((ax03 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax03 * 20)) + ((int)threadIdx.y)) + 128))];
      placeholder_d_shared_local[((ax03 + 18))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax03 * 20)) + ((int)threadIdx.y)) + 132))];
      placeholder_d_shared_local[((ax03 + 24))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax03 * 20)) + ((int)threadIdx.y)) + 136))];
    }
    for (int k_inner_inner1 = 0; k_inner_inner1 < 6; ++k_inner_inner1) {
      for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
        T_matmul_NN_local[(i_c1)] = (T_matmul_NN_local[(i_c1)] + (placeholder_shared_local[(((i_c1 * 6) + k_inner_inner1))] * placeholder_d_shared_local[(k_inner_inner1)]));
        T_matmul_NN_local[((i_c1 + 4))] = (T_matmul_NN_local[((i_c1 + 4))] + (placeholder_shared_local[(((i_c1 * 6) + k_inner_inner1))] * placeholder_d_shared_local[((k_inner_inner1 + 6))]));
        T_matmul_NN_local[((i_c1 + 8))] = (T_matmul_NN_local[((i_c1 + 8))] + (placeholder_shared_local[(((i_c1 * 6) + k_inner_inner1))] * placeholder_d_shared_local[((k_inner_inner1 + 12))]));
        T_matmul_NN_local[((i_c1 + 12))] = (T_matmul_NN_local[((i_c1 + 12))] + (placeholder_shared_local[(((i_c1 * 6) + k_inner_inner1))] * placeholder_d_shared_local[((k_inner_inner1 + 18))]));
        T_matmul_NN_local[((i_c1 + 16))] = (T_matmul_NN_local[((i_c1 + 16))] + (placeholder_shared_local[(((i_c1 * 6) + k_inner_inner1))] * placeholder_d_shared_local[((k_inner_inner1 + 24))]));
        T_matmul_NN_local[((i_c1 + 2))] = (T_matmul_NN_local[((i_c1 + 2))] + (placeholder_shared_local[((((i_c1 * 6) + k_inner_inner1) + 12))] * placeholder_d_shared_local[(k_inner_inner1)]));
        T_matmul_NN_local[((i_c1 + 6))] = (T_matmul_NN_local[((i_c1 + 6))] + (placeholder_shared_local[((((i_c1 * 6) + k_inner_inner1) + 12))] * placeholder_d_shared_local[((k_inner_inner1 + 6))]));
        T_matmul_NN_local[((i_c1 + 10))] = (T_matmul_NN_local[((i_c1 + 10))] + (placeholder_shared_local[((((i_c1 * 6) + k_inner_inner1) + 12))] * placeholder_d_shared_local[((k_inner_inner1 + 12))]));
        T_matmul_NN_local[((i_c1 + 14))] = (T_matmul_NN_local[((i_c1 + 14))] + (placeholder_shared_local[((((i_c1 * 6) + k_inner_inner1) + 12))] * placeholder_d_shared_local[((k_inner_inner1 + 18))]));
        T_matmul_NN_local[((i_c1 + 18))] = (T_matmul_NN_local[((i_c1 + 18))] + (placeholder_shared_local[((((i_c1 * 6) + k_inner_inner1) + 12))] * placeholder_d_shared_local[((k_inner_inner1 + 24))]));
      }
    }
    for (int ax04 = 0; ax04 < 2; ++ax04) {
      for (int ax12 = 0; ax12 < 6; ++ax12) {
        placeholder_shared_local[(((ax04 * 6) + ax12))] = placeholder_shared[(((((((k_outer_outer & 1) * 288) + (((int)threadIdx.x) * 36)) + (ax04 * 18)) + ax12) + 12))];
        placeholder_shared_local[((((ax04 * 6) + ax12) + 12))] = placeholder_shared[(((((((k_outer_outer & 1) * 288) + (((int)threadIdx.x) * 36)) + (ax04 * 18)) + ax12) + 156))];
      }
    }
    for (int ax05 = 0; ax05 < 6; ++ax05) {
      placeholder_d_shared_local[(ax05)] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax05 * 20)) + ((int)threadIdx.y)) + 240))];
      placeholder_d_shared_local[((ax05 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax05 * 20)) + ((int)threadIdx.y)) + 244))];
      placeholder_d_shared_local[((ax05 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax05 * 20)) + ((int)threadIdx.y)) + 248))];
      placeholder_d_shared_local[((ax05 + 18))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax05 * 20)) + ((int)threadIdx.y)) + 252))];
      placeholder_d_shared_local[((ax05 + 24))] = placeholder_d_shared[((((((k_outer_outer & 1) * 360) + (ax05 * 20)) + ((int)threadIdx.y)) + 256))];
    }
    for (int k_inner_inner2 = 0; k_inner_inner2 < 6; ++k_inner_inner2) {
      for (int i_c2 = 0; i_c2 < 2; ++i_c2) {
        T_matmul_NN_local[(i_c2)] = (T_matmul_NN_local[(i_c2)] + (placeholder_shared_local[(((i_c2 * 6) + k_inner_inner2))] * placeholder_d_shared_local[(k_inner_inner2)]));
        T_matmul_NN_local[((i_c2 + 4))] = (T_matmul_NN_local[((i_c2 + 4))] + (placeholder_shared_local[(((i_c2 * 6) + k_inner_inner2))] * placeholder_d_shared_local[((k_inner_inner2 + 6))]));
        T_matmul_NN_local[((i_c2 + 8))] = (T_matmul_NN_local[((i_c2 + 8))] + (placeholder_shared_local[(((i_c2 * 6) + k_inner_inner2))] * placeholder_d_shared_local[((k_inner_inner2 + 12))]));
        T_matmul_NN_local[((i_c2 + 12))] = (T_matmul_NN_local[((i_c2 + 12))] + (placeholder_shared_local[(((i_c2 * 6) + k_inner_inner2))] * placeholder_d_shared_local[((k_inner_inner2 + 18))]));
        T_matmul_NN_local[((i_c2 + 16))] = (T_matmul_NN_local[((i_c2 + 16))] + (placeholder_shared_local[(((i_c2 * 6) + k_inner_inner2))] * placeholder_d_shared_local[((k_inner_inner2 + 24))]));
        T_matmul_NN_local[((i_c2 + 2))] = (T_matmul_NN_local[((i_c2 + 2))] + (placeholder_shared_local[((((i_c2 * 6) + k_inner_inner2) + 12))] * placeholder_d_shared_local[(k_inner_inner2)]));
        T_matmul_NN_local[((i_c2 + 6))] = (T_matmul_NN_local[((i_c2 + 6))] + (placeholder_shared_local[((((i_c2 * 6) + k_inner_inner2) + 12))] * placeholder_d_shared_local[((k_inner_inner2 + 6))]));
        T_matmul_NN_local[((i_c2 + 10))] = (T_matmul_NN_local[((i_c2 + 10))] + (placeholder_shared_local[((((i_c2 * 6) + k_inner_inner2) + 12))] * placeholder_d_shared_local[((k_inner_inner2 + 12))]));
        T_matmul_NN_local[((i_c2 + 14))] = (T_matmul_NN_local[((i_c2 + 14))] + (placeholder_shared_local[((((i_c2 * 6) + k_inner_inner2) + 12))] * placeholder_d_shared_local[((k_inner_inner2 + 18))]));
        T_matmul_NN_local[((i_c2 + 18))] = (T_matmul_NN_local[((i_c2 + 18))] + (placeholder_shared_local[((((i_c2 * 6) + k_inner_inner2) + 12))] * placeholder_d_shared_local[((k_inner_inner2 + 24))]));
      }
    }
  }
  __syncthreads();
  for (int ax06 = 0; ax06 < 2; ++ax06) {
    for (int ax13 = 0; ax13 < 6; ++ax13) {
      placeholder_shared_local1[(((ax06 * 6) + ax13))] = placeholder_shared[(((((((int)threadIdx.x) * 36) + (ax06 * 18)) + ax13) + 288))];
      placeholder_shared_local1[((((ax06 * 6) + ax13) + 12))] = placeholder_shared[(((((((int)threadIdx.x) * 36) + (ax06 * 18)) + ax13) + 432))];
    }
  }
  for (int ax07 = 0; ax07 < 6; ++ax07) {
    placeholder_d_shared_local1[(ax07)] = placeholder_d_shared[((((ax07 * 20) + ((int)threadIdx.y)) + 360))];
    placeholder_d_shared_local1[((ax07 + 6))] = placeholder_d_shared[((((ax07 * 20) + ((int)threadIdx.y)) + 364))];
    placeholder_d_shared_local1[((ax07 + 12))] = placeholder_d_shared[((((ax07 * 20) + ((int)threadIdx.y)) + 368))];
    placeholder_d_shared_local1[((ax07 + 18))] = placeholder_d_shared[((((ax07 * 20) + ((int)threadIdx.y)) + 372))];
    placeholder_d_shared_local1[((ax07 + 24))] = placeholder_d_shared[((((ax07 * 20) + ((int)threadIdx.y)) + 376))];
  }
  for (int k_inner_inner3 = 0; k_inner_inner3 < 6; ++k_inner_inner3) {
    for (int i_c3 = 0; i_c3 < 2; ++i_c3) {
      T_matmul_NN_local[(i_c3)] = (T_matmul_NN_local[(i_c3)] + (placeholder_shared_local1[(((i_c3 * 6) + k_inner_inner3))] * placeholder_d_shared_local1[(k_inner_inner3)]));
      T_matmul_NN_local[((i_c3 + 4))] = (T_matmul_NN_local[((i_c3 + 4))] + (placeholder_shared_local1[(((i_c3 * 6) + k_inner_inner3))] * placeholder_d_shared_local1[((k_inner_inner3 + 6))]));
      T_matmul_NN_local[((i_c3 + 8))] = (T_matmul_NN_local[((i_c3 + 8))] + (placeholder_shared_local1[(((i_c3 * 6) + k_inner_inner3))] * placeholder_d_shared_local1[((k_inner_inner3 + 12))]));
      T_matmul_NN_local[((i_c3 + 12))] = (T_matmul_NN_local[((i_c3 + 12))] + (placeholder_shared_local1[(((i_c3 * 6) + k_inner_inner3))] * placeholder_d_shared_local1[((k_inner_inner3 + 18))]));
      T_matmul_NN_local[((i_c3 + 16))] = (T_matmul_NN_local[((i_c3 + 16))] + (placeholder_shared_local1[(((i_c3 * 6) + k_inner_inner3))] * placeholder_d_shared_local1[((k_inner_inner3 + 24))]));
      T_matmul_NN_local[((i_c3 + 2))] = (T_matmul_NN_local[((i_c3 + 2))] + (placeholder_shared_local1[((((i_c3 * 6) + k_inner_inner3) + 12))] * placeholder_d_shared_local1[(k_inner_inner3)]));
      T_matmul_NN_local[((i_c3 + 6))] = (T_matmul_NN_local[((i_c3 + 6))] + (placeholder_shared_local1[((((i_c3 * 6) + k_inner_inner3) + 12))] * placeholder_d_shared_local1[((k_inner_inner3 + 6))]));
      T_matmul_NN_local[((i_c3 + 10))] = (T_matmul_NN_local[((i_c3 + 10))] + (placeholder_shared_local1[((((i_c3 * 6) + k_inner_inner3) + 12))] * placeholder_d_shared_local1[((k_inner_inner3 + 12))]));
      T_matmul_NN_local[((i_c3 + 14))] = (T_matmul_NN_local[((i_c3 + 14))] + (placeholder_shared_local1[((((i_c3 * 6) + k_inner_inner3) + 12))] * placeholder_d_shared_local1[((k_inner_inner3 + 18))]));
      T_matmul_NN_local[((i_c3 + 18))] = (T_matmul_NN_local[((i_c3 + 18))] + (placeholder_shared_local1[((((i_c3 * 6) + k_inner_inner3) + 12))] * placeholder_d_shared_local1[((k_inner_inner3 + 24))]));
    }
  }
  for (int ax08 = 0; ax08 < 2; ++ax08) {
    for (int ax14 = 0; ax14 < 6; ++ax14) {
      placeholder_shared_local1[(((ax08 * 6) + ax14))] = placeholder_shared[(((((((int)threadIdx.x) * 36) + (ax08 * 18)) + ax14) + 294))];
      placeholder_shared_local1[((((ax08 * 6) + ax14) + 12))] = placeholder_shared[(((((((int)threadIdx.x) * 36) + (ax08 * 18)) + ax14) + 438))];
    }
  }
  for (int ax09 = 0; ax09 < 6; ++ax09) {
    placeholder_d_shared_local1[(ax09)] = placeholder_d_shared[((((ax09 * 20) + ((int)threadIdx.y)) + 480))];
    placeholder_d_shared_local1[((ax09 + 6))] = placeholder_d_shared[((((ax09 * 20) + ((int)threadIdx.y)) + 484))];
    placeholder_d_shared_local1[((ax09 + 12))] = placeholder_d_shared[((((ax09 * 20) + ((int)threadIdx.y)) + 488))];
    placeholder_d_shared_local1[((ax09 + 18))] = placeholder_d_shared[((((ax09 * 20) + ((int)threadIdx.y)) + 492))];
    placeholder_d_shared_local1[((ax09 + 24))] = placeholder_d_shared[((((ax09 * 20) + ((int)threadIdx.y)) + 496))];
  }
  for (int k_inner_inner4 = 0; k_inner_inner4 < 6; ++k_inner_inner4) {
    for (int i_c4 = 0; i_c4 < 2; ++i_c4) {
      T_matmul_NN_local[(i_c4)] = (T_matmul_NN_local[(i_c4)] + (placeholder_shared_local1[(((i_c4 * 6) + k_inner_inner4))] * placeholder_d_shared_local1[(k_inner_inner4)]));
      T_matmul_NN_local[((i_c4 + 4))] = (T_matmul_NN_local[((i_c4 + 4))] + (placeholder_shared_local1[(((i_c4 * 6) + k_inner_inner4))] * placeholder_d_shared_local1[((k_inner_inner4 + 6))]));
      T_matmul_NN_local[((i_c4 + 8))] = (T_matmul_NN_local[((i_c4 + 8))] + (placeholder_shared_local1[(((i_c4 * 6) + k_inner_inner4))] * placeholder_d_shared_local1[((k_inner_inner4 + 12))]));
      T_matmul_NN_local[((i_c4 + 12))] = (T_matmul_NN_local[((i_c4 + 12))] + (placeholder_shared_local1[(((i_c4 * 6) + k_inner_inner4))] * placeholder_d_shared_local1[((k_inner_inner4 + 18))]));
      T_matmul_NN_local[((i_c4 + 16))] = (T_matmul_NN_local[((i_c4 + 16))] + (placeholder_shared_local1[(((i_c4 * 6) + k_inner_inner4))] * placeholder_d_shared_local1[((k_inner_inner4 + 24))]));
      T_matmul_NN_local[((i_c4 + 2))] = (T_matmul_NN_local[((i_c4 + 2))] + (placeholder_shared_local1[((((i_c4 * 6) + k_inner_inner4) + 12))] * placeholder_d_shared_local1[(k_inner_inner4)]));
      T_matmul_NN_local[((i_c4 + 6))] = (T_matmul_NN_local[((i_c4 + 6))] + (placeholder_shared_local1[((((i_c4 * 6) + k_inner_inner4) + 12))] * placeholder_d_shared_local1[((k_inner_inner4 + 6))]));
      T_matmul_NN_local[((i_c4 + 10))] = (T_matmul_NN_local[((i_c4 + 10))] + (placeholder_shared_local1[((((i_c4 * 6) + k_inner_inner4) + 12))] * placeholder_d_shared_local1[((k_inner_inner4 + 12))]));
      T_matmul_NN_local[((i_c4 + 14))] = (T_matmul_NN_local[((i_c4 + 14))] + (placeholder_shared_local1[((((i_c4 * 6) + k_inner_inner4) + 12))] * placeholder_d_shared_local1[((k_inner_inner4 + 18))]));
      T_matmul_NN_local[((i_c4 + 18))] = (T_matmul_NN_local[((i_c4 + 18))] + (placeholder_shared_local1[((((i_c4 * 6) + k_inner_inner4) + 12))] * placeholder_d_shared_local1[((k_inner_inner4 + 24))]));
    }
  }
  for (int ax010 = 0; ax010 < 2; ++ax010) {
    for (int ax15 = 0; ax15 < 6; ++ax15) {
      placeholder_shared_local1[(((ax010 * 6) + ax15))] = placeholder_shared[(((((((int)threadIdx.x) * 36) + (ax010 * 18)) + ax15) + 300))];
      placeholder_shared_local1[((((ax010 * 6) + ax15) + 12))] = placeholder_shared[(((((((int)threadIdx.x) * 36) + (ax010 * 18)) + ax15) + 444))];
    }
  }
  for (int ax011 = 0; ax011 < 6; ++ax011) {
    placeholder_d_shared_local1[(ax011)] = placeholder_d_shared[((((ax011 * 20) + ((int)threadIdx.y)) + 600))];
    placeholder_d_shared_local1[((ax011 + 6))] = placeholder_d_shared[((((ax011 * 20) + ((int)threadIdx.y)) + 604))];
    placeholder_d_shared_local1[((ax011 + 12))] = placeholder_d_shared[((((ax011 * 20) + ((int)threadIdx.y)) + 608))];
    placeholder_d_shared_local1[((ax011 + 18))] = placeholder_d_shared[((((ax011 * 20) + ((int)threadIdx.y)) + 612))];
    placeholder_d_shared_local1[((ax011 + 24))] = placeholder_d_shared[((((ax011 * 20) + ((int)threadIdx.y)) + 616))];
  }
  for (int k_inner_inner5 = 0; k_inner_inner5 < 6; ++k_inner_inner5) {
    for (int i_c5 = 0; i_c5 < 2; ++i_c5) {
      T_matmul_NN_local[(i_c5)] = (T_matmul_NN_local[(i_c5)] + (placeholder_shared_local1[(((i_c5 * 6) + k_inner_inner5))] * placeholder_d_shared_local1[(k_inner_inner5)]));
      T_matmul_NN_local[((i_c5 + 4))] = (T_matmul_NN_local[((i_c5 + 4))] + (placeholder_shared_local1[(((i_c5 * 6) + k_inner_inner5))] * placeholder_d_shared_local1[((k_inner_inner5 + 6))]));
      T_matmul_NN_local[((i_c5 + 8))] = (T_matmul_NN_local[((i_c5 + 8))] + (placeholder_shared_local1[(((i_c5 * 6) + k_inner_inner5))] * placeholder_d_shared_local1[((k_inner_inner5 + 12))]));
      T_matmul_NN_local[((i_c5 + 12))] = (T_matmul_NN_local[((i_c5 + 12))] + (placeholder_shared_local1[(((i_c5 * 6) + k_inner_inner5))] * placeholder_d_shared_local1[((k_inner_inner5 + 18))]));
      T_matmul_NN_local[((i_c5 + 16))] = (T_matmul_NN_local[((i_c5 + 16))] + (placeholder_shared_local1[(((i_c5 * 6) + k_inner_inner5))] * placeholder_d_shared_local1[((k_inner_inner5 + 24))]));
      T_matmul_NN_local[((i_c5 + 2))] = (T_matmul_NN_local[((i_c5 + 2))] + (placeholder_shared_local1[((((i_c5 * 6) + k_inner_inner5) + 12))] * placeholder_d_shared_local1[(k_inner_inner5)]));
      T_matmul_NN_local[((i_c5 + 6))] = (T_matmul_NN_local[((i_c5 + 6))] + (placeholder_shared_local1[((((i_c5 * 6) + k_inner_inner5) + 12))] * placeholder_d_shared_local1[((k_inner_inner5 + 6))]));
      T_matmul_NN_local[((i_c5 + 10))] = (T_matmul_NN_local[((i_c5 + 10))] + (placeholder_shared_local1[((((i_c5 * 6) + k_inner_inner5) + 12))] * placeholder_d_shared_local1[((k_inner_inner5 + 12))]));
      T_matmul_NN_local[((i_c5 + 14))] = (T_matmul_NN_local[((i_c5 + 14))] + (placeholder_shared_local1[((((i_c5 * 6) + k_inner_inner5) + 12))] * placeholder_d_shared_local1[((k_inner_inner5 + 18))]));
      T_matmul_NN_local[((i_c5 + 18))] = (T_matmul_NN_local[((i_c5 + 18))] + (placeholder_shared_local1[((((i_c5 * 6) + k_inner_inner5) + 12))] * placeholder_d_shared_local1[((k_inner_inner5 + 24))]));
    }
  }
  for (int i_inner_inner_inner = 0; i_inner_inner_inner < 2; ++i_inner_inner_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)))] = T_matmul_NN_local[(i_inner_inner_inner)];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)) + 4))] = T_matmul_NN_local[((i_inner_inner_inner + 4))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)) + 8))] = T_matmul_NN_local[((i_inner_inner_inner + 8))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)) + 12))] = T_matmul_NN_local[((i_inner_inner_inner + 12))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)) + 16))] = T_matmul_NN_local[((i_inner_inner_inner + 16))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)) + 8000))] = T_matmul_NN_local[((i_inner_inner_inner + 2))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)) + 8004))] = T_matmul_NN_local[((i_inner_inner_inner + 6))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)) + 8008))] = T_matmul_NN_local[((i_inner_inner_inner + 10))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)) + 8012))] = T_matmul_NN_local[((i_inner_inner_inner + 14))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16000) + (((int)threadIdx.x) * 2000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 20)) + ((int)threadIdx.y)) + 8016))] = T_matmul_NN_local[((i_inner_inner_inner + 18))];
  }
}

