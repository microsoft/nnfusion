//8192_32_1_2_2_1
//65536_4096_1024
//dim3 grid(8192, 32, 1);
//dim3 block(2, 2, 1);

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
extern "C" __global__ void __launch_bounds__(4) matmul_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[64];
  __shared__ float placeholder_shared[64];
  __shared__ float placeholder_d_shared[256];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[32];
  float placeholder_shared_local1[8];
  float placeholder_d_shared_local1[32];
  for (int j_c_init = 0; j_c_init < 16; ++j_c_init) {
    T_matmul_NN_local[(j_c_init)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 16))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 32))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 48))] = 0.000000e+00f;
  }
  for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 4) {
        placeholder_shared[(((((((int)threadIdx.y) * 16) + (ax0_inner * 4)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (ax0_inner * 4096)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
    for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
      for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
        placeholder_d_shared[((((((((int)threadIdx.y) * 64) + (ax0_inner1 * 32)) + (ax1_outer * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[(((((((((int)threadIdx.y) * 2048) + (ax0_inner1 * 1024)) + (((int)blockIdx.y) * 32)) + (ax1_outer * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 1023; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner2 = 0; ax0_inner2 < 4; ++ax0_inner2) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 4) {
          if ((((((int)threadIdx.x) * 4) + (k_outer_outer * 4)) + ax1_inner_inner2) < 4092) {
            placeholder_shared[((((((((k_outer_outer + 1) & 1) * 32) + (((int)threadIdx.y) * 16)) + (ax0_inner2 * 4)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[((((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + (ax0_inner2 * 4096)) + (((int)threadIdx.x) * 4)) + (k_outer_outer * 4)) + ax1_inner_inner2) + 4))];
          }
        }
      }
    }
    for (int ax0_inner3 = 0; ax0_inner3 < 2; ++ax0_inner3) {
      for (int ax1_outer1 = 0; ax1_outer1 < 4; ++ax1_outer1) {
        for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
          placeholder_d_shared[(((((((((k_outer_outer + 1) & 1) * 128) + (((int)threadIdx.y) * 64)) + (ax0_inner3 * 32)) + (ax1_outer1 * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[(((((((((k_outer_outer * 4096) + (((int)threadIdx.y) * 2048)) + (ax0_inner3 * 1024)) + (((int)blockIdx.y) * 32)) + (ax1_outer1 * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 4096))];
        }
      }
    }
    for (int ax1 = 0; ax1 < 2; ++ax1) {
      placeholder_shared_local[(ax1)] = placeholder_shared[(((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax1))];
      placeholder_shared_local[((ax1 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax1) + 8))];
      placeholder_shared_local[((ax1 + 4))] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax1) + 16))];
      placeholder_shared_local[((ax1 + 6))] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax1) + 24))];
    }
    for (int ax0 = 0; ax0 < 2; ++ax0) {
      for (int ax11 = 0; ax11 < 16; ++ax11) {
        placeholder_d_shared_local[(((ax0 * 16) + ax11))] = placeholder_d_shared[((((((k_outer_outer & 1) * 128) + (ax0 * 32)) + (((int)threadIdx.y) * 16)) + ax11))];
      }
    }
    for (int k_inner_inner = 0; k_inner_inner < 2; ++k_inner_inner) {
      for (int j_c = 0; j_c < 16; ++j_c) {
        T_matmul_NN_local[(j_c)] = (T_matmul_NN_local[(j_c)] + (placeholder_shared_local[(k_inner_inner)] * placeholder_d_shared_local[(((k_inner_inner * 16) + j_c))]));
        T_matmul_NN_local[((j_c + 16))] = (T_matmul_NN_local[((j_c + 16))] + (placeholder_shared_local[((k_inner_inner + 2))] * placeholder_d_shared_local[(((k_inner_inner * 16) + j_c))]));
        T_matmul_NN_local[((j_c + 32))] = (T_matmul_NN_local[((j_c + 32))] + (placeholder_shared_local[((k_inner_inner + 4))] * placeholder_d_shared_local[(((k_inner_inner * 16) + j_c))]));
        T_matmul_NN_local[((j_c + 48))] = (T_matmul_NN_local[((j_c + 48))] + (placeholder_shared_local[((k_inner_inner + 6))] * placeholder_d_shared_local[(((k_inner_inner * 16) + j_c))]));
      }
    }
    for (int ax12 = 0; ax12 < 2; ++ax12) {
      placeholder_shared_local[(ax12)] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax12) + 2))];
      placeholder_shared_local[((ax12 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax12) + 10))];
      placeholder_shared_local[((ax12 + 4))] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax12) + 18))];
      placeholder_shared_local[((ax12 + 6))] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax12) + 26))];
    }
    for (int ax01 = 0; ax01 < 2; ++ax01) {
      for (int ax13 = 0; ax13 < 16; ++ax13) {
        placeholder_d_shared_local[(((ax01 * 16) + ax13))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 128) + (ax01 * 32)) + (((int)threadIdx.y) * 16)) + ax13) + 64))];
      }
    }
    for (int k_inner_inner1 = 0; k_inner_inner1 < 2; ++k_inner_inner1) {
      for (int j_c1 = 0; j_c1 < 16; ++j_c1) {
        T_matmul_NN_local[(j_c1)] = (T_matmul_NN_local[(j_c1)] + (placeholder_shared_local[(k_inner_inner1)] * placeholder_d_shared_local[(((k_inner_inner1 * 16) + j_c1))]));
        T_matmul_NN_local[((j_c1 + 16))] = (T_matmul_NN_local[((j_c1 + 16))] + (placeholder_shared_local[((k_inner_inner1 + 2))] * placeholder_d_shared_local[(((k_inner_inner1 * 16) + j_c1))]));
        T_matmul_NN_local[((j_c1 + 32))] = (T_matmul_NN_local[((j_c1 + 32))] + (placeholder_shared_local[((k_inner_inner1 + 4))] * placeholder_d_shared_local[(((k_inner_inner1 * 16) + j_c1))]));
        T_matmul_NN_local[((j_c1 + 48))] = (T_matmul_NN_local[((j_c1 + 48))] + (placeholder_shared_local[((k_inner_inner1 + 6))] * placeholder_d_shared_local[(((k_inner_inner1 * 16) + j_c1))]));
      }
    }
  }
  __syncthreads();
  for (int ax14 = 0; ax14 < 2; ++ax14) {
    placeholder_shared_local1[(ax14)] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax14) + 32))];
    placeholder_shared_local1[((ax14 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax14) + 40))];
    placeholder_shared_local1[((ax14 + 4))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax14) + 48))];
    placeholder_shared_local1[((ax14 + 6))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax14) + 56))];
  }
  for (int ax02 = 0; ax02 < 2; ++ax02) {
    for (int ax15 = 0; ax15 < 16; ++ax15) {
      placeholder_d_shared_local1[(((ax02 * 16) + ax15))] = placeholder_d_shared[(((((ax02 * 32) + (((int)threadIdx.y) * 16)) + ax15) + 128))];
    }
  }
  for (int k_inner_inner2 = 0; k_inner_inner2 < 2; ++k_inner_inner2) {
    for (int j_c2 = 0; j_c2 < 16; ++j_c2) {
      T_matmul_NN_local[(j_c2)] = (T_matmul_NN_local[(j_c2)] + (placeholder_shared_local1[(k_inner_inner2)] * placeholder_d_shared_local1[(((k_inner_inner2 * 16) + j_c2))]));
      T_matmul_NN_local[((j_c2 + 16))] = (T_matmul_NN_local[((j_c2 + 16))] + (placeholder_shared_local1[((k_inner_inner2 + 2))] * placeholder_d_shared_local1[(((k_inner_inner2 * 16) + j_c2))]));
      T_matmul_NN_local[((j_c2 + 32))] = (T_matmul_NN_local[((j_c2 + 32))] + (placeholder_shared_local1[((k_inner_inner2 + 4))] * placeholder_d_shared_local1[(((k_inner_inner2 * 16) + j_c2))]));
      T_matmul_NN_local[((j_c2 + 48))] = (T_matmul_NN_local[((j_c2 + 48))] + (placeholder_shared_local1[((k_inner_inner2 + 6))] * placeholder_d_shared_local1[(((k_inner_inner2 * 16) + j_c2))]));
    }
  }
  for (int ax16 = 0; ax16 < 2; ++ax16) {
    placeholder_shared_local1[(ax16)] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax16) + 34))];
    placeholder_shared_local1[((ax16 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax16) + 42))];
    placeholder_shared_local1[((ax16 + 4))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax16) + 50))];
    placeholder_shared_local1[((ax16 + 6))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax16) + 58))];
  }
  for (int ax03 = 0; ax03 < 2; ++ax03) {
    for (int ax17 = 0; ax17 < 16; ++ax17) {
      placeholder_d_shared_local1[(((ax03 * 16) + ax17))] = placeholder_d_shared[(((((ax03 * 32) + (((int)threadIdx.y) * 16)) + ax17) + 192))];
    }
  }
  for (int k_inner_inner3 = 0; k_inner_inner3 < 2; ++k_inner_inner3) {
    for (int j_c3 = 0; j_c3 < 16; ++j_c3) {
      T_matmul_NN_local[(j_c3)] = (T_matmul_NN_local[(j_c3)] + (placeholder_shared_local1[(k_inner_inner3)] * placeholder_d_shared_local1[(((k_inner_inner3 * 16) + j_c3))]));
      T_matmul_NN_local[((j_c3 + 16))] = (T_matmul_NN_local[((j_c3 + 16))] + (placeholder_shared_local1[((k_inner_inner3 + 2))] * placeholder_d_shared_local1[(((k_inner_inner3 * 16) + j_c3))]));
      T_matmul_NN_local[((j_c3 + 32))] = (T_matmul_NN_local[((j_c3 + 32))] + (placeholder_shared_local1[((k_inner_inner3 + 4))] * placeholder_d_shared_local1[(((k_inner_inner3 * 16) + j_c3))]));
      T_matmul_NN_local[((j_c3 + 48))] = (T_matmul_NN_local[((j_c3 + 48))] + (placeholder_shared_local1[((k_inner_inner3 + 6))] * placeholder_d_shared_local1[(((k_inner_inner3 * 16) + j_c3))]));
    }
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 16; ++j_inner_inner_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + j_inner_inner_inner))] = T_matmul_NN_local[(j_inner_inner_inner)];
    T_matmul_NN[(((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + j_inner_inner_inner) + 2048))] = T_matmul_NN_local[((j_inner_inner_inner + 16))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + j_inner_inner_inner) + 4096))] = T_matmul_NN_local[((j_inner_inner_inner + 32))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 16)) + j_inner_inner_inner) + 6144))] = T_matmul_NN_local[((j_inner_inner_inner + 48))];
  }
}

