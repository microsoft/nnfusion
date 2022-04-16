//2_25_1_4_4_1
//128_2048_1000
//dim3 grid(2, 25, 1);
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
  float T_matmul_NN_local[160];
  __shared__ float placeholder_shared[4096];
  __shared__ float placeholder_d_shared[2560];
  float placeholder_shared_local[512];
  float placeholder_d_shared_local[320];
  float placeholder_shared_local1[512];
  float placeholder_d_shared_local1[320];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      T_matmul_NN_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 32))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 64))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 96))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 128))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 16))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 48))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 80))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 112))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 144))] = 0.000000e+00f;
    }
  }
  for (int ax0_inner = 0; ax0_inner < 16; ++ax0_inner) {
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
        placeholder_shared[((((((((int)threadIdx.y) * 512) + (ax0_inner * 32)) + (ax1_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[(((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.y) * 32768)) + (ax0_inner * 2048)) + (ax1_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax0_inner1 = 0; ax0_inner1 < 8; ++ax0_inner1) {
    for (int ax1_outer1 = 0; ax1_outer1 < 3; ++ax1_outer1) {
      for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
        if ((((ax1_outer1 * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1) < 40) {
          if (((((((int)blockIdx.y) * 40) + (ax1_outer1 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1) < 1000) {
            placeholder_d_shared[((((((((int)threadIdx.y) * 320) + (ax0_inner1 * 40)) + (ax1_outer1 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[(((((((((int)threadIdx.y) * 8000) + (ax0_inner1 * 1000)) + (((int)blockIdx.y) * 40)) + (ax1_outer1 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
          }
        }
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 63; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner2 = 0; ax0_inner2 < 16; ++ax0_inner2) {
      for (int ax1_outer2 = 0; ax1_outer2 < 2; ++ax1_outer2) {
        for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
          placeholder_shared[(((((((((k_outer_outer + 1) & 1) * 2048) + (((int)threadIdx.y) * 512)) + (ax0_inner2 * 32)) + (ax1_outer2 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[(((((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.y) * 32768)) + (ax0_inner2 * 2048)) + (k_outer_outer * 32)) + (ax1_outer2 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) + 32))];
        }
      }
    }
    for (int ax0_inner3 = 0; ax0_inner3 < 8; ++ax0_inner3) {
      for (int ax1_outer3 = 0; ax1_outer3 < 3; ++ax1_outer3) {
        for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
          if ((((ax1_outer3 * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) < 40) {
            if (((((((int)blockIdx.y) * 40) + (ax1_outer3 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) < 1000) {
              placeholder_d_shared[(((((((((k_outer_outer + 1) & 1) * 1280) + (((int)threadIdx.y) * 320)) + (ax0_inner3 * 40)) + (ax1_outer3 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[(((((((((k_outer_outer * 32000) + (((int)threadIdx.y) * 8000)) + (ax0_inner3 * 1000)) + (((int)blockIdx.y) * 40)) + (ax1_outer3 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 32000))];
            }
          }
        }
      }
    }
    for (int ax0 = 0; ax0 < 8; ++ax0) {
      for (int ax1 = 0; ax1 < 32; ++ax1) {
        placeholder_shared_local[(((ax0 * 32) + ax1))] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.x) * 256)) + (ax0 * 32)) + ax1))];
        placeholder_shared_local[((((ax0 * 32) + ax1) + 256))] = placeholder_shared[(((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.x) * 256)) + (ax0 * 32)) + ax1) + 1024))];
      }
    }
    for (int ax01 = 0; ax01 < 32; ++ax01) {
      for (int ax11 = 0; ax11 < 2; ++ax11) {
        placeholder_d_shared_local[(((ax01 * 2) + ax11))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1280) + (ax01 * 40)) + (((int)threadIdx.y) * 2)) + ax11))];
        placeholder_d_shared_local[((((ax01 * 2) + ax11) + 64))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 1280) + (ax01 * 40)) + (((int)threadIdx.y) * 2)) + ax11) + 8))];
        placeholder_d_shared_local[((((ax01 * 2) + ax11) + 128))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 1280) + (ax01 * 40)) + (((int)threadIdx.y) * 2)) + ax11) + 16))];
        placeholder_d_shared_local[((((ax01 * 2) + ax11) + 192))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 1280) + (ax01 * 40)) + (((int)threadIdx.y) * 2)) + ax11) + 24))];
        placeholder_d_shared_local[((((ax01 * 2) + ax11) + 256))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 1280) + (ax01 * 40)) + (((int)threadIdx.y) * 2)) + ax11) + 32))];
      }
    }
    for (int k_inner_inner = 0; k_inner_inner < 32; ++k_inner_inner) {
      for (int i_c = 0; i_c < 8; ++i_c) {
        for (int j_c = 0; j_c < 2; ++j_c) {
          T_matmul_NN_local[(((i_c * 2) + j_c))] = (T_matmul_NN_local[(((i_c * 2) + j_c))] + (placeholder_shared_local[(((i_c * 32) + k_inner_inner))] * placeholder_d_shared_local[(((k_inner_inner * 2) + j_c))]));
          T_matmul_NN_local[((((i_c * 2) + j_c) + 32))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 32))] + (placeholder_shared_local[(((i_c * 32) + k_inner_inner))] * placeholder_d_shared_local[((((k_inner_inner * 2) + j_c) + 64))]));
          T_matmul_NN_local[((((i_c * 2) + j_c) + 64))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 64))] + (placeholder_shared_local[(((i_c * 32) + k_inner_inner))] * placeholder_d_shared_local[((((k_inner_inner * 2) + j_c) + 128))]));
          T_matmul_NN_local[((((i_c * 2) + j_c) + 96))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 96))] + (placeholder_shared_local[(((i_c * 32) + k_inner_inner))] * placeholder_d_shared_local[((((k_inner_inner * 2) + j_c) + 192))]));
          T_matmul_NN_local[((((i_c * 2) + j_c) + 128))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 128))] + (placeholder_shared_local[(((i_c * 32) + k_inner_inner))] * placeholder_d_shared_local[((((k_inner_inner * 2) + j_c) + 256))]));
          T_matmul_NN_local[((((i_c * 2) + j_c) + 16))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 16))] + (placeholder_shared_local[((((i_c * 32) + k_inner_inner) + 256))] * placeholder_d_shared_local[(((k_inner_inner * 2) + j_c))]));
          T_matmul_NN_local[((((i_c * 2) + j_c) + 48))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 48))] + (placeholder_shared_local[((((i_c * 32) + k_inner_inner) + 256))] * placeholder_d_shared_local[((((k_inner_inner * 2) + j_c) + 64))]));
          T_matmul_NN_local[((((i_c * 2) + j_c) + 80))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 80))] + (placeholder_shared_local[((((i_c * 32) + k_inner_inner) + 256))] * placeholder_d_shared_local[((((k_inner_inner * 2) + j_c) + 128))]));
          T_matmul_NN_local[((((i_c * 2) + j_c) + 112))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 112))] + (placeholder_shared_local[((((i_c * 32) + k_inner_inner) + 256))] * placeholder_d_shared_local[((((k_inner_inner * 2) + j_c) + 192))]));
          T_matmul_NN_local[((((i_c * 2) + j_c) + 144))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 144))] + (placeholder_shared_local[((((i_c * 32) + k_inner_inner) + 256))] * placeholder_d_shared_local[((((k_inner_inner * 2) + j_c) + 256))]));
        }
      }
    }
  }
  __syncthreads();
  for (int ax02 = 0; ax02 < 8; ++ax02) {
    for (int ax12 = 0; ax12 < 32; ++ax12) {
      placeholder_shared_local1[(((ax02 * 32) + ax12))] = placeholder_shared[(((((((int)threadIdx.x) * 256) + (ax02 * 32)) + ax12) + 2048))];
      placeholder_shared_local1[((((ax02 * 32) + ax12) + 256))] = placeholder_shared[(((((((int)threadIdx.x) * 256) + (ax02 * 32)) + ax12) + 3072))];
    }
  }
  for (int ax03 = 0; ax03 < 32; ++ax03) {
    for (int ax13 = 0; ax13 < 2; ++ax13) {
      placeholder_d_shared_local1[(((ax03 * 2) + ax13))] = placeholder_d_shared[(((((ax03 * 40) + (((int)threadIdx.y) * 2)) + ax13) + 1280))];
      placeholder_d_shared_local1[((((ax03 * 2) + ax13) + 64))] = placeholder_d_shared[(((((ax03 * 40) + (((int)threadIdx.y) * 2)) + ax13) + 1288))];
      placeholder_d_shared_local1[((((ax03 * 2) + ax13) + 128))] = placeholder_d_shared[(((((ax03 * 40) + (((int)threadIdx.y) * 2)) + ax13) + 1296))];
      placeholder_d_shared_local1[((((ax03 * 2) + ax13) + 192))] = placeholder_d_shared[(((((ax03 * 40) + (((int)threadIdx.y) * 2)) + ax13) + 1304))];
      placeholder_d_shared_local1[((((ax03 * 2) + ax13) + 256))] = placeholder_d_shared[(((((ax03 * 40) + (((int)threadIdx.y) * 2)) + ax13) + 1312))];
    }
  }
  for (int k_inner_inner1 = 0; k_inner_inner1 < 32; ++k_inner_inner1) {
    for (int i_c1 = 0; i_c1 < 8; ++i_c1) {
      for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
        T_matmul_NN_local[(((i_c1 * 2) + j_c1))] = (T_matmul_NN_local[(((i_c1 * 2) + j_c1))] + (placeholder_shared_local1[(((i_c1 * 32) + k_inner_inner1))] * placeholder_d_shared_local1[(((k_inner_inner1 * 2) + j_c1))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 32))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 32))] + (placeholder_shared_local1[(((i_c1 * 32) + k_inner_inner1))] * placeholder_d_shared_local1[((((k_inner_inner1 * 2) + j_c1) + 64))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 64))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 64))] + (placeholder_shared_local1[(((i_c1 * 32) + k_inner_inner1))] * placeholder_d_shared_local1[((((k_inner_inner1 * 2) + j_c1) + 128))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 96))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 96))] + (placeholder_shared_local1[(((i_c1 * 32) + k_inner_inner1))] * placeholder_d_shared_local1[((((k_inner_inner1 * 2) + j_c1) + 192))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 128))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 128))] + (placeholder_shared_local1[(((i_c1 * 32) + k_inner_inner1))] * placeholder_d_shared_local1[((((k_inner_inner1 * 2) + j_c1) + 256))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 16))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 16))] + (placeholder_shared_local1[((((i_c1 * 32) + k_inner_inner1) + 256))] * placeholder_d_shared_local1[(((k_inner_inner1 * 2) + j_c1))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 48))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 48))] + (placeholder_shared_local1[((((i_c1 * 32) + k_inner_inner1) + 256))] * placeholder_d_shared_local1[((((k_inner_inner1 * 2) + j_c1) + 64))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 80))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 80))] + (placeholder_shared_local1[((((i_c1 * 32) + k_inner_inner1) + 256))] * placeholder_d_shared_local1[((((k_inner_inner1 * 2) + j_c1) + 128))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 112))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 112))] + (placeholder_shared_local1[((((i_c1 * 32) + k_inner_inner1) + 256))] * placeholder_d_shared_local1[((((k_inner_inner1 * 2) + j_c1) + 192))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 144))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 144))] + (placeholder_shared_local1[((((i_c1 * 32) + k_inner_inner1) + 256))] * placeholder_d_shared_local1[((((k_inner_inner1 * 2) + j_c1) + 256))]));
      }
    }
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 2; ++j_inner_inner_inner) {
    for (int i_inner_inner_inner = 0; i_inner_inner_inner < 8; ++i_inner_inner_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner))] = T_matmul_NN_local[(((i_inner_inner_inner * 2) + j_inner_inner_inner))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 8))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 32))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 16))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 64))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 24))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 96))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 128))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32000))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 16))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32008))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 48))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32016))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 80))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32024))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 112))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 64000) + (((int)threadIdx.x) * 8000)) + (i_inner_inner_inner * 1000)) + (((int)blockIdx.y) * 40)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32032))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 144))];
    }
  }
}

