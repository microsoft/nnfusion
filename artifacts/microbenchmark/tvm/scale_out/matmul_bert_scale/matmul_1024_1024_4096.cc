//64_512_1_1_1_1
//1024_1024_4096
//dim3 grid(64, 512, 1);
//dim3 block(1, 1, 1);

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
extern "C" __global__ void matmul_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[128];
  __shared__ float placeholder_shared[128];
  __shared__ float placeholder_d_shared[64];
  float placeholder_shared_local[32];
  float placeholder_d_shared_local[16];
  float placeholder_shared_local1[32];
  float placeholder_d_shared_local1[16];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
      T_matmul_NN_local[(((i_c_init * 4) + j_c_init))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 64))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 8))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 72))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 16))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 80))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 24))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 88))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 32))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 96))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 40))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 104))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 48))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 112))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 56))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 120))] = 0.000000e+00f;
    }
  }
  for (int ax0_inner = 0; ax0_inner < 16; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      placeholder_shared[(((ax0_inner * 4) + ax1_inner_inner))] = placeholder[((((((int)blockIdx.x) * 16384) + (ax0_inner * 1024)) + ax1_inner_inner))];
    }
  }
  for (int ax0_inner1 = 0; ax0_inner1 < 4; ++ax0_inner1) {
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
        placeholder_d_shared[((((ax0_inner1 * 8) + (ax1_outer * 4)) + ax1_inner_inner1))] = placeholder1[(((((ax0_inner1 * 4096) + (((int)blockIdx.y) * 8)) + (ax1_outer * 4)) + ax1_inner_inner1))];
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 255; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner2 = 0; ax0_inner2 < 16; ++ax0_inner2) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        placeholder_shared[((((((k_outer_outer + 1) & 1) * 64) + (ax0_inner2 * 4)) + ax1_inner_inner2))] = placeholder[((((((((int)blockIdx.x) * 16384) + (ax0_inner2 * 1024)) + (k_outer_outer * 4)) + ax1_inner_inner2) + 4))];
      }
    }
    for (int ax0_inner3 = 0; ax0_inner3 < 4; ++ax0_inner3) {
      for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
        for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
          placeholder_d_shared[(((((((k_outer_outer + 1) & 1) * 32) + (ax0_inner3 * 8)) + (ax1_outer1 * 4)) + ax1_inner_inner3))] = placeholder1[(((((((k_outer_outer * 16384) + (ax0_inner3 * 4096)) + (((int)blockIdx.y) * 8)) + (ax1_outer1 * 4)) + ax1_inner_inner3) + 16384))];
        }
      }
    }
    for (int ax0 = 0; ax0 < 2; ++ax0) {
      for (int ax1 = 0; ax1 < 2; ++ax1) {
        placeholder_shared_local[(((ax0 * 2) + ax1))] = placeholder_shared[(((((k_outer_outer & 1) * 64) + (ax0 * 4)) + ax1))];
        placeholder_shared_local[((((ax0 * 2) + ax1) + 4))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax0 * 4)) + ax1) + 8))];
        placeholder_shared_local[((((ax0 * 2) + ax1) + 8))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax0 * 4)) + ax1) + 16))];
        placeholder_shared_local[((((ax0 * 2) + ax1) + 12))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax0 * 4)) + ax1) + 24))];
        placeholder_shared_local[((((ax0 * 2) + ax1) + 16))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax0 * 4)) + ax1) + 32))];
        placeholder_shared_local[((((ax0 * 2) + ax1) + 20))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax0 * 4)) + ax1) + 40))];
        placeholder_shared_local[((((ax0 * 2) + ax1) + 24))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax0 * 4)) + ax1) + 48))];
        placeholder_shared_local[((((ax0 * 2) + ax1) + 28))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax0 * 4)) + ax1) + 56))];
      }
    }
    for (int ax01 = 0; ax01 < 2; ++ax01) {
      for (int ax11 = 0; ax11 < 4; ++ax11) {
        placeholder_d_shared_local[(((ax01 * 4) + ax11))] = placeholder_d_shared[(((((k_outer_outer & 1) * 32) + (ax01 * 8)) + ax11))];
        placeholder_d_shared_local[((((ax01 * 4) + ax11) + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 32) + (ax01 * 8)) + ax11) + 4))];
      }
    }
    for (int k_inner_inner = 0; k_inner_inner < 2; ++k_inner_inner) {
      for (int i_c = 0; i_c < 2; ++i_c) {
        for (int j_c = 0; j_c < 4; ++j_c) {
          T_matmul_NN_local[(((i_c * 4) + j_c))] = (T_matmul_NN_local[(((i_c * 4) + j_c))] + (placeholder_shared_local[(((i_c * 2) + k_inner_inner))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 64))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 64))] + (placeholder_shared_local[(((i_c * 2) + k_inner_inner))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 8))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 8))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 4))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 72))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 72))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 4))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 16))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 16))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 8))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 80))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 80))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 8))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 24))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 24))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 12))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 88))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 88))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 12))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 32))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 32))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 16))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 96))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 96))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 16))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 40))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 40))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 20))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 104))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 104))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 20))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 48))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 48))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 24))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 112))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 112))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 24))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 56))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 56))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 28))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
          T_matmul_NN_local[((((i_c * 4) + j_c) + 120))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 120))] + (placeholder_shared_local[((((i_c * 2) + k_inner_inner) + 28))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
        }
      }
    }
    for (int ax02 = 0; ax02 < 2; ++ax02) {
      for (int ax12 = 0; ax12 < 2; ++ax12) {
        placeholder_shared_local[(((ax02 * 2) + ax12))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax02 * 4)) + ax12) + 2))];
        placeholder_shared_local[((((ax02 * 2) + ax12) + 4))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax02 * 4)) + ax12) + 10))];
        placeholder_shared_local[((((ax02 * 2) + ax12) + 8))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax02 * 4)) + ax12) + 18))];
        placeholder_shared_local[((((ax02 * 2) + ax12) + 12))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax02 * 4)) + ax12) + 26))];
        placeholder_shared_local[((((ax02 * 2) + ax12) + 16))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax02 * 4)) + ax12) + 34))];
        placeholder_shared_local[((((ax02 * 2) + ax12) + 20))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax02 * 4)) + ax12) + 42))];
        placeholder_shared_local[((((ax02 * 2) + ax12) + 24))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax02 * 4)) + ax12) + 50))];
        placeholder_shared_local[((((ax02 * 2) + ax12) + 28))] = placeholder_shared[((((((k_outer_outer & 1) * 64) + (ax02 * 4)) + ax12) + 58))];
      }
    }
    for (int ax03 = 0; ax03 < 2; ++ax03) {
      for (int ax13 = 0; ax13 < 4; ++ax13) {
        placeholder_d_shared_local[(((ax03 * 4) + ax13))] = placeholder_d_shared[((((((k_outer_outer & 1) * 32) + (ax03 * 8)) + ax13) + 16))];
        placeholder_d_shared_local[((((ax03 * 4) + ax13) + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 32) + (ax03 * 8)) + ax13) + 20))];
      }
    }
    for (int k_inner_inner1 = 0; k_inner_inner1 < 2; ++k_inner_inner1) {
      for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
        for (int j_c1 = 0; j_c1 < 4; ++j_c1) {
          T_matmul_NN_local[(((i_c1 * 4) + j_c1))] = (T_matmul_NN_local[(((i_c1 * 4) + j_c1))] + (placeholder_shared_local[(((i_c1 * 2) + k_inner_inner1))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 64))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 64))] + (placeholder_shared_local[(((i_c1 * 2) + k_inner_inner1))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 8))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 8))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 4))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 72))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 72))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 4))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 16))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 16))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 8))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 80))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 80))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 8))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 24))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 24))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 12))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 88))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 88))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 12))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 32))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 32))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 16))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 96))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 96))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 16))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 40))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 40))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 20))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 104))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 104))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 20))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 48))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 48))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 24))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 112))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 112))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 24))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 56))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 56))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 28))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
          T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 120))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 120))] + (placeholder_shared_local[((((i_c1 * 2) + k_inner_inner1) + 28))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
        }
      }
    }
  }
  __syncthreads();
  for (int ax04 = 0; ax04 < 2; ++ax04) {
    for (int ax14 = 0; ax14 < 2; ++ax14) {
      placeholder_shared_local1[(((ax04 * 2) + ax14))] = placeholder_shared[((((ax04 * 4) + ax14) + 64))];
      placeholder_shared_local1[((((ax04 * 2) + ax14) + 4))] = placeholder_shared[((((ax04 * 4) + ax14) + 72))];
      placeholder_shared_local1[((((ax04 * 2) + ax14) + 8))] = placeholder_shared[((((ax04 * 4) + ax14) + 80))];
      placeholder_shared_local1[((((ax04 * 2) + ax14) + 12))] = placeholder_shared[((((ax04 * 4) + ax14) + 88))];
      placeholder_shared_local1[((((ax04 * 2) + ax14) + 16))] = placeholder_shared[((((ax04 * 4) + ax14) + 96))];
      placeholder_shared_local1[((((ax04 * 2) + ax14) + 20))] = placeholder_shared[((((ax04 * 4) + ax14) + 104))];
      placeholder_shared_local1[((((ax04 * 2) + ax14) + 24))] = placeholder_shared[((((ax04 * 4) + ax14) + 112))];
      placeholder_shared_local1[((((ax04 * 2) + ax14) + 28))] = placeholder_shared[((((ax04 * 4) + ax14) + 120))];
    }
  }
  for (int ax05 = 0; ax05 < 2; ++ax05) {
    for (int ax15 = 0; ax15 < 4; ++ax15) {
      placeholder_d_shared_local1[(((ax05 * 4) + ax15))] = placeholder_d_shared[((((ax05 * 8) + ax15) + 32))];
      placeholder_d_shared_local1[((((ax05 * 4) + ax15) + 8))] = placeholder_d_shared[((((ax05 * 8) + ax15) + 36))];
    }
  }
  for (int k_inner_inner2 = 0; k_inner_inner2 < 2; ++k_inner_inner2) {
    for (int i_c2 = 0; i_c2 < 2; ++i_c2) {
      for (int j_c2 = 0; j_c2 < 4; ++j_c2) {
        T_matmul_NN_local[(((i_c2 * 4) + j_c2))] = (T_matmul_NN_local[(((i_c2 * 4) + j_c2))] + (placeholder_shared_local1[(((i_c2 * 2) + k_inner_inner2))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 64))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 64))] + (placeholder_shared_local1[(((i_c2 * 2) + k_inner_inner2))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 8))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 8))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 4))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 72))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 72))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 4))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 16))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 16))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 8))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 80))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 80))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 8))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 24))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 24))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 12))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 88))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 88))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 12))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 32))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 32))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 16))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 96))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 96))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 16))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 40))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 40))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 20))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 104))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 104))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 20))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 48))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 48))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 24))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 112))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 112))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 24))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 56))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 56))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 28))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
        T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 120))] = (T_matmul_NN_local[((((i_c2 * 4) + j_c2) + 120))] + (placeholder_shared_local1[((((i_c2 * 2) + k_inner_inner2) + 28))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
      }
    }
  }
  for (int ax06 = 0; ax06 < 2; ++ax06) {
    for (int ax16 = 0; ax16 < 2; ++ax16) {
      placeholder_shared_local1[(((ax06 * 2) + ax16))] = placeholder_shared[((((ax06 * 4) + ax16) + 66))];
      placeholder_shared_local1[((((ax06 * 2) + ax16) + 4))] = placeholder_shared[((((ax06 * 4) + ax16) + 74))];
      placeholder_shared_local1[((((ax06 * 2) + ax16) + 8))] = placeholder_shared[((((ax06 * 4) + ax16) + 82))];
      placeholder_shared_local1[((((ax06 * 2) + ax16) + 12))] = placeholder_shared[((((ax06 * 4) + ax16) + 90))];
      placeholder_shared_local1[((((ax06 * 2) + ax16) + 16))] = placeholder_shared[((((ax06 * 4) + ax16) + 98))];
      placeholder_shared_local1[((((ax06 * 2) + ax16) + 20))] = placeholder_shared[((((ax06 * 4) + ax16) + 106))];
      placeholder_shared_local1[((((ax06 * 2) + ax16) + 24))] = placeholder_shared[((((ax06 * 4) + ax16) + 114))];
      placeholder_shared_local1[((((ax06 * 2) + ax16) + 28))] = placeholder_shared[((((ax06 * 4) + ax16) + 122))];
    }
  }
  for (int ax07 = 0; ax07 < 2; ++ax07) {
    for (int ax17 = 0; ax17 < 4; ++ax17) {
      placeholder_d_shared_local1[(((ax07 * 4) + ax17))] = placeholder_d_shared[((((ax07 * 8) + ax17) + 48))];
      placeholder_d_shared_local1[((((ax07 * 4) + ax17) + 8))] = placeholder_d_shared[((((ax07 * 8) + ax17) + 52))];
    }
  }
  for (int k_inner_inner3 = 0; k_inner_inner3 < 2; ++k_inner_inner3) {
    for (int i_c3 = 0; i_c3 < 2; ++i_c3) {
      for (int j_c3 = 0; j_c3 < 4; ++j_c3) {
        T_matmul_NN_local[(((i_c3 * 4) + j_c3))] = (T_matmul_NN_local[(((i_c3 * 4) + j_c3))] + (placeholder_shared_local1[(((i_c3 * 2) + k_inner_inner3))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 64))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 64))] + (placeholder_shared_local1[(((i_c3 * 2) + k_inner_inner3))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 8))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 8))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 4))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 72))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 72))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 4))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 16))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 16))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 8))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 80))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 80))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 8))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 24))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 24))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 12))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 88))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 88))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 12))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 32))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 32))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 16))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 96))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 96))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 16))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 40))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 40))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 20))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 104))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 104))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 20))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 48))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 48))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 24))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 112))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 112))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 24))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 56))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 56))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 28))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
        T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 120))] = (T_matmul_NN_local[((((i_c3 * 4) + j_c3) + 120))] + (placeholder_shared_local1[((((i_c3 * 2) + k_inner_inner3) + 28))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
      }
    }
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 4; ++j_inner_inner_inner) {
    for (int i_inner_inner_inner = 0; i_inner_inner_inner < 2; ++i_inner_inner_inner) {
      T_matmul_NN[(((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner))] = T_matmul_NN_local[(((i_inner_inner_inner * 4) + j_inner_inner_inner))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 4))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 64))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 8192))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 8))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 8196))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 72))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 16384))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 16))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 16388))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 80))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 24576))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 24))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 24580))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 88))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 32768))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 32))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 32772))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 96))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 40960))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 40))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 40964))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 104))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 49152))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 48))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 49156))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 112))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 57344))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 56))];
      T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 8)) + j_inner_inner_inner) + 57348))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 120))];
    }
  }
}

