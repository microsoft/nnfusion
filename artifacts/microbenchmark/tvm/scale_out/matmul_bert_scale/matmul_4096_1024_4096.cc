//512_128_1_2_2_1
//4096_1024_4096
//dim3 grid(512, 128, 1);
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
  for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
    T_matmul_NN_local[(j_c_init)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 16))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 32))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 48))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 4))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 20))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 36))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 52))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 8))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 24))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 40))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 56))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 12))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 28))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 44))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 60))] = 0.000000e+00f;
  }
  for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 4) {
        placeholder_shared[(((((((int)threadIdx.y) * 16) + (ax0_inner * 4)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 4096)) + (ax0_inner * 1024)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
    for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
      for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
        placeholder_d_shared[((((((((int)threadIdx.y) * 64) + (ax0_inner1 * 32)) + (ax1_outer * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[(((((((((int)threadIdx.y) * 8192) + (ax0_inner1 * 4096)) + (((int)blockIdx.y) * 32)) + (ax1_outer * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 255; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner2 = 0; ax0_inner2 < 4; ++ax0_inner2) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 4) {
          if ((((((int)threadIdx.x) * 4) + (k_outer_outer * 4)) + ax1_inner_inner2) < 1020) {
            placeholder_shared[((((((((k_outer_outer + 1) & 1) * 32) + (((int)threadIdx.y) * 16)) + (ax0_inner2 * 4)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[((((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 4096)) + (ax0_inner2 * 1024)) + (((int)threadIdx.x) * 4)) + (k_outer_outer * 4)) + ax1_inner_inner2) + 4))];
          }
        }
      }
    }
    for (int ax0_inner3 = 0; ax0_inner3 < 2; ++ax0_inner3) {
      for (int ax1_outer1 = 0; ax1_outer1 < 4; ++ax1_outer1) {
        for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
          placeholder_d_shared[(((((((((k_outer_outer + 1) & 1) * 128) + (((int)threadIdx.y) * 64)) + (ax0_inner3 * 32)) + (ax1_outer1 * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[(((((((((k_outer_outer * 16384) + (((int)threadIdx.y) * 8192)) + (ax0_inner3 * 4096)) + (((int)blockIdx.y) * 32)) + (ax1_outer1 * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 16384))];
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
      for (int ax11 = 0; ax11 < 4; ++ax11) {
        placeholder_d_shared_local[(((ax0 * 4) + ax11))] = placeholder_d_shared[((((((k_outer_outer & 1) * 128) + (ax0 * 32)) + (((int)threadIdx.y) * 4)) + ax11))];
        placeholder_d_shared_local[((((ax0 * 4) + ax11) + 8))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 128) + (ax0 * 32)) + (((int)threadIdx.y) * 4)) + ax11) + 8))];
        placeholder_d_shared_local[((((ax0 * 4) + ax11) + 16))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 128) + (ax0 * 32)) + (((int)threadIdx.y) * 4)) + ax11) + 16))];
        placeholder_d_shared_local[((((ax0 * 4) + ax11) + 24))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 128) + (ax0 * 32)) + (((int)threadIdx.y) * 4)) + ax11) + 24))];
      }
    }
    for (int k_inner_inner = 0; k_inner_inner < 2; ++k_inner_inner) {
      for (int j_c = 0; j_c < 4; ++j_c) {
        T_matmul_NN_local[(j_c)] = (T_matmul_NN_local[(j_c)] + (placeholder_shared_local[(k_inner_inner)] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
        T_matmul_NN_local[((j_c + 16))] = (T_matmul_NN_local[((j_c + 16))] + (placeholder_shared_local[(k_inner_inner)] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
        T_matmul_NN_local[((j_c + 32))] = (T_matmul_NN_local[((j_c + 32))] + (placeholder_shared_local[(k_inner_inner)] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 16))]));
        T_matmul_NN_local[((j_c + 48))] = (T_matmul_NN_local[((j_c + 48))] + (placeholder_shared_local[(k_inner_inner)] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 24))]));
        T_matmul_NN_local[((j_c + 4))] = (T_matmul_NN_local[((j_c + 4))] + (placeholder_shared_local[((k_inner_inner + 2))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
        T_matmul_NN_local[((j_c + 20))] = (T_matmul_NN_local[((j_c + 20))] + (placeholder_shared_local[((k_inner_inner + 2))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
        T_matmul_NN_local[((j_c + 36))] = (T_matmul_NN_local[((j_c + 36))] + (placeholder_shared_local[((k_inner_inner + 2))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 16))]));
        T_matmul_NN_local[((j_c + 52))] = (T_matmul_NN_local[((j_c + 52))] + (placeholder_shared_local[((k_inner_inner + 2))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 24))]));
        T_matmul_NN_local[((j_c + 8))] = (T_matmul_NN_local[((j_c + 8))] + (placeholder_shared_local[((k_inner_inner + 4))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
        T_matmul_NN_local[((j_c + 24))] = (T_matmul_NN_local[((j_c + 24))] + (placeholder_shared_local[((k_inner_inner + 4))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
        T_matmul_NN_local[((j_c + 40))] = (T_matmul_NN_local[((j_c + 40))] + (placeholder_shared_local[((k_inner_inner + 4))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 16))]));
        T_matmul_NN_local[((j_c + 56))] = (T_matmul_NN_local[((j_c + 56))] + (placeholder_shared_local[((k_inner_inner + 4))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 24))]));
        T_matmul_NN_local[((j_c + 12))] = (T_matmul_NN_local[((j_c + 12))] + (placeholder_shared_local[((k_inner_inner + 6))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
        T_matmul_NN_local[((j_c + 28))] = (T_matmul_NN_local[((j_c + 28))] + (placeholder_shared_local[((k_inner_inner + 6))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 8))]));
        T_matmul_NN_local[((j_c + 44))] = (T_matmul_NN_local[((j_c + 44))] + (placeholder_shared_local[((k_inner_inner + 6))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 16))]));
        T_matmul_NN_local[((j_c + 60))] = (T_matmul_NN_local[((j_c + 60))] + (placeholder_shared_local[((k_inner_inner + 6))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 24))]));
      }
    }
    for (int ax12 = 0; ax12 < 2; ++ax12) {
      placeholder_shared_local[(ax12)] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax12) + 2))];
      placeholder_shared_local[((ax12 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax12) + 10))];
      placeholder_shared_local[((ax12 + 4))] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax12) + 18))];
      placeholder_shared_local[((ax12 + 6))] = placeholder_shared[((((((k_outer_outer & 1) * 32) + (((int)threadIdx.x) * 4)) + ax12) + 26))];
    }
    for (int ax01 = 0; ax01 < 2; ++ax01) {
      for (int ax13 = 0; ax13 < 4; ++ax13) {
        placeholder_d_shared_local[(((ax01 * 4) + ax13))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 128) + (ax01 * 32)) + (((int)threadIdx.y) * 4)) + ax13) + 64))];
        placeholder_d_shared_local[((((ax01 * 4) + ax13) + 8))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 128) + (ax01 * 32)) + (((int)threadIdx.y) * 4)) + ax13) + 72))];
        placeholder_d_shared_local[((((ax01 * 4) + ax13) + 16))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 128) + (ax01 * 32)) + (((int)threadIdx.y) * 4)) + ax13) + 80))];
        placeholder_d_shared_local[((((ax01 * 4) + ax13) + 24))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 128) + (ax01 * 32)) + (((int)threadIdx.y) * 4)) + ax13) + 88))];
      }
    }
    for (int k_inner_inner1 = 0; k_inner_inner1 < 2; ++k_inner_inner1) {
      for (int j_c1 = 0; j_c1 < 4; ++j_c1) {
        T_matmul_NN_local[(j_c1)] = (T_matmul_NN_local[(j_c1)] + (placeholder_shared_local[(k_inner_inner1)] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
        T_matmul_NN_local[((j_c1 + 16))] = (T_matmul_NN_local[((j_c1 + 16))] + (placeholder_shared_local[(k_inner_inner1)] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
        T_matmul_NN_local[((j_c1 + 32))] = (T_matmul_NN_local[((j_c1 + 32))] + (placeholder_shared_local[(k_inner_inner1)] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 16))]));
        T_matmul_NN_local[((j_c1 + 48))] = (T_matmul_NN_local[((j_c1 + 48))] + (placeholder_shared_local[(k_inner_inner1)] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 24))]));
        T_matmul_NN_local[((j_c1 + 4))] = (T_matmul_NN_local[((j_c1 + 4))] + (placeholder_shared_local[((k_inner_inner1 + 2))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
        T_matmul_NN_local[((j_c1 + 20))] = (T_matmul_NN_local[((j_c1 + 20))] + (placeholder_shared_local[((k_inner_inner1 + 2))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
        T_matmul_NN_local[((j_c1 + 36))] = (T_matmul_NN_local[((j_c1 + 36))] + (placeholder_shared_local[((k_inner_inner1 + 2))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 16))]));
        T_matmul_NN_local[((j_c1 + 52))] = (T_matmul_NN_local[((j_c1 + 52))] + (placeholder_shared_local[((k_inner_inner1 + 2))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 24))]));
        T_matmul_NN_local[((j_c1 + 8))] = (T_matmul_NN_local[((j_c1 + 8))] + (placeholder_shared_local[((k_inner_inner1 + 4))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
        T_matmul_NN_local[((j_c1 + 24))] = (T_matmul_NN_local[((j_c1 + 24))] + (placeholder_shared_local[((k_inner_inner1 + 4))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
        T_matmul_NN_local[((j_c1 + 40))] = (T_matmul_NN_local[((j_c1 + 40))] + (placeholder_shared_local[((k_inner_inner1 + 4))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 16))]));
        T_matmul_NN_local[((j_c1 + 56))] = (T_matmul_NN_local[((j_c1 + 56))] + (placeholder_shared_local[((k_inner_inner1 + 4))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 24))]));
        T_matmul_NN_local[((j_c1 + 12))] = (T_matmul_NN_local[((j_c1 + 12))] + (placeholder_shared_local[((k_inner_inner1 + 6))] * placeholder_d_shared_local[(((k_inner_inner1 * 4) + j_c1))]));
        T_matmul_NN_local[((j_c1 + 28))] = (T_matmul_NN_local[((j_c1 + 28))] + (placeholder_shared_local[((k_inner_inner1 + 6))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 8))]));
        T_matmul_NN_local[((j_c1 + 44))] = (T_matmul_NN_local[((j_c1 + 44))] + (placeholder_shared_local[((k_inner_inner1 + 6))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 16))]));
        T_matmul_NN_local[((j_c1 + 60))] = (T_matmul_NN_local[((j_c1 + 60))] + (placeholder_shared_local[((k_inner_inner1 + 6))] * placeholder_d_shared_local[((((k_inner_inner1 * 4) + j_c1) + 24))]));
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
    for (int ax15 = 0; ax15 < 4; ++ax15) {
      placeholder_d_shared_local1[(((ax02 * 4) + ax15))] = placeholder_d_shared[(((((ax02 * 32) + (((int)threadIdx.y) * 4)) + ax15) + 128))];
      placeholder_d_shared_local1[((((ax02 * 4) + ax15) + 8))] = placeholder_d_shared[(((((ax02 * 32) + (((int)threadIdx.y) * 4)) + ax15) + 136))];
      placeholder_d_shared_local1[((((ax02 * 4) + ax15) + 16))] = placeholder_d_shared[(((((ax02 * 32) + (((int)threadIdx.y) * 4)) + ax15) + 144))];
      placeholder_d_shared_local1[((((ax02 * 4) + ax15) + 24))] = placeholder_d_shared[(((((ax02 * 32) + (((int)threadIdx.y) * 4)) + ax15) + 152))];
    }
  }
  for (int k_inner_inner2 = 0; k_inner_inner2 < 2; ++k_inner_inner2) {
    for (int j_c2 = 0; j_c2 < 4; ++j_c2) {
      T_matmul_NN_local[(j_c2)] = (T_matmul_NN_local[(j_c2)] + (placeholder_shared_local1[(k_inner_inner2)] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
      T_matmul_NN_local[((j_c2 + 16))] = (T_matmul_NN_local[((j_c2 + 16))] + (placeholder_shared_local1[(k_inner_inner2)] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
      T_matmul_NN_local[((j_c2 + 32))] = (T_matmul_NN_local[((j_c2 + 32))] + (placeholder_shared_local1[(k_inner_inner2)] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 16))]));
      T_matmul_NN_local[((j_c2 + 48))] = (T_matmul_NN_local[((j_c2 + 48))] + (placeholder_shared_local1[(k_inner_inner2)] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 24))]));
      T_matmul_NN_local[((j_c2 + 4))] = (T_matmul_NN_local[((j_c2 + 4))] + (placeholder_shared_local1[((k_inner_inner2 + 2))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
      T_matmul_NN_local[((j_c2 + 20))] = (T_matmul_NN_local[((j_c2 + 20))] + (placeholder_shared_local1[((k_inner_inner2 + 2))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
      T_matmul_NN_local[((j_c2 + 36))] = (T_matmul_NN_local[((j_c2 + 36))] + (placeholder_shared_local1[((k_inner_inner2 + 2))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 16))]));
      T_matmul_NN_local[((j_c2 + 52))] = (T_matmul_NN_local[((j_c2 + 52))] + (placeholder_shared_local1[((k_inner_inner2 + 2))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 24))]));
      T_matmul_NN_local[((j_c2 + 8))] = (T_matmul_NN_local[((j_c2 + 8))] + (placeholder_shared_local1[((k_inner_inner2 + 4))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
      T_matmul_NN_local[((j_c2 + 24))] = (T_matmul_NN_local[((j_c2 + 24))] + (placeholder_shared_local1[((k_inner_inner2 + 4))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
      T_matmul_NN_local[((j_c2 + 40))] = (T_matmul_NN_local[((j_c2 + 40))] + (placeholder_shared_local1[((k_inner_inner2 + 4))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 16))]));
      T_matmul_NN_local[((j_c2 + 56))] = (T_matmul_NN_local[((j_c2 + 56))] + (placeholder_shared_local1[((k_inner_inner2 + 4))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 24))]));
      T_matmul_NN_local[((j_c2 + 12))] = (T_matmul_NN_local[((j_c2 + 12))] + (placeholder_shared_local1[((k_inner_inner2 + 6))] * placeholder_d_shared_local1[(((k_inner_inner2 * 4) + j_c2))]));
      T_matmul_NN_local[((j_c2 + 28))] = (T_matmul_NN_local[((j_c2 + 28))] + (placeholder_shared_local1[((k_inner_inner2 + 6))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 8))]));
      T_matmul_NN_local[((j_c2 + 44))] = (T_matmul_NN_local[((j_c2 + 44))] + (placeholder_shared_local1[((k_inner_inner2 + 6))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 16))]));
      T_matmul_NN_local[((j_c2 + 60))] = (T_matmul_NN_local[((j_c2 + 60))] + (placeholder_shared_local1[((k_inner_inner2 + 6))] * placeholder_d_shared_local1[((((k_inner_inner2 * 4) + j_c2) + 24))]));
    }
  }
  for (int ax16 = 0; ax16 < 2; ++ax16) {
    placeholder_shared_local1[(ax16)] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax16) + 34))];
    placeholder_shared_local1[((ax16 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax16) + 42))];
    placeholder_shared_local1[((ax16 + 4))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax16) + 50))];
    placeholder_shared_local1[((ax16 + 6))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax16) + 58))];
  }
  for (int ax03 = 0; ax03 < 2; ++ax03) {
    for (int ax17 = 0; ax17 < 4; ++ax17) {
      placeholder_d_shared_local1[(((ax03 * 4) + ax17))] = placeholder_d_shared[(((((ax03 * 32) + (((int)threadIdx.y) * 4)) + ax17) + 192))];
      placeholder_d_shared_local1[((((ax03 * 4) + ax17) + 8))] = placeholder_d_shared[(((((ax03 * 32) + (((int)threadIdx.y) * 4)) + ax17) + 200))];
      placeholder_d_shared_local1[((((ax03 * 4) + ax17) + 16))] = placeholder_d_shared[(((((ax03 * 32) + (((int)threadIdx.y) * 4)) + ax17) + 208))];
      placeholder_d_shared_local1[((((ax03 * 4) + ax17) + 24))] = placeholder_d_shared[(((((ax03 * 32) + (((int)threadIdx.y) * 4)) + ax17) + 216))];
    }
  }
  for (int k_inner_inner3 = 0; k_inner_inner3 < 2; ++k_inner_inner3) {
    for (int j_c3 = 0; j_c3 < 4; ++j_c3) {
      T_matmul_NN_local[(j_c3)] = (T_matmul_NN_local[(j_c3)] + (placeholder_shared_local1[(k_inner_inner3)] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
      T_matmul_NN_local[((j_c3 + 16))] = (T_matmul_NN_local[((j_c3 + 16))] + (placeholder_shared_local1[(k_inner_inner3)] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
      T_matmul_NN_local[((j_c3 + 32))] = (T_matmul_NN_local[((j_c3 + 32))] + (placeholder_shared_local1[(k_inner_inner3)] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 16))]));
      T_matmul_NN_local[((j_c3 + 48))] = (T_matmul_NN_local[((j_c3 + 48))] + (placeholder_shared_local1[(k_inner_inner3)] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 24))]));
      T_matmul_NN_local[((j_c3 + 4))] = (T_matmul_NN_local[((j_c3 + 4))] + (placeholder_shared_local1[((k_inner_inner3 + 2))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
      T_matmul_NN_local[((j_c3 + 20))] = (T_matmul_NN_local[((j_c3 + 20))] + (placeholder_shared_local1[((k_inner_inner3 + 2))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
      T_matmul_NN_local[((j_c3 + 36))] = (T_matmul_NN_local[((j_c3 + 36))] + (placeholder_shared_local1[((k_inner_inner3 + 2))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 16))]));
      T_matmul_NN_local[((j_c3 + 52))] = (T_matmul_NN_local[((j_c3 + 52))] + (placeholder_shared_local1[((k_inner_inner3 + 2))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 24))]));
      T_matmul_NN_local[((j_c3 + 8))] = (T_matmul_NN_local[((j_c3 + 8))] + (placeholder_shared_local1[((k_inner_inner3 + 4))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
      T_matmul_NN_local[((j_c3 + 24))] = (T_matmul_NN_local[((j_c3 + 24))] + (placeholder_shared_local1[((k_inner_inner3 + 4))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
      T_matmul_NN_local[((j_c3 + 40))] = (T_matmul_NN_local[((j_c3 + 40))] + (placeholder_shared_local1[((k_inner_inner3 + 4))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 16))]));
      T_matmul_NN_local[((j_c3 + 56))] = (T_matmul_NN_local[((j_c3 + 56))] + (placeholder_shared_local1[((k_inner_inner3 + 4))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 24))]));
      T_matmul_NN_local[((j_c3 + 12))] = (T_matmul_NN_local[((j_c3 + 12))] + (placeholder_shared_local1[((k_inner_inner3 + 6))] * placeholder_d_shared_local1[(((k_inner_inner3 * 4) + j_c3))]));
      T_matmul_NN_local[((j_c3 + 28))] = (T_matmul_NN_local[((j_c3 + 28))] + (placeholder_shared_local1[((k_inner_inner3 + 6))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 8))]));
      T_matmul_NN_local[((j_c3 + 44))] = (T_matmul_NN_local[((j_c3 + 44))] + (placeholder_shared_local1[((k_inner_inner3 + 6))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 16))]));
      T_matmul_NN_local[((j_c3 + 60))] = (T_matmul_NN_local[((j_c3 + 60))] + (placeholder_shared_local1[((k_inner_inner3 + 6))] * placeholder_d_shared_local1[((((k_inner_inner3 * 4) + j_c3) + 24))]));
    }
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 4; ++j_inner_inner_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner))] = T_matmul_NN_local[(j_inner_inner_inner)];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8))] = T_matmul_NN_local[((j_inner_inner_inner + 16))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16))] = T_matmul_NN_local[((j_inner_inner_inner + 32))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 24))] = T_matmul_NN_local[((j_inner_inner_inner + 48))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8192))] = T_matmul_NN_local[((j_inner_inner_inner + 4))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8200))] = T_matmul_NN_local[((j_inner_inner_inner + 20))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8208))] = T_matmul_NN_local[((j_inner_inner_inner + 36))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8216))] = T_matmul_NN_local[((j_inner_inner_inner + 52))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16384))] = T_matmul_NN_local[((j_inner_inner_inner + 8))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16392))] = T_matmul_NN_local[((j_inner_inner_inner + 24))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16400))] = T_matmul_NN_local[((j_inner_inner_inner + 40))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16408))] = T_matmul_NN_local[((j_inner_inner_inner + 56))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 24576))] = T_matmul_NN_local[((j_inner_inner_inner + 12))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 24584))] = T_matmul_NN_local[((j_inner_inner_inner + 28))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 24592))] = T_matmul_NN_local[((j_inner_inner_inner + 44))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 24600))] = T_matmul_NN_local[((j_inner_inner_inner + 60))];
  }
}

