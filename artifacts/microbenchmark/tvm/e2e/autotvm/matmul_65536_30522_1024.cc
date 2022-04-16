//1024_4_1_16_16_1
//65536_30522_1024
//dim3 grid(1024, 4, 1);
//dim3 block(16, 16, 1);

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
extern "C" __global__ void __launch_bounds__(256) matmul_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[64];
  __shared__ float placeholder_shared[768];
  __shared__ float placeholder_d_shared[3072];
  float placeholder_shared_local[24];
  float placeholder_d_shared_local[96];
  float placeholder_shared_local1[24];
  float placeholder_d_shared_local1[96];
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
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 6) {
        placeholder_shared[(((((((int)threadIdx.y) * 24) + (ax0_inner * 6)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 1953408) + (((int)threadIdx.y) * 122088)) + (ax0_inner * 30522)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
    for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
      if (((int)threadIdx.y) < 6) {
        placeholder_d_shared[(((((((int)threadIdx.y) * 256) + (ax1_outer * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[((((((((int)threadIdx.y) * 1024) + (((int)blockIdx.y) * 256)) + (ax1_outer * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 5086; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner1 = 0; ax0_inner1 < 4; ++ax0_inner1) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 6) {
          if ((((k_outer_outer * 6) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) < 30516) {
            placeholder_shared[((((((((k_outer_outer + 1) & 1) * 384) + (((int)threadIdx.y) * 24)) + (ax0_inner1 * 6)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[((((((((((int)blockIdx.x) * 1953408) + (((int)threadIdx.y) * 122088)) + (ax0_inner1 * 30522)) + (k_outer_outer * 6)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) + 6))];
          }
        }
      }
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 4; ++ax1_outer1) {
      for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
        if (((int)threadIdx.y) < 6) {
          placeholder_d_shared[((((((((k_outer_outer + 1) & 1) * 1536) + (((int)threadIdx.y) * 256)) + (ax1_outer1 * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[((((((((k_outer_outer * 6144) + (((int)threadIdx.y) * 1024)) + (((int)blockIdx.y) * 256)) + (ax1_outer1 * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 6144))];
        }
      }
    }
    for (int ax1 = 0; ax1 < 6; ++ax1) {
      placeholder_shared_local[(ax1)] = placeholder_shared[(((((k_outer_outer & 1) * 384) + (((int)threadIdx.x) * 6)) + ax1))];
      placeholder_shared_local[((ax1 + 6))] = placeholder_shared[((((((k_outer_outer & 1) * 384) + (((int)threadIdx.x) * 6)) + ax1) + 96))];
      placeholder_shared_local[((ax1 + 12))] = placeholder_shared[((((((k_outer_outer & 1) * 384) + (((int)threadIdx.x) * 6)) + ax1) + 192))];
      placeholder_shared_local[((ax1 + 18))] = placeholder_shared[((((((k_outer_outer & 1) * 384) + (((int)threadIdx.x) * 6)) + ax1) + 288))];
    }
    for (int ax0 = 0; ax0 < 6; ++ax0) {
      for (int ax11 = 0; ax11 < 4; ++ax11) {
        placeholder_d_shared_local[(((ax0 * 4) + ax11))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1536) + (ax0 * 256)) + (((int)threadIdx.y) * 4)) + ax11))];
        placeholder_d_shared_local[((((ax0 * 4) + ax11) + 24))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 1536) + (ax0 * 256)) + (((int)threadIdx.y) * 4)) + ax11) + 64))];
        placeholder_d_shared_local[((((ax0 * 4) + ax11) + 48))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 1536) + (ax0 * 256)) + (((int)threadIdx.y) * 4)) + ax11) + 128))];
        placeholder_d_shared_local[((((ax0 * 4) + ax11) + 72))] = placeholder_d_shared[(((((((k_outer_outer & 1) * 1536) + (ax0 * 256)) + (((int)threadIdx.y) * 4)) + ax11) + 192))];
      }
    }
    for (int k_inner_inner = 0; k_inner_inner < 6; ++k_inner_inner) {
      for (int j_c = 0; j_c < 4; ++j_c) {
        T_matmul_NN_local[(j_c)] = (T_matmul_NN_local[(j_c)] + (placeholder_shared_local[(k_inner_inner)] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
        T_matmul_NN_local[((j_c + 16))] = (T_matmul_NN_local[((j_c + 16))] + (placeholder_shared_local[(k_inner_inner)] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 24))]));
        T_matmul_NN_local[((j_c + 32))] = (T_matmul_NN_local[((j_c + 32))] + (placeholder_shared_local[(k_inner_inner)] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 48))]));
        T_matmul_NN_local[((j_c + 48))] = (T_matmul_NN_local[((j_c + 48))] + (placeholder_shared_local[(k_inner_inner)] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 72))]));
        T_matmul_NN_local[((j_c + 4))] = (T_matmul_NN_local[((j_c + 4))] + (placeholder_shared_local[((k_inner_inner + 6))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
        T_matmul_NN_local[((j_c + 20))] = (T_matmul_NN_local[((j_c + 20))] + (placeholder_shared_local[((k_inner_inner + 6))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 24))]));
        T_matmul_NN_local[((j_c + 36))] = (T_matmul_NN_local[((j_c + 36))] + (placeholder_shared_local[((k_inner_inner + 6))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 48))]));
        T_matmul_NN_local[((j_c + 52))] = (T_matmul_NN_local[((j_c + 52))] + (placeholder_shared_local[((k_inner_inner + 6))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 72))]));
        T_matmul_NN_local[((j_c + 8))] = (T_matmul_NN_local[((j_c + 8))] + (placeholder_shared_local[((k_inner_inner + 12))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
        T_matmul_NN_local[((j_c + 24))] = (T_matmul_NN_local[((j_c + 24))] + (placeholder_shared_local[((k_inner_inner + 12))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 24))]));
        T_matmul_NN_local[((j_c + 40))] = (T_matmul_NN_local[((j_c + 40))] + (placeholder_shared_local[((k_inner_inner + 12))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 48))]));
        T_matmul_NN_local[((j_c + 56))] = (T_matmul_NN_local[((j_c + 56))] + (placeholder_shared_local[((k_inner_inner + 12))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 72))]));
        T_matmul_NN_local[((j_c + 12))] = (T_matmul_NN_local[((j_c + 12))] + (placeholder_shared_local[((k_inner_inner + 18))] * placeholder_d_shared_local[(((k_inner_inner * 4) + j_c))]));
        T_matmul_NN_local[((j_c + 28))] = (T_matmul_NN_local[((j_c + 28))] + (placeholder_shared_local[((k_inner_inner + 18))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 24))]));
        T_matmul_NN_local[((j_c + 44))] = (T_matmul_NN_local[((j_c + 44))] + (placeholder_shared_local[((k_inner_inner + 18))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 48))]));
        T_matmul_NN_local[((j_c + 60))] = (T_matmul_NN_local[((j_c + 60))] + (placeholder_shared_local[((k_inner_inner + 18))] * placeholder_d_shared_local[((((k_inner_inner * 4) + j_c) + 72))]));
      }
    }
  }
  __syncthreads();
  for (int ax12 = 0; ax12 < 6; ++ax12) {
    placeholder_shared_local1[(ax12)] = placeholder_shared[(((((int)threadIdx.x) * 6) + ax12))];
    placeholder_shared_local1[((ax12 + 6))] = placeholder_shared[((((((int)threadIdx.x) * 6) + ax12) + 96))];
    placeholder_shared_local1[((ax12 + 12))] = placeholder_shared[((((((int)threadIdx.x) * 6) + ax12) + 192))];
    placeholder_shared_local1[((ax12 + 18))] = placeholder_shared[((((((int)threadIdx.x) * 6) + ax12) + 288))];
  }
  for (int ax01 = 0; ax01 < 6; ++ax01) {
    for (int ax13 = 0; ax13 < 4; ++ax13) {
      placeholder_d_shared_local1[(((ax01 * 4) + ax13))] = placeholder_d_shared[((((ax01 * 256) + (((int)threadIdx.y) * 4)) + ax13))];
      placeholder_d_shared_local1[((((ax01 * 4) + ax13) + 24))] = placeholder_d_shared[(((((ax01 * 256) + (((int)threadIdx.y) * 4)) + ax13) + 64))];
      placeholder_d_shared_local1[((((ax01 * 4) + ax13) + 48))] = placeholder_d_shared[(((((ax01 * 256) + (((int)threadIdx.y) * 4)) + ax13) + 128))];
      placeholder_d_shared_local1[((((ax01 * 4) + ax13) + 72))] = placeholder_d_shared[(((((ax01 * 256) + (((int)threadIdx.y) * 4)) + ax13) + 192))];
    }
  }
  for (int k_inner_inner1 = 0; k_inner_inner1 < 6; ++k_inner_inner1) {
    for (int j_c1 = 0; j_c1 < 4; ++j_c1) {
      T_matmul_NN_local[(j_c1)] = (T_matmul_NN_local[(j_c1)] + (placeholder_shared_local1[(k_inner_inner1)] * placeholder_d_shared_local1[(((k_inner_inner1 * 4) + j_c1))]));
      T_matmul_NN_local[((j_c1 + 16))] = (T_matmul_NN_local[((j_c1 + 16))] + (placeholder_shared_local1[(k_inner_inner1)] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 24))]));
      T_matmul_NN_local[((j_c1 + 32))] = (T_matmul_NN_local[((j_c1 + 32))] + (placeholder_shared_local1[(k_inner_inner1)] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 48))]));
      T_matmul_NN_local[((j_c1 + 48))] = (T_matmul_NN_local[((j_c1 + 48))] + (placeholder_shared_local1[(k_inner_inner1)] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 72))]));
      T_matmul_NN_local[((j_c1 + 4))] = (T_matmul_NN_local[((j_c1 + 4))] + (placeholder_shared_local1[((k_inner_inner1 + 6))] * placeholder_d_shared_local1[(((k_inner_inner1 * 4) + j_c1))]));
      T_matmul_NN_local[((j_c1 + 20))] = (T_matmul_NN_local[((j_c1 + 20))] + (placeholder_shared_local1[((k_inner_inner1 + 6))] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 24))]));
      T_matmul_NN_local[((j_c1 + 36))] = (T_matmul_NN_local[((j_c1 + 36))] + (placeholder_shared_local1[((k_inner_inner1 + 6))] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 48))]));
      T_matmul_NN_local[((j_c1 + 52))] = (T_matmul_NN_local[((j_c1 + 52))] + (placeholder_shared_local1[((k_inner_inner1 + 6))] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 72))]));
      T_matmul_NN_local[((j_c1 + 8))] = (T_matmul_NN_local[((j_c1 + 8))] + (placeholder_shared_local1[((k_inner_inner1 + 12))] * placeholder_d_shared_local1[(((k_inner_inner1 * 4) + j_c1))]));
      T_matmul_NN_local[((j_c1 + 24))] = (T_matmul_NN_local[((j_c1 + 24))] + (placeholder_shared_local1[((k_inner_inner1 + 12))] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 24))]));
      T_matmul_NN_local[((j_c1 + 40))] = (T_matmul_NN_local[((j_c1 + 40))] + (placeholder_shared_local1[((k_inner_inner1 + 12))] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 48))]));
      T_matmul_NN_local[((j_c1 + 56))] = (T_matmul_NN_local[((j_c1 + 56))] + (placeholder_shared_local1[((k_inner_inner1 + 12))] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 72))]));
      T_matmul_NN_local[((j_c1 + 12))] = (T_matmul_NN_local[((j_c1 + 12))] + (placeholder_shared_local1[((k_inner_inner1 + 18))] * placeholder_d_shared_local1[(((k_inner_inner1 * 4) + j_c1))]));
      T_matmul_NN_local[((j_c1 + 28))] = (T_matmul_NN_local[((j_c1 + 28))] + (placeholder_shared_local1[((k_inner_inner1 + 18))] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 24))]));
      T_matmul_NN_local[((j_c1 + 44))] = (T_matmul_NN_local[((j_c1 + 44))] + (placeholder_shared_local1[((k_inner_inner1 + 18))] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 48))]));
      T_matmul_NN_local[((j_c1 + 60))] = (T_matmul_NN_local[((j_c1 + 60))] + (placeholder_shared_local1[((k_inner_inner1 + 18))] * placeholder_d_shared_local1[((((k_inner_inner1 * 4) + j_c1) + 72))]));
    }
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 4; ++j_inner_inner_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner))] = T_matmul_NN_local[(j_inner_inner_inner)];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 64))] = T_matmul_NN_local[((j_inner_inner_inner + 16))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 128))] = T_matmul_NN_local[((j_inner_inner_inner + 32))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 192))] = T_matmul_NN_local[((j_inner_inner_inner + 48))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16384))] = T_matmul_NN_local[((j_inner_inner_inner + 4))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16448))] = T_matmul_NN_local[((j_inner_inner_inner + 20))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16512))] = T_matmul_NN_local[((j_inner_inner_inner + 36))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16576))] = T_matmul_NN_local[((j_inner_inner_inner + 52))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 32768))] = T_matmul_NN_local[((j_inner_inner_inner + 8))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 32832))] = T_matmul_NN_local[((j_inner_inner_inner + 24))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 32896))] = T_matmul_NN_local[((j_inner_inner_inner + 40))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 32960))] = T_matmul_NN_local[((j_inner_inner_inner + 56))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 49152))] = T_matmul_NN_local[((j_inner_inner_inner + 12))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 49216))] = T_matmul_NN_local[((j_inner_inner_inner + 28))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 49280))] = T_matmul_NN_local[((j_inner_inner_inner + 44))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 49344))] = T_matmul_NN_local[((j_inner_inner_inner + 60))];
  }
}

