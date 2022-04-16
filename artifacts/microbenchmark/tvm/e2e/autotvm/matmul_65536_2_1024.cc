//4096_16_1_2_2_1
//65536_2_1024
//dim3 grid(4096, 16, 1);
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
  float T_matmul_NN_local[256];
  __shared__ float placeholder_shared[32];
  __shared__ float placeholder_d_shared[128];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[32];
  float placeholder_shared_local1[8];
  float placeholder_d_shared_local1[32];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
      T_matmul_NN_local[(((i_c_init * 4) + j_c_init))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 32))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 64))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 96))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 128))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 160))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 192))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 224))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 16))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 48))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 80))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 112))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 144))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 176))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 208))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 240))] = 0.000000e+00f;
    }
  }
  for (int ax0_inner = 0; ax0_inner < 8; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 1) {
        placeholder_shared[(((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + ax0_inner) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + (ax0_inner * 2)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax1_outer = 0; ax1_outer < 8; ++ax1_outer) {
    for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
      if (((int)threadIdx.y) < 1) {
        placeholder_d_shared[(((((((int)threadIdx.y) * 64) + (ax1_outer * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[((((((((int)threadIdx.y) * 1024) + (((int)blockIdx.y) * 64)) + (ax1_outer * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 1; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner1 = 0; ax0_inner1 < 8; ++ax0_inner1) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 1) {
          placeholder_shared[((((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + ax0_inner1) + ax1_inner_inner2) + 16))] = placeholder[(((((((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + (ax0_inner1 * 2)) + ax1_inner_inner2) + 1))];
        }
      }
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 8; ++ax1_outer1) {
      for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
        if (((int)threadIdx.y) < 1) {
          placeholder_d_shared[((((((((int)threadIdx.y) * 64) + (ax1_outer1 * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 64))] = placeholder1[(((((((((int)threadIdx.y) * 1024) + (((int)blockIdx.y) * 64)) + (ax1_outer1 * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 1024))];
        }
      }
    }
    for (int ax0 = 0; ax0 < 4; ++ax0) {
      placeholder_shared_local[(ax0)] = placeholder_shared[(((((int)threadIdx.x) * 4) + ax0))];
      placeholder_shared_local[((ax0 + 4))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax0) + 8))];
    }
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      placeholder_d_shared_local[(ax1)] = placeholder_d_shared[(((((int)threadIdx.y) * 4) + ax1))];
      placeholder_d_shared_local[((ax1 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax1) + 8))];
      placeholder_d_shared_local[((ax1 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax1) + 16))];
      placeholder_d_shared_local[((ax1 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax1) + 24))];
      placeholder_d_shared_local[((ax1 + 16))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax1) + 32))];
      placeholder_d_shared_local[((ax1 + 20))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax1) + 40))];
      placeholder_d_shared_local[((ax1 + 24))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax1) + 48))];
      placeholder_d_shared_local[((ax1 + 28))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax1) + 56))];
    }
    for (int i_c = 0; i_c < 4; ++i_c) {
      for (int j_c = 0; j_c < 4; ++j_c) {
        T_matmul_NN_local[(((i_c * 4) + j_c))] = (T_matmul_NN_local[(((i_c * 4) + j_c))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(j_c)]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 32))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 32))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[((j_c + 4))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 64))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 64))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[((j_c + 8))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 96))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 96))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[((j_c + 12))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 128))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 128))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[((j_c + 16))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 160))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 160))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[((j_c + 20))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 192))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 192))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[((j_c + 24))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 224))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 224))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[((j_c + 28))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 16))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 16))] + (placeholder_shared_local[((i_c + 4))] * placeholder_d_shared_local[(j_c)]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 48))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 48))] + (placeholder_shared_local[((i_c + 4))] * placeholder_d_shared_local[((j_c + 4))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 80))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 80))] + (placeholder_shared_local[((i_c + 4))] * placeholder_d_shared_local[((j_c + 8))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 112))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 112))] + (placeholder_shared_local[((i_c + 4))] * placeholder_d_shared_local[((j_c + 12))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 144))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 144))] + (placeholder_shared_local[((i_c + 4))] * placeholder_d_shared_local[((j_c + 16))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 176))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 176))] + (placeholder_shared_local[((i_c + 4))] * placeholder_d_shared_local[((j_c + 20))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 208))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 208))] + (placeholder_shared_local[((i_c + 4))] * placeholder_d_shared_local[((j_c + 24))]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 240))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 240))] + (placeholder_shared_local[((i_c + 4))] * placeholder_d_shared_local[((j_c + 28))]));
      }
    }
  }
  __syncthreads();
  for (int ax01 = 0; ax01 < 4; ++ax01) {
    placeholder_shared_local1[(ax01)] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax01) + 16))];
    placeholder_shared_local1[((ax01 + 4))] = placeholder_shared[((((((int)threadIdx.x) * 4) + ax01) + 24))];
  }
  for (int ax11 = 0; ax11 < 4; ++ax11) {
    placeholder_d_shared_local1[(ax11)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax11) + 64))];
    placeholder_d_shared_local1[((ax11 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax11) + 72))];
    placeholder_d_shared_local1[((ax11 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax11) + 80))];
    placeholder_d_shared_local1[((ax11 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax11) + 88))];
    placeholder_d_shared_local1[((ax11 + 16))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax11) + 96))];
    placeholder_d_shared_local1[((ax11 + 20))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax11) + 104))];
    placeholder_d_shared_local1[((ax11 + 24))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax11) + 112))];
    placeholder_d_shared_local1[((ax11 + 28))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax11) + 120))];
  }
  for (int i_c1 = 0; i_c1 < 4; ++i_c1) {
    for (int j_c1 = 0; j_c1 < 4; ++j_c1) {
      T_matmul_NN_local[(((i_c1 * 4) + j_c1))] = (T_matmul_NN_local[(((i_c1 * 4) + j_c1))] + (placeholder_shared_local1[(i_c1)] * placeholder_d_shared_local1[(j_c1)]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 32))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 32))] + (placeholder_shared_local1[(i_c1)] * placeholder_d_shared_local1[((j_c1 + 4))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 64))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 64))] + (placeholder_shared_local1[(i_c1)] * placeholder_d_shared_local1[((j_c1 + 8))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 96))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 96))] + (placeholder_shared_local1[(i_c1)] * placeholder_d_shared_local1[((j_c1 + 12))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 128))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 128))] + (placeholder_shared_local1[(i_c1)] * placeholder_d_shared_local1[((j_c1 + 16))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 160))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 160))] + (placeholder_shared_local1[(i_c1)] * placeholder_d_shared_local1[((j_c1 + 20))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 192))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 192))] + (placeholder_shared_local1[(i_c1)] * placeholder_d_shared_local1[((j_c1 + 24))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 224))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 224))] + (placeholder_shared_local1[(i_c1)] * placeholder_d_shared_local1[((j_c1 + 28))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 16))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 16))] + (placeholder_shared_local1[((i_c1 + 4))] * placeholder_d_shared_local1[(j_c1)]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 48))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 48))] + (placeholder_shared_local1[((i_c1 + 4))] * placeholder_d_shared_local1[((j_c1 + 4))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 80))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 80))] + (placeholder_shared_local1[((i_c1 + 4))] * placeholder_d_shared_local1[((j_c1 + 8))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 112))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 112))] + (placeholder_shared_local1[((i_c1 + 4))] * placeholder_d_shared_local1[((j_c1 + 12))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 144))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 144))] + (placeholder_shared_local1[((i_c1 + 4))] * placeholder_d_shared_local1[((j_c1 + 16))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 176))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 176))] + (placeholder_shared_local1[((i_c1 + 4))] * placeholder_d_shared_local1[((j_c1 + 20))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 208))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 208))] + (placeholder_shared_local1[((i_c1 + 4))] * placeholder_d_shared_local1[((j_c1 + 24))]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 240))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 240))] + (placeholder_shared_local1[((i_c1 + 4))] * placeholder_d_shared_local1[((j_c1 + 28))]));
    }
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 4; ++j_inner_inner_inner) {
    for (int i_inner_inner_inner = 0; i_inner_inner_inner < 4; ++i_inner_inner_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner))] = T_matmul_NN_local[(((i_inner_inner_inner * 4) + j_inner_inner_inner))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 32))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 16))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 64))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 24))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 96))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 32))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 128))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 40))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 160))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 48))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 192))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 56))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 224))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8192))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 16))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8200))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 48))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8208))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 80))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8216))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 112))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8224))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 144))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8232))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 176))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8240))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 208))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 8248))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 240))];
    }
  }
}

