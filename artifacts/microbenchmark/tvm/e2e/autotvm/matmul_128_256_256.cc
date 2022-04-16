//16_4_1_8_8_1
//128_256_256
//dim3 grid(16, 4, 1);
//dim3 block(8, 8, 1);

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
extern "C" __global__ void __launch_bounds__(64) matmul_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_matmul_NN) {
  float T_matmul_NN_local[8];
  __shared__ float placeholder_shared[256];
  __shared__ float placeholder_d_shared[2048];
  float placeholder_shared_local[1];
  float placeholder_d_shared_local[8];
  float placeholder_shared_local1[1];
  float placeholder_d_shared_local1[8];
  for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
    T_matmul_NN_local[(j_c_init)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 2))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 4))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 6))] = 0.000000e+00f;
  }
  for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
    if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 16) {
      placeholder_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[(((((((int)blockIdx.x) * 2048) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
    }
  }
  for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
        placeholder_d_shared[((((((((int)threadIdx.y) * 128) + (ax0_inner * 64)) + (ax1_outer * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[(((((((((int)threadIdx.y) * 512) + (ax0_inner * 256)) + (((int)blockIdx.y) * 64)) + (ax1_outer * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 15; ++k_outer_outer) {
    __syncthreads();
    for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 16) {
        if ((((k_outer_outer * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) < 240) {
          placeholder_shared[(((((((k_outer_outer + 1) & 1) * 128) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[(((((((((int)blockIdx.x) * 2048) + (((int)threadIdx.y) * 256)) + (k_outer_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) + 16))];
        }
      }
    }
    for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
      for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
        for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
          placeholder_d_shared[(((((((((k_outer_outer + 1) & 1) * 1024) + (((int)threadIdx.y) * 128)) + (ax0_inner1 * 64)) + (ax1_outer1 * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[(((((((((k_outer_outer * 4096) + (((int)threadIdx.y) * 512)) + (ax0_inner1 * 256)) + (((int)blockIdx.y) * 64)) + (ax1_outer1 * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 4096))];
        }
      }
    }
    placeholder_shared_local[(0)] = placeholder_shared[((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)))];
    for (int ax1 = 0; ax1 < 2; ++ax1) {
      placeholder_d_shared_local[(ax1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax1))];
      placeholder_d_shared_local[((ax1 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax1) + 16))];
      placeholder_d_shared_local[((ax1 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax1) + 32))];
      placeholder_d_shared_local[((ax1 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax1) + 48))];
    }
    for (int j_c = 0; j_c < 2; ++j_c) {
      T_matmul_NN_local[(j_c)] = (T_matmul_NN_local[(j_c)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c)]));
      T_matmul_NN_local[((j_c + 2))] = (T_matmul_NN_local[((j_c + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c + 2))]));
      T_matmul_NN_local[((j_c + 4))] = (T_matmul_NN_local[((j_c + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c + 4))]));
      T_matmul_NN_local[((j_c + 6))] = (T_matmul_NN_local[((j_c + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 1))];
    for (int ax11 = 0; ax11 < 2; ++ax11) {
      placeholder_d_shared_local[(ax11)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax11) + 64))];
      placeholder_d_shared_local[((ax11 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax11) + 80))];
      placeholder_d_shared_local[((ax11 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax11) + 96))];
      placeholder_d_shared_local[((ax11 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax11) + 112))];
    }
    for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
      T_matmul_NN_local[(j_c1)] = (T_matmul_NN_local[(j_c1)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c1)]));
      T_matmul_NN_local[((j_c1 + 2))] = (T_matmul_NN_local[((j_c1 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c1 + 2))]));
      T_matmul_NN_local[((j_c1 + 4))] = (T_matmul_NN_local[((j_c1 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c1 + 4))]));
      T_matmul_NN_local[((j_c1 + 6))] = (T_matmul_NN_local[((j_c1 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c1 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 2))];
    for (int ax12 = 0; ax12 < 2; ++ax12) {
      placeholder_d_shared_local[(ax12)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax12) + 128))];
      placeholder_d_shared_local[((ax12 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax12) + 144))];
      placeholder_d_shared_local[((ax12 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax12) + 160))];
      placeholder_d_shared_local[((ax12 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax12) + 176))];
    }
    for (int j_c2 = 0; j_c2 < 2; ++j_c2) {
      T_matmul_NN_local[(j_c2)] = (T_matmul_NN_local[(j_c2)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c2)]));
      T_matmul_NN_local[((j_c2 + 2))] = (T_matmul_NN_local[((j_c2 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c2 + 2))]));
      T_matmul_NN_local[((j_c2 + 4))] = (T_matmul_NN_local[((j_c2 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c2 + 4))]));
      T_matmul_NN_local[((j_c2 + 6))] = (T_matmul_NN_local[((j_c2 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c2 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 3))];
    for (int ax13 = 0; ax13 < 2; ++ax13) {
      placeholder_d_shared_local[(ax13)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax13) + 192))];
      placeholder_d_shared_local[((ax13 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax13) + 208))];
      placeholder_d_shared_local[((ax13 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax13) + 224))];
      placeholder_d_shared_local[((ax13 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax13) + 240))];
    }
    for (int j_c3 = 0; j_c3 < 2; ++j_c3) {
      T_matmul_NN_local[(j_c3)] = (T_matmul_NN_local[(j_c3)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c3)]));
      T_matmul_NN_local[((j_c3 + 2))] = (T_matmul_NN_local[((j_c3 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c3 + 2))]));
      T_matmul_NN_local[((j_c3 + 4))] = (T_matmul_NN_local[((j_c3 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c3 + 4))]));
      T_matmul_NN_local[((j_c3 + 6))] = (T_matmul_NN_local[((j_c3 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c3 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 4))];
    for (int ax14 = 0; ax14 < 2; ++ax14) {
      placeholder_d_shared_local[(ax14)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax14) + 256))];
      placeholder_d_shared_local[((ax14 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax14) + 272))];
      placeholder_d_shared_local[((ax14 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax14) + 288))];
      placeholder_d_shared_local[((ax14 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax14) + 304))];
    }
    for (int j_c4 = 0; j_c4 < 2; ++j_c4) {
      T_matmul_NN_local[(j_c4)] = (T_matmul_NN_local[(j_c4)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c4)]));
      T_matmul_NN_local[((j_c4 + 2))] = (T_matmul_NN_local[((j_c4 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c4 + 2))]));
      T_matmul_NN_local[((j_c4 + 4))] = (T_matmul_NN_local[((j_c4 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c4 + 4))]));
      T_matmul_NN_local[((j_c4 + 6))] = (T_matmul_NN_local[((j_c4 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c4 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 5))];
    for (int ax15 = 0; ax15 < 2; ++ax15) {
      placeholder_d_shared_local[(ax15)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax15) + 320))];
      placeholder_d_shared_local[((ax15 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax15) + 336))];
      placeholder_d_shared_local[((ax15 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax15) + 352))];
      placeholder_d_shared_local[((ax15 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax15) + 368))];
    }
    for (int j_c5 = 0; j_c5 < 2; ++j_c5) {
      T_matmul_NN_local[(j_c5)] = (T_matmul_NN_local[(j_c5)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c5)]));
      T_matmul_NN_local[((j_c5 + 2))] = (T_matmul_NN_local[((j_c5 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c5 + 2))]));
      T_matmul_NN_local[((j_c5 + 4))] = (T_matmul_NN_local[((j_c5 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c5 + 4))]));
      T_matmul_NN_local[((j_c5 + 6))] = (T_matmul_NN_local[((j_c5 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c5 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 6))];
    for (int ax16 = 0; ax16 < 2; ++ax16) {
      placeholder_d_shared_local[(ax16)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax16) + 384))];
      placeholder_d_shared_local[((ax16 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax16) + 400))];
      placeholder_d_shared_local[((ax16 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax16) + 416))];
      placeholder_d_shared_local[((ax16 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax16) + 432))];
    }
    for (int j_c6 = 0; j_c6 < 2; ++j_c6) {
      T_matmul_NN_local[(j_c6)] = (T_matmul_NN_local[(j_c6)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c6)]));
      T_matmul_NN_local[((j_c6 + 2))] = (T_matmul_NN_local[((j_c6 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c6 + 2))]));
      T_matmul_NN_local[((j_c6 + 4))] = (T_matmul_NN_local[((j_c6 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c6 + 4))]));
      T_matmul_NN_local[((j_c6 + 6))] = (T_matmul_NN_local[((j_c6 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c6 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 7))];
    for (int ax17 = 0; ax17 < 2; ++ax17) {
      placeholder_d_shared_local[(ax17)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax17) + 448))];
      placeholder_d_shared_local[((ax17 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax17) + 464))];
      placeholder_d_shared_local[((ax17 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax17) + 480))];
      placeholder_d_shared_local[((ax17 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax17) + 496))];
    }
    for (int j_c7 = 0; j_c7 < 2; ++j_c7) {
      T_matmul_NN_local[(j_c7)] = (T_matmul_NN_local[(j_c7)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c7)]));
      T_matmul_NN_local[((j_c7 + 2))] = (T_matmul_NN_local[((j_c7 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c7 + 2))]));
      T_matmul_NN_local[((j_c7 + 4))] = (T_matmul_NN_local[((j_c7 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c7 + 4))]));
      T_matmul_NN_local[((j_c7 + 6))] = (T_matmul_NN_local[((j_c7 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c7 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 8))];
    for (int ax18 = 0; ax18 < 2; ++ax18) {
      placeholder_d_shared_local[(ax18)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax18) + 512))];
      placeholder_d_shared_local[((ax18 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax18) + 528))];
      placeholder_d_shared_local[((ax18 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax18) + 544))];
      placeholder_d_shared_local[((ax18 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax18) + 560))];
    }
    for (int j_c8 = 0; j_c8 < 2; ++j_c8) {
      T_matmul_NN_local[(j_c8)] = (T_matmul_NN_local[(j_c8)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c8)]));
      T_matmul_NN_local[((j_c8 + 2))] = (T_matmul_NN_local[((j_c8 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c8 + 2))]));
      T_matmul_NN_local[((j_c8 + 4))] = (T_matmul_NN_local[((j_c8 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c8 + 4))]));
      T_matmul_NN_local[((j_c8 + 6))] = (T_matmul_NN_local[((j_c8 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c8 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 9))];
    for (int ax19 = 0; ax19 < 2; ++ax19) {
      placeholder_d_shared_local[(ax19)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax19) + 576))];
      placeholder_d_shared_local[((ax19 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax19) + 592))];
      placeholder_d_shared_local[((ax19 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax19) + 608))];
      placeholder_d_shared_local[((ax19 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax19) + 624))];
    }
    for (int j_c9 = 0; j_c9 < 2; ++j_c9) {
      T_matmul_NN_local[(j_c9)] = (T_matmul_NN_local[(j_c9)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c9)]));
      T_matmul_NN_local[((j_c9 + 2))] = (T_matmul_NN_local[((j_c9 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c9 + 2))]));
      T_matmul_NN_local[((j_c9 + 4))] = (T_matmul_NN_local[((j_c9 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c9 + 4))]));
      T_matmul_NN_local[((j_c9 + 6))] = (T_matmul_NN_local[((j_c9 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c9 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 10))];
    for (int ax110 = 0; ax110 < 2; ++ax110) {
      placeholder_d_shared_local[(ax110)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax110) + 640))];
      placeholder_d_shared_local[((ax110 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax110) + 656))];
      placeholder_d_shared_local[((ax110 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax110) + 672))];
      placeholder_d_shared_local[((ax110 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax110) + 688))];
    }
    for (int j_c10 = 0; j_c10 < 2; ++j_c10) {
      T_matmul_NN_local[(j_c10)] = (T_matmul_NN_local[(j_c10)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c10)]));
      T_matmul_NN_local[((j_c10 + 2))] = (T_matmul_NN_local[((j_c10 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c10 + 2))]));
      T_matmul_NN_local[((j_c10 + 4))] = (T_matmul_NN_local[((j_c10 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c10 + 4))]));
      T_matmul_NN_local[((j_c10 + 6))] = (T_matmul_NN_local[((j_c10 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c10 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 11))];
    for (int ax111 = 0; ax111 < 2; ++ax111) {
      placeholder_d_shared_local[(ax111)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax111) + 704))];
      placeholder_d_shared_local[((ax111 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax111) + 720))];
      placeholder_d_shared_local[((ax111 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax111) + 736))];
      placeholder_d_shared_local[((ax111 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax111) + 752))];
    }
    for (int j_c11 = 0; j_c11 < 2; ++j_c11) {
      T_matmul_NN_local[(j_c11)] = (T_matmul_NN_local[(j_c11)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c11)]));
      T_matmul_NN_local[((j_c11 + 2))] = (T_matmul_NN_local[((j_c11 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c11 + 2))]));
      T_matmul_NN_local[((j_c11 + 4))] = (T_matmul_NN_local[((j_c11 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c11 + 4))]));
      T_matmul_NN_local[((j_c11 + 6))] = (T_matmul_NN_local[((j_c11 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c11 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 12))];
    for (int ax112 = 0; ax112 < 2; ++ax112) {
      placeholder_d_shared_local[(ax112)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax112) + 768))];
      placeholder_d_shared_local[((ax112 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax112) + 784))];
      placeholder_d_shared_local[((ax112 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax112) + 800))];
      placeholder_d_shared_local[((ax112 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax112) + 816))];
    }
    for (int j_c12 = 0; j_c12 < 2; ++j_c12) {
      T_matmul_NN_local[(j_c12)] = (T_matmul_NN_local[(j_c12)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c12)]));
      T_matmul_NN_local[((j_c12 + 2))] = (T_matmul_NN_local[((j_c12 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c12 + 2))]));
      T_matmul_NN_local[((j_c12 + 4))] = (T_matmul_NN_local[((j_c12 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c12 + 4))]));
      T_matmul_NN_local[((j_c12 + 6))] = (T_matmul_NN_local[((j_c12 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c12 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 13))];
    for (int ax113 = 0; ax113 < 2; ++ax113) {
      placeholder_d_shared_local[(ax113)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax113) + 832))];
      placeholder_d_shared_local[((ax113 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax113) + 848))];
      placeholder_d_shared_local[((ax113 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax113) + 864))];
      placeholder_d_shared_local[((ax113 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax113) + 880))];
    }
    for (int j_c13 = 0; j_c13 < 2; ++j_c13) {
      T_matmul_NN_local[(j_c13)] = (T_matmul_NN_local[(j_c13)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c13)]));
      T_matmul_NN_local[((j_c13 + 2))] = (T_matmul_NN_local[((j_c13 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c13 + 2))]));
      T_matmul_NN_local[((j_c13 + 4))] = (T_matmul_NN_local[((j_c13 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c13 + 4))]));
      T_matmul_NN_local[((j_c13 + 6))] = (T_matmul_NN_local[((j_c13 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c13 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 14))];
    for (int ax114 = 0; ax114 < 2; ++ax114) {
      placeholder_d_shared_local[(ax114)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax114) + 896))];
      placeholder_d_shared_local[((ax114 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax114) + 912))];
      placeholder_d_shared_local[((ax114 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax114) + 928))];
      placeholder_d_shared_local[((ax114 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax114) + 944))];
    }
    for (int j_c14 = 0; j_c14 < 2; ++j_c14) {
      T_matmul_NN_local[(j_c14)] = (T_matmul_NN_local[(j_c14)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c14)]));
      T_matmul_NN_local[((j_c14 + 2))] = (T_matmul_NN_local[((j_c14 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c14 + 2))]));
      T_matmul_NN_local[((j_c14 + 4))] = (T_matmul_NN_local[((j_c14 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c14 + 4))]));
      T_matmul_NN_local[((j_c14 + 6))] = (T_matmul_NN_local[((j_c14 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c14 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 16)) + 15))];
    for (int ax115 = 0; ax115 < 2; ++ax115) {
      placeholder_d_shared_local[(ax115)] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax115) + 960))];
      placeholder_d_shared_local[((ax115 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax115) + 976))];
      placeholder_d_shared_local[((ax115 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax115) + 992))];
      placeholder_d_shared_local[((ax115 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 1024) + (((int)threadIdx.y) * 2)) + ax115) + 1008))];
    }
    for (int j_c15 = 0; j_c15 < 2; ++j_c15) {
      T_matmul_NN_local[(j_c15)] = (T_matmul_NN_local[(j_c15)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c15)]));
      T_matmul_NN_local[((j_c15 + 2))] = (T_matmul_NN_local[((j_c15 + 2))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c15 + 2))]));
      T_matmul_NN_local[((j_c15 + 4))] = (T_matmul_NN_local[((j_c15 + 4))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c15 + 4))]));
      T_matmul_NN_local[((j_c15 + 6))] = (T_matmul_NN_local[((j_c15 + 6))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c15 + 6))]));
    }
  }
  __syncthreads();
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 128))];
  for (int ax116 = 0; ax116 < 2; ++ax116) {
    placeholder_d_shared_local1[(ax116)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax116) + 1024))];
    placeholder_d_shared_local1[((ax116 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax116) + 1040))];
    placeholder_d_shared_local1[((ax116 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax116) + 1056))];
    placeholder_d_shared_local1[((ax116 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax116) + 1072))];
  }
  for (int j_c16 = 0; j_c16 < 2; ++j_c16) {
    T_matmul_NN_local[(j_c16)] = (T_matmul_NN_local[(j_c16)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c16)]));
    T_matmul_NN_local[((j_c16 + 2))] = (T_matmul_NN_local[((j_c16 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c16 + 2))]));
    T_matmul_NN_local[((j_c16 + 4))] = (T_matmul_NN_local[((j_c16 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c16 + 4))]));
    T_matmul_NN_local[((j_c16 + 6))] = (T_matmul_NN_local[((j_c16 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c16 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 129))];
  for (int ax117 = 0; ax117 < 2; ++ax117) {
    placeholder_d_shared_local1[(ax117)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax117) + 1088))];
    placeholder_d_shared_local1[((ax117 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax117) + 1104))];
    placeholder_d_shared_local1[((ax117 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax117) + 1120))];
    placeholder_d_shared_local1[((ax117 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax117) + 1136))];
  }
  for (int j_c17 = 0; j_c17 < 2; ++j_c17) {
    T_matmul_NN_local[(j_c17)] = (T_matmul_NN_local[(j_c17)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c17)]));
    T_matmul_NN_local[((j_c17 + 2))] = (T_matmul_NN_local[((j_c17 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c17 + 2))]));
    T_matmul_NN_local[((j_c17 + 4))] = (T_matmul_NN_local[((j_c17 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c17 + 4))]));
    T_matmul_NN_local[((j_c17 + 6))] = (T_matmul_NN_local[((j_c17 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c17 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 130))];
  for (int ax118 = 0; ax118 < 2; ++ax118) {
    placeholder_d_shared_local1[(ax118)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax118) + 1152))];
    placeholder_d_shared_local1[((ax118 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax118) + 1168))];
    placeholder_d_shared_local1[((ax118 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax118) + 1184))];
    placeholder_d_shared_local1[((ax118 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax118) + 1200))];
  }
  for (int j_c18 = 0; j_c18 < 2; ++j_c18) {
    T_matmul_NN_local[(j_c18)] = (T_matmul_NN_local[(j_c18)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c18)]));
    T_matmul_NN_local[((j_c18 + 2))] = (T_matmul_NN_local[((j_c18 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c18 + 2))]));
    T_matmul_NN_local[((j_c18 + 4))] = (T_matmul_NN_local[((j_c18 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c18 + 4))]));
    T_matmul_NN_local[((j_c18 + 6))] = (T_matmul_NN_local[((j_c18 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c18 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 131))];
  for (int ax119 = 0; ax119 < 2; ++ax119) {
    placeholder_d_shared_local1[(ax119)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax119) + 1216))];
    placeholder_d_shared_local1[((ax119 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax119) + 1232))];
    placeholder_d_shared_local1[((ax119 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax119) + 1248))];
    placeholder_d_shared_local1[((ax119 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax119) + 1264))];
  }
  for (int j_c19 = 0; j_c19 < 2; ++j_c19) {
    T_matmul_NN_local[(j_c19)] = (T_matmul_NN_local[(j_c19)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c19)]));
    T_matmul_NN_local[((j_c19 + 2))] = (T_matmul_NN_local[((j_c19 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c19 + 2))]));
    T_matmul_NN_local[((j_c19 + 4))] = (T_matmul_NN_local[((j_c19 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c19 + 4))]));
    T_matmul_NN_local[((j_c19 + 6))] = (T_matmul_NN_local[((j_c19 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c19 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 132))];
  for (int ax120 = 0; ax120 < 2; ++ax120) {
    placeholder_d_shared_local1[(ax120)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax120) + 1280))];
    placeholder_d_shared_local1[((ax120 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax120) + 1296))];
    placeholder_d_shared_local1[((ax120 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax120) + 1312))];
    placeholder_d_shared_local1[((ax120 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax120) + 1328))];
  }
  for (int j_c20 = 0; j_c20 < 2; ++j_c20) {
    T_matmul_NN_local[(j_c20)] = (T_matmul_NN_local[(j_c20)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c20)]));
    T_matmul_NN_local[((j_c20 + 2))] = (T_matmul_NN_local[((j_c20 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c20 + 2))]));
    T_matmul_NN_local[((j_c20 + 4))] = (T_matmul_NN_local[((j_c20 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c20 + 4))]));
    T_matmul_NN_local[((j_c20 + 6))] = (T_matmul_NN_local[((j_c20 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c20 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 133))];
  for (int ax121 = 0; ax121 < 2; ++ax121) {
    placeholder_d_shared_local1[(ax121)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax121) + 1344))];
    placeholder_d_shared_local1[((ax121 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax121) + 1360))];
    placeholder_d_shared_local1[((ax121 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax121) + 1376))];
    placeholder_d_shared_local1[((ax121 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax121) + 1392))];
  }
  for (int j_c21 = 0; j_c21 < 2; ++j_c21) {
    T_matmul_NN_local[(j_c21)] = (T_matmul_NN_local[(j_c21)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c21)]));
    T_matmul_NN_local[((j_c21 + 2))] = (T_matmul_NN_local[((j_c21 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c21 + 2))]));
    T_matmul_NN_local[((j_c21 + 4))] = (T_matmul_NN_local[((j_c21 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c21 + 4))]));
    T_matmul_NN_local[((j_c21 + 6))] = (T_matmul_NN_local[((j_c21 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c21 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 134))];
  for (int ax122 = 0; ax122 < 2; ++ax122) {
    placeholder_d_shared_local1[(ax122)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax122) + 1408))];
    placeholder_d_shared_local1[((ax122 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax122) + 1424))];
    placeholder_d_shared_local1[((ax122 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax122) + 1440))];
    placeholder_d_shared_local1[((ax122 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax122) + 1456))];
  }
  for (int j_c22 = 0; j_c22 < 2; ++j_c22) {
    T_matmul_NN_local[(j_c22)] = (T_matmul_NN_local[(j_c22)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c22)]));
    T_matmul_NN_local[((j_c22 + 2))] = (T_matmul_NN_local[((j_c22 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c22 + 2))]));
    T_matmul_NN_local[((j_c22 + 4))] = (T_matmul_NN_local[((j_c22 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c22 + 4))]));
    T_matmul_NN_local[((j_c22 + 6))] = (T_matmul_NN_local[((j_c22 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c22 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 135))];
  for (int ax123 = 0; ax123 < 2; ++ax123) {
    placeholder_d_shared_local1[(ax123)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax123) + 1472))];
    placeholder_d_shared_local1[((ax123 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax123) + 1488))];
    placeholder_d_shared_local1[((ax123 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax123) + 1504))];
    placeholder_d_shared_local1[((ax123 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax123) + 1520))];
  }
  for (int j_c23 = 0; j_c23 < 2; ++j_c23) {
    T_matmul_NN_local[(j_c23)] = (T_matmul_NN_local[(j_c23)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c23)]));
    T_matmul_NN_local[((j_c23 + 2))] = (T_matmul_NN_local[((j_c23 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c23 + 2))]));
    T_matmul_NN_local[((j_c23 + 4))] = (T_matmul_NN_local[((j_c23 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c23 + 4))]));
    T_matmul_NN_local[((j_c23 + 6))] = (T_matmul_NN_local[((j_c23 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c23 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 136))];
  for (int ax124 = 0; ax124 < 2; ++ax124) {
    placeholder_d_shared_local1[(ax124)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax124) + 1536))];
    placeholder_d_shared_local1[((ax124 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax124) + 1552))];
    placeholder_d_shared_local1[((ax124 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax124) + 1568))];
    placeholder_d_shared_local1[((ax124 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax124) + 1584))];
  }
  for (int j_c24 = 0; j_c24 < 2; ++j_c24) {
    T_matmul_NN_local[(j_c24)] = (T_matmul_NN_local[(j_c24)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c24)]));
    T_matmul_NN_local[((j_c24 + 2))] = (T_matmul_NN_local[((j_c24 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c24 + 2))]));
    T_matmul_NN_local[((j_c24 + 4))] = (T_matmul_NN_local[((j_c24 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c24 + 4))]));
    T_matmul_NN_local[((j_c24 + 6))] = (T_matmul_NN_local[((j_c24 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c24 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 137))];
  for (int ax125 = 0; ax125 < 2; ++ax125) {
    placeholder_d_shared_local1[(ax125)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax125) + 1600))];
    placeholder_d_shared_local1[((ax125 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax125) + 1616))];
    placeholder_d_shared_local1[((ax125 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax125) + 1632))];
    placeholder_d_shared_local1[((ax125 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax125) + 1648))];
  }
  for (int j_c25 = 0; j_c25 < 2; ++j_c25) {
    T_matmul_NN_local[(j_c25)] = (T_matmul_NN_local[(j_c25)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c25)]));
    T_matmul_NN_local[((j_c25 + 2))] = (T_matmul_NN_local[((j_c25 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c25 + 2))]));
    T_matmul_NN_local[((j_c25 + 4))] = (T_matmul_NN_local[((j_c25 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c25 + 4))]));
    T_matmul_NN_local[((j_c25 + 6))] = (T_matmul_NN_local[((j_c25 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c25 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 138))];
  for (int ax126 = 0; ax126 < 2; ++ax126) {
    placeholder_d_shared_local1[(ax126)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax126) + 1664))];
    placeholder_d_shared_local1[((ax126 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax126) + 1680))];
    placeholder_d_shared_local1[((ax126 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax126) + 1696))];
    placeholder_d_shared_local1[((ax126 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax126) + 1712))];
  }
  for (int j_c26 = 0; j_c26 < 2; ++j_c26) {
    T_matmul_NN_local[(j_c26)] = (T_matmul_NN_local[(j_c26)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c26)]));
    T_matmul_NN_local[((j_c26 + 2))] = (T_matmul_NN_local[((j_c26 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c26 + 2))]));
    T_matmul_NN_local[((j_c26 + 4))] = (T_matmul_NN_local[((j_c26 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c26 + 4))]));
    T_matmul_NN_local[((j_c26 + 6))] = (T_matmul_NN_local[((j_c26 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c26 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 139))];
  for (int ax127 = 0; ax127 < 2; ++ax127) {
    placeholder_d_shared_local1[(ax127)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax127) + 1728))];
    placeholder_d_shared_local1[((ax127 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax127) + 1744))];
    placeholder_d_shared_local1[((ax127 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax127) + 1760))];
    placeholder_d_shared_local1[((ax127 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax127) + 1776))];
  }
  for (int j_c27 = 0; j_c27 < 2; ++j_c27) {
    T_matmul_NN_local[(j_c27)] = (T_matmul_NN_local[(j_c27)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c27)]));
    T_matmul_NN_local[((j_c27 + 2))] = (T_matmul_NN_local[((j_c27 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c27 + 2))]));
    T_matmul_NN_local[((j_c27 + 4))] = (T_matmul_NN_local[((j_c27 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c27 + 4))]));
    T_matmul_NN_local[((j_c27 + 6))] = (T_matmul_NN_local[((j_c27 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c27 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 140))];
  for (int ax128 = 0; ax128 < 2; ++ax128) {
    placeholder_d_shared_local1[(ax128)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax128) + 1792))];
    placeholder_d_shared_local1[((ax128 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax128) + 1808))];
    placeholder_d_shared_local1[((ax128 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax128) + 1824))];
    placeholder_d_shared_local1[((ax128 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax128) + 1840))];
  }
  for (int j_c28 = 0; j_c28 < 2; ++j_c28) {
    T_matmul_NN_local[(j_c28)] = (T_matmul_NN_local[(j_c28)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c28)]));
    T_matmul_NN_local[((j_c28 + 2))] = (T_matmul_NN_local[((j_c28 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c28 + 2))]));
    T_matmul_NN_local[((j_c28 + 4))] = (T_matmul_NN_local[((j_c28 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c28 + 4))]));
    T_matmul_NN_local[((j_c28 + 6))] = (T_matmul_NN_local[((j_c28 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c28 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 141))];
  for (int ax129 = 0; ax129 < 2; ++ax129) {
    placeholder_d_shared_local1[(ax129)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax129) + 1856))];
    placeholder_d_shared_local1[((ax129 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax129) + 1872))];
    placeholder_d_shared_local1[((ax129 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax129) + 1888))];
    placeholder_d_shared_local1[((ax129 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax129) + 1904))];
  }
  for (int j_c29 = 0; j_c29 < 2; ++j_c29) {
    T_matmul_NN_local[(j_c29)] = (T_matmul_NN_local[(j_c29)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c29)]));
    T_matmul_NN_local[((j_c29 + 2))] = (T_matmul_NN_local[((j_c29 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c29 + 2))]));
    T_matmul_NN_local[((j_c29 + 4))] = (T_matmul_NN_local[((j_c29 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c29 + 4))]));
    T_matmul_NN_local[((j_c29 + 6))] = (T_matmul_NN_local[((j_c29 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c29 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 142))];
  for (int ax130 = 0; ax130 < 2; ++ax130) {
    placeholder_d_shared_local1[(ax130)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax130) + 1920))];
    placeholder_d_shared_local1[((ax130 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax130) + 1936))];
    placeholder_d_shared_local1[((ax130 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax130) + 1952))];
    placeholder_d_shared_local1[((ax130 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax130) + 1968))];
  }
  for (int j_c30 = 0; j_c30 < 2; ++j_c30) {
    T_matmul_NN_local[(j_c30)] = (T_matmul_NN_local[(j_c30)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c30)]));
    T_matmul_NN_local[((j_c30 + 2))] = (T_matmul_NN_local[((j_c30 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c30 + 2))]));
    T_matmul_NN_local[((j_c30 + 4))] = (T_matmul_NN_local[((j_c30 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c30 + 4))]));
    T_matmul_NN_local[((j_c30 + 6))] = (T_matmul_NN_local[((j_c30 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c30 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 143))];
  for (int ax131 = 0; ax131 < 2; ++ax131) {
    placeholder_d_shared_local1[(ax131)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax131) + 1984))];
    placeholder_d_shared_local1[((ax131 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax131) + 2000))];
    placeholder_d_shared_local1[((ax131 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax131) + 2016))];
    placeholder_d_shared_local1[((ax131 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax131) + 2032))];
  }
  for (int j_c31 = 0; j_c31 < 2; ++j_c31) {
    T_matmul_NN_local[(j_c31)] = (T_matmul_NN_local[(j_c31)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c31)]));
    T_matmul_NN_local[((j_c31 + 2))] = (T_matmul_NN_local[((j_c31 + 2))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c31 + 2))]));
    T_matmul_NN_local[((j_c31 + 4))] = (T_matmul_NN_local[((j_c31 + 4))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c31 + 4))]));
    T_matmul_NN_local[((j_c31 + 6))] = (T_matmul_NN_local[((j_c31 + 6))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c31 + 6))]));
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 2; ++j_inner_inner_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner))] = T_matmul_NN_local[(j_inner_inner_inner)];
    T_matmul_NN[(((((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 16))] = T_matmul_NN_local[((j_inner_inner_inner + 2))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32))] = T_matmul_NN_local[((j_inner_inner_inner + 4))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 256)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 48))] = T_matmul_NN_local[((j_inner_inner_inner + 6))];
  }
}

