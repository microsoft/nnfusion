//1024_16_1_16_16_1
//65536_1024_1024
//dim3 grid(1024, 16, 1);
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
  float T_matmul_NN_local[16];
  __shared__ float placeholder_shared[1024];
  __shared__ float placeholder_d_shared[1024];
  float placeholder_shared_local[4];
  float placeholder_d_shared_local[4];
  float placeholder_shared_local1[4];
  float placeholder_d_shared_local1[4];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      T_matmul_NN_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 8))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 4))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 2) + j_c_init) + 12))] = 0.000000e+00f;
    }
  }
  for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 8) {
        placeholder_shared[(((((((int)threadIdx.y) * 32) + (ax0_inner * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 4096)) + (ax0_inner * 1024)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
    if (((int)threadIdx.y) < 8) {
      placeholder_d_shared[((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[(((((((int)threadIdx.y) * 1024) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 127; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner1 = 0; ax0_inner1 < 4; ++ax0_inner1) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 8) {
          if ((((k_outer_outer * 8) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) < 1016) {
            placeholder_shared[((((((((k_outer_outer + 1) & 1) * 512) + (((int)threadIdx.y) * 32)) + (ax0_inner1 * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[((((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 4096)) + (ax0_inner1 * 1024)) + (k_outer_outer * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) + 8))];
          }
        }
      }
    }
    for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
      if (((int)threadIdx.y) < 8) {
        placeholder_d_shared[(((((((k_outer_outer + 1) & 1) * 512) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[(((((((k_outer_outer * 8192) + (((int)threadIdx.y) * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 8192))];
      }
    }
    for (int ax0 = 0; ax0 < 2; ++ax0) {
      placeholder_shared_local[(ax0)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax0 * 8)))];
      placeholder_shared_local[((ax0 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax0 * 8)) + 256))];
    }
    for (int ax1 = 0; ax1 < 2; ++ax1) {
      placeholder_d_shared_local[(ax1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax1))];
      placeholder_d_shared_local[((ax1 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax1) + 32))];
    }
    for (int i_c = 0; i_c < 2; ++i_c) {
      for (int j_c = 0; j_c < 2; ++j_c) {
        T_matmul_NN_local[(((i_c * 2) + j_c))] = (T_matmul_NN_local[(((i_c * 2) + j_c))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(j_c)]));
        T_matmul_NN_local[((((i_c * 2) + j_c) + 8))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 8))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[((j_c + 2))]));
        T_matmul_NN_local[((((i_c * 2) + j_c) + 4))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 4))] + (placeholder_shared_local[((i_c + 2))] * placeholder_d_shared_local[(j_c)]));
        T_matmul_NN_local[((((i_c * 2) + j_c) + 12))] = (T_matmul_NN_local[((((i_c * 2) + j_c) + 12))] + (placeholder_shared_local[((i_c + 2))] * placeholder_d_shared_local[((j_c + 2))]));
      }
    }
    for (int ax01 = 0; ax01 < 2; ++ax01) {
      placeholder_shared_local[(ax01)] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax01 * 8)) + 1))];
      placeholder_shared_local[((ax01 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax01 * 8)) + 257))];
    }
    for (int ax11 = 0; ax11 < 2; ++ax11) {
      placeholder_d_shared_local[(ax11)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax11) + 64))];
      placeholder_d_shared_local[((ax11 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax11) + 96))];
    }
    for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
      for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
        T_matmul_NN_local[(((i_c1 * 2) + j_c1))] = (T_matmul_NN_local[(((i_c1 * 2) + j_c1))] + (placeholder_shared_local[(i_c1)] * placeholder_d_shared_local[(j_c1)]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 8))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 8))] + (placeholder_shared_local[(i_c1)] * placeholder_d_shared_local[((j_c1 + 2))]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 4))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 4))] + (placeholder_shared_local[((i_c1 + 2))] * placeholder_d_shared_local[(j_c1)]));
        T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 12))] = (T_matmul_NN_local[((((i_c1 * 2) + j_c1) + 12))] + (placeholder_shared_local[((i_c1 + 2))] * placeholder_d_shared_local[((j_c1 + 2))]));
      }
    }
    for (int ax02 = 0; ax02 < 2; ++ax02) {
      placeholder_shared_local[(ax02)] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax02 * 8)) + 2))];
      placeholder_shared_local[((ax02 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax02 * 8)) + 258))];
    }
    for (int ax12 = 0; ax12 < 2; ++ax12) {
      placeholder_d_shared_local[(ax12)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax12) + 128))];
      placeholder_d_shared_local[((ax12 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax12) + 160))];
    }
    for (int i_c2 = 0; i_c2 < 2; ++i_c2) {
      for (int j_c2 = 0; j_c2 < 2; ++j_c2) {
        T_matmul_NN_local[(((i_c2 * 2) + j_c2))] = (T_matmul_NN_local[(((i_c2 * 2) + j_c2))] + (placeholder_shared_local[(i_c2)] * placeholder_d_shared_local[(j_c2)]));
        T_matmul_NN_local[((((i_c2 * 2) + j_c2) + 8))] = (T_matmul_NN_local[((((i_c2 * 2) + j_c2) + 8))] + (placeholder_shared_local[(i_c2)] * placeholder_d_shared_local[((j_c2 + 2))]));
        T_matmul_NN_local[((((i_c2 * 2) + j_c2) + 4))] = (T_matmul_NN_local[((((i_c2 * 2) + j_c2) + 4))] + (placeholder_shared_local[((i_c2 + 2))] * placeholder_d_shared_local[(j_c2)]));
        T_matmul_NN_local[((((i_c2 * 2) + j_c2) + 12))] = (T_matmul_NN_local[((((i_c2 * 2) + j_c2) + 12))] + (placeholder_shared_local[((i_c2 + 2))] * placeholder_d_shared_local[((j_c2 + 2))]));
      }
    }
    for (int ax03 = 0; ax03 < 2; ++ax03) {
      placeholder_shared_local[(ax03)] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax03 * 8)) + 3))];
      placeholder_shared_local[((ax03 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax03 * 8)) + 259))];
    }
    for (int ax13 = 0; ax13 < 2; ++ax13) {
      placeholder_d_shared_local[(ax13)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax13) + 192))];
      placeholder_d_shared_local[((ax13 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax13) + 224))];
    }
    for (int i_c3 = 0; i_c3 < 2; ++i_c3) {
      for (int j_c3 = 0; j_c3 < 2; ++j_c3) {
        T_matmul_NN_local[(((i_c3 * 2) + j_c3))] = (T_matmul_NN_local[(((i_c3 * 2) + j_c3))] + (placeholder_shared_local[(i_c3)] * placeholder_d_shared_local[(j_c3)]));
        T_matmul_NN_local[((((i_c3 * 2) + j_c3) + 8))] = (T_matmul_NN_local[((((i_c3 * 2) + j_c3) + 8))] + (placeholder_shared_local[(i_c3)] * placeholder_d_shared_local[((j_c3 + 2))]));
        T_matmul_NN_local[((((i_c3 * 2) + j_c3) + 4))] = (T_matmul_NN_local[((((i_c3 * 2) + j_c3) + 4))] + (placeholder_shared_local[((i_c3 + 2))] * placeholder_d_shared_local[(j_c3)]));
        T_matmul_NN_local[((((i_c3 * 2) + j_c3) + 12))] = (T_matmul_NN_local[((((i_c3 * 2) + j_c3) + 12))] + (placeholder_shared_local[((i_c3 + 2))] * placeholder_d_shared_local[((j_c3 + 2))]));
      }
    }
    for (int ax04 = 0; ax04 < 2; ++ax04) {
      placeholder_shared_local[(ax04)] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax04 * 8)) + 4))];
      placeholder_shared_local[((ax04 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax04 * 8)) + 260))];
    }
    for (int ax14 = 0; ax14 < 2; ++ax14) {
      placeholder_d_shared_local[(ax14)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax14) + 256))];
      placeholder_d_shared_local[((ax14 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax14) + 288))];
    }
    for (int i_c4 = 0; i_c4 < 2; ++i_c4) {
      for (int j_c4 = 0; j_c4 < 2; ++j_c4) {
        T_matmul_NN_local[(((i_c4 * 2) + j_c4))] = (T_matmul_NN_local[(((i_c4 * 2) + j_c4))] + (placeholder_shared_local[(i_c4)] * placeholder_d_shared_local[(j_c4)]));
        T_matmul_NN_local[((((i_c4 * 2) + j_c4) + 8))] = (T_matmul_NN_local[((((i_c4 * 2) + j_c4) + 8))] + (placeholder_shared_local[(i_c4)] * placeholder_d_shared_local[((j_c4 + 2))]));
        T_matmul_NN_local[((((i_c4 * 2) + j_c4) + 4))] = (T_matmul_NN_local[((((i_c4 * 2) + j_c4) + 4))] + (placeholder_shared_local[((i_c4 + 2))] * placeholder_d_shared_local[(j_c4)]));
        T_matmul_NN_local[((((i_c4 * 2) + j_c4) + 12))] = (T_matmul_NN_local[((((i_c4 * 2) + j_c4) + 12))] + (placeholder_shared_local[((i_c4 + 2))] * placeholder_d_shared_local[((j_c4 + 2))]));
      }
    }
    for (int ax05 = 0; ax05 < 2; ++ax05) {
      placeholder_shared_local[(ax05)] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax05 * 8)) + 5))];
      placeholder_shared_local[((ax05 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax05 * 8)) + 261))];
    }
    for (int ax15 = 0; ax15 < 2; ++ax15) {
      placeholder_d_shared_local[(ax15)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax15) + 320))];
      placeholder_d_shared_local[((ax15 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax15) + 352))];
    }
    for (int i_c5 = 0; i_c5 < 2; ++i_c5) {
      for (int j_c5 = 0; j_c5 < 2; ++j_c5) {
        T_matmul_NN_local[(((i_c5 * 2) + j_c5))] = (T_matmul_NN_local[(((i_c5 * 2) + j_c5))] + (placeholder_shared_local[(i_c5)] * placeholder_d_shared_local[(j_c5)]));
        T_matmul_NN_local[((((i_c5 * 2) + j_c5) + 8))] = (T_matmul_NN_local[((((i_c5 * 2) + j_c5) + 8))] + (placeholder_shared_local[(i_c5)] * placeholder_d_shared_local[((j_c5 + 2))]));
        T_matmul_NN_local[((((i_c5 * 2) + j_c5) + 4))] = (T_matmul_NN_local[((((i_c5 * 2) + j_c5) + 4))] + (placeholder_shared_local[((i_c5 + 2))] * placeholder_d_shared_local[(j_c5)]));
        T_matmul_NN_local[((((i_c5 * 2) + j_c5) + 12))] = (T_matmul_NN_local[((((i_c5 * 2) + j_c5) + 12))] + (placeholder_shared_local[((i_c5 + 2))] * placeholder_d_shared_local[((j_c5 + 2))]));
      }
    }
    for (int ax06 = 0; ax06 < 2; ++ax06) {
      placeholder_shared_local[(ax06)] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax06 * 8)) + 6))];
      placeholder_shared_local[((ax06 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax06 * 8)) + 262))];
    }
    for (int ax16 = 0; ax16 < 2; ++ax16) {
      placeholder_d_shared_local[(ax16)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax16) + 384))];
      placeholder_d_shared_local[((ax16 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax16) + 416))];
    }
    for (int i_c6 = 0; i_c6 < 2; ++i_c6) {
      for (int j_c6 = 0; j_c6 < 2; ++j_c6) {
        T_matmul_NN_local[(((i_c6 * 2) + j_c6))] = (T_matmul_NN_local[(((i_c6 * 2) + j_c6))] + (placeholder_shared_local[(i_c6)] * placeholder_d_shared_local[(j_c6)]));
        T_matmul_NN_local[((((i_c6 * 2) + j_c6) + 8))] = (T_matmul_NN_local[((((i_c6 * 2) + j_c6) + 8))] + (placeholder_shared_local[(i_c6)] * placeholder_d_shared_local[((j_c6 + 2))]));
        T_matmul_NN_local[((((i_c6 * 2) + j_c6) + 4))] = (T_matmul_NN_local[((((i_c6 * 2) + j_c6) + 4))] + (placeholder_shared_local[((i_c6 + 2))] * placeholder_d_shared_local[(j_c6)]));
        T_matmul_NN_local[((((i_c6 * 2) + j_c6) + 12))] = (T_matmul_NN_local[((((i_c6 * 2) + j_c6) + 12))] + (placeholder_shared_local[((i_c6 + 2))] * placeholder_d_shared_local[((j_c6 + 2))]));
      }
    }
    for (int ax07 = 0; ax07 < 2; ++ax07) {
      placeholder_shared_local[(ax07)] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax07 * 8)) + 7))];
      placeholder_shared_local[((ax07 + 2))] = placeholder_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 16)) + (ax07 * 8)) + 263))];
    }
    for (int ax17 = 0; ax17 < 2; ++ax17) {
      placeholder_d_shared_local[(ax17)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax17) + 448))];
      placeholder_d_shared_local[((ax17 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax17) + 480))];
    }
    for (int i_c7 = 0; i_c7 < 2; ++i_c7) {
      for (int j_c7 = 0; j_c7 < 2; ++j_c7) {
        T_matmul_NN_local[(((i_c7 * 2) + j_c7))] = (T_matmul_NN_local[(((i_c7 * 2) + j_c7))] + (placeholder_shared_local[(i_c7)] * placeholder_d_shared_local[(j_c7)]));
        T_matmul_NN_local[((((i_c7 * 2) + j_c7) + 8))] = (T_matmul_NN_local[((((i_c7 * 2) + j_c7) + 8))] + (placeholder_shared_local[(i_c7)] * placeholder_d_shared_local[((j_c7 + 2))]));
        T_matmul_NN_local[((((i_c7 * 2) + j_c7) + 4))] = (T_matmul_NN_local[((((i_c7 * 2) + j_c7) + 4))] + (placeholder_shared_local[((i_c7 + 2))] * placeholder_d_shared_local[(j_c7)]));
        T_matmul_NN_local[((((i_c7 * 2) + j_c7) + 12))] = (T_matmul_NN_local[((((i_c7 * 2) + j_c7) + 12))] + (placeholder_shared_local[((i_c7 + 2))] * placeholder_d_shared_local[((j_c7 + 2))]));
      }
    }
  }
  __syncthreads();
  for (int ax08 = 0; ax08 < 2; ++ax08) {
    placeholder_shared_local1[(ax08)] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax08 * 8)) + 512))];
    placeholder_shared_local1[((ax08 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax08 * 8)) + 768))];
  }
  for (int ax18 = 0; ax18 < 2; ++ax18) {
    placeholder_d_shared_local1[(ax18)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax18) + 512))];
    placeholder_d_shared_local1[((ax18 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax18) + 544))];
  }
  for (int i_c8 = 0; i_c8 < 2; ++i_c8) {
    for (int j_c8 = 0; j_c8 < 2; ++j_c8) {
      T_matmul_NN_local[(((i_c8 * 2) + j_c8))] = (T_matmul_NN_local[(((i_c8 * 2) + j_c8))] + (placeholder_shared_local1[(i_c8)] * placeholder_d_shared_local1[(j_c8)]));
      T_matmul_NN_local[((((i_c8 * 2) + j_c8) + 8))] = (T_matmul_NN_local[((((i_c8 * 2) + j_c8) + 8))] + (placeholder_shared_local1[(i_c8)] * placeholder_d_shared_local1[((j_c8 + 2))]));
      T_matmul_NN_local[((((i_c8 * 2) + j_c8) + 4))] = (T_matmul_NN_local[((((i_c8 * 2) + j_c8) + 4))] + (placeholder_shared_local1[((i_c8 + 2))] * placeholder_d_shared_local1[(j_c8)]));
      T_matmul_NN_local[((((i_c8 * 2) + j_c8) + 12))] = (T_matmul_NN_local[((((i_c8 * 2) + j_c8) + 12))] + (placeholder_shared_local1[((i_c8 + 2))] * placeholder_d_shared_local1[((j_c8 + 2))]));
    }
  }
  for (int ax09 = 0; ax09 < 2; ++ax09) {
    placeholder_shared_local1[(ax09)] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax09 * 8)) + 513))];
    placeholder_shared_local1[((ax09 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax09 * 8)) + 769))];
  }
  for (int ax19 = 0; ax19 < 2; ++ax19) {
    placeholder_d_shared_local1[(ax19)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax19) + 576))];
    placeholder_d_shared_local1[((ax19 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax19) + 608))];
  }
  for (int i_c9 = 0; i_c9 < 2; ++i_c9) {
    for (int j_c9 = 0; j_c9 < 2; ++j_c9) {
      T_matmul_NN_local[(((i_c9 * 2) + j_c9))] = (T_matmul_NN_local[(((i_c9 * 2) + j_c9))] + (placeholder_shared_local1[(i_c9)] * placeholder_d_shared_local1[(j_c9)]));
      T_matmul_NN_local[((((i_c9 * 2) + j_c9) + 8))] = (T_matmul_NN_local[((((i_c9 * 2) + j_c9) + 8))] + (placeholder_shared_local1[(i_c9)] * placeholder_d_shared_local1[((j_c9 + 2))]));
      T_matmul_NN_local[((((i_c9 * 2) + j_c9) + 4))] = (T_matmul_NN_local[((((i_c9 * 2) + j_c9) + 4))] + (placeholder_shared_local1[((i_c9 + 2))] * placeholder_d_shared_local1[(j_c9)]));
      T_matmul_NN_local[((((i_c9 * 2) + j_c9) + 12))] = (T_matmul_NN_local[((((i_c9 * 2) + j_c9) + 12))] + (placeholder_shared_local1[((i_c9 + 2))] * placeholder_d_shared_local1[((j_c9 + 2))]));
    }
  }
  for (int ax010 = 0; ax010 < 2; ++ax010) {
    placeholder_shared_local1[(ax010)] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax010 * 8)) + 514))];
    placeholder_shared_local1[((ax010 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax010 * 8)) + 770))];
  }
  for (int ax110 = 0; ax110 < 2; ++ax110) {
    placeholder_d_shared_local1[(ax110)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax110) + 640))];
    placeholder_d_shared_local1[((ax110 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax110) + 672))];
  }
  for (int i_c10 = 0; i_c10 < 2; ++i_c10) {
    for (int j_c10 = 0; j_c10 < 2; ++j_c10) {
      T_matmul_NN_local[(((i_c10 * 2) + j_c10))] = (T_matmul_NN_local[(((i_c10 * 2) + j_c10))] + (placeholder_shared_local1[(i_c10)] * placeholder_d_shared_local1[(j_c10)]));
      T_matmul_NN_local[((((i_c10 * 2) + j_c10) + 8))] = (T_matmul_NN_local[((((i_c10 * 2) + j_c10) + 8))] + (placeholder_shared_local1[(i_c10)] * placeholder_d_shared_local1[((j_c10 + 2))]));
      T_matmul_NN_local[((((i_c10 * 2) + j_c10) + 4))] = (T_matmul_NN_local[((((i_c10 * 2) + j_c10) + 4))] + (placeholder_shared_local1[((i_c10 + 2))] * placeholder_d_shared_local1[(j_c10)]));
      T_matmul_NN_local[((((i_c10 * 2) + j_c10) + 12))] = (T_matmul_NN_local[((((i_c10 * 2) + j_c10) + 12))] + (placeholder_shared_local1[((i_c10 + 2))] * placeholder_d_shared_local1[((j_c10 + 2))]));
    }
  }
  for (int ax011 = 0; ax011 < 2; ++ax011) {
    placeholder_shared_local1[(ax011)] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax011 * 8)) + 515))];
    placeholder_shared_local1[((ax011 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax011 * 8)) + 771))];
  }
  for (int ax111 = 0; ax111 < 2; ++ax111) {
    placeholder_d_shared_local1[(ax111)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax111) + 704))];
    placeholder_d_shared_local1[((ax111 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax111) + 736))];
  }
  for (int i_c11 = 0; i_c11 < 2; ++i_c11) {
    for (int j_c11 = 0; j_c11 < 2; ++j_c11) {
      T_matmul_NN_local[(((i_c11 * 2) + j_c11))] = (T_matmul_NN_local[(((i_c11 * 2) + j_c11))] + (placeholder_shared_local1[(i_c11)] * placeholder_d_shared_local1[(j_c11)]));
      T_matmul_NN_local[((((i_c11 * 2) + j_c11) + 8))] = (T_matmul_NN_local[((((i_c11 * 2) + j_c11) + 8))] + (placeholder_shared_local1[(i_c11)] * placeholder_d_shared_local1[((j_c11 + 2))]));
      T_matmul_NN_local[((((i_c11 * 2) + j_c11) + 4))] = (T_matmul_NN_local[((((i_c11 * 2) + j_c11) + 4))] + (placeholder_shared_local1[((i_c11 + 2))] * placeholder_d_shared_local1[(j_c11)]));
      T_matmul_NN_local[((((i_c11 * 2) + j_c11) + 12))] = (T_matmul_NN_local[((((i_c11 * 2) + j_c11) + 12))] + (placeholder_shared_local1[((i_c11 + 2))] * placeholder_d_shared_local1[((j_c11 + 2))]));
    }
  }
  for (int ax012 = 0; ax012 < 2; ++ax012) {
    placeholder_shared_local1[(ax012)] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax012 * 8)) + 516))];
    placeholder_shared_local1[((ax012 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax012 * 8)) + 772))];
  }
  for (int ax112 = 0; ax112 < 2; ++ax112) {
    placeholder_d_shared_local1[(ax112)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax112) + 768))];
    placeholder_d_shared_local1[((ax112 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax112) + 800))];
  }
  for (int i_c12 = 0; i_c12 < 2; ++i_c12) {
    for (int j_c12 = 0; j_c12 < 2; ++j_c12) {
      T_matmul_NN_local[(((i_c12 * 2) + j_c12))] = (T_matmul_NN_local[(((i_c12 * 2) + j_c12))] + (placeholder_shared_local1[(i_c12)] * placeholder_d_shared_local1[(j_c12)]));
      T_matmul_NN_local[((((i_c12 * 2) + j_c12) + 8))] = (T_matmul_NN_local[((((i_c12 * 2) + j_c12) + 8))] + (placeholder_shared_local1[(i_c12)] * placeholder_d_shared_local1[((j_c12 + 2))]));
      T_matmul_NN_local[((((i_c12 * 2) + j_c12) + 4))] = (T_matmul_NN_local[((((i_c12 * 2) + j_c12) + 4))] + (placeholder_shared_local1[((i_c12 + 2))] * placeholder_d_shared_local1[(j_c12)]));
      T_matmul_NN_local[((((i_c12 * 2) + j_c12) + 12))] = (T_matmul_NN_local[((((i_c12 * 2) + j_c12) + 12))] + (placeholder_shared_local1[((i_c12 + 2))] * placeholder_d_shared_local1[((j_c12 + 2))]));
    }
  }
  for (int ax013 = 0; ax013 < 2; ++ax013) {
    placeholder_shared_local1[(ax013)] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax013 * 8)) + 517))];
    placeholder_shared_local1[((ax013 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax013 * 8)) + 773))];
  }
  for (int ax113 = 0; ax113 < 2; ++ax113) {
    placeholder_d_shared_local1[(ax113)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax113) + 832))];
    placeholder_d_shared_local1[((ax113 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax113) + 864))];
  }
  for (int i_c13 = 0; i_c13 < 2; ++i_c13) {
    for (int j_c13 = 0; j_c13 < 2; ++j_c13) {
      T_matmul_NN_local[(((i_c13 * 2) + j_c13))] = (T_matmul_NN_local[(((i_c13 * 2) + j_c13))] + (placeholder_shared_local1[(i_c13)] * placeholder_d_shared_local1[(j_c13)]));
      T_matmul_NN_local[((((i_c13 * 2) + j_c13) + 8))] = (T_matmul_NN_local[((((i_c13 * 2) + j_c13) + 8))] + (placeholder_shared_local1[(i_c13)] * placeholder_d_shared_local1[((j_c13 + 2))]));
      T_matmul_NN_local[((((i_c13 * 2) + j_c13) + 4))] = (T_matmul_NN_local[((((i_c13 * 2) + j_c13) + 4))] + (placeholder_shared_local1[((i_c13 + 2))] * placeholder_d_shared_local1[(j_c13)]));
      T_matmul_NN_local[((((i_c13 * 2) + j_c13) + 12))] = (T_matmul_NN_local[((((i_c13 * 2) + j_c13) + 12))] + (placeholder_shared_local1[((i_c13 + 2))] * placeholder_d_shared_local1[((j_c13 + 2))]));
    }
  }
  for (int ax014 = 0; ax014 < 2; ++ax014) {
    placeholder_shared_local1[(ax014)] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax014 * 8)) + 518))];
    placeholder_shared_local1[((ax014 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax014 * 8)) + 774))];
  }
  for (int ax114 = 0; ax114 < 2; ++ax114) {
    placeholder_d_shared_local1[(ax114)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax114) + 896))];
    placeholder_d_shared_local1[((ax114 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax114) + 928))];
  }
  for (int i_c14 = 0; i_c14 < 2; ++i_c14) {
    for (int j_c14 = 0; j_c14 < 2; ++j_c14) {
      T_matmul_NN_local[(((i_c14 * 2) + j_c14))] = (T_matmul_NN_local[(((i_c14 * 2) + j_c14))] + (placeholder_shared_local1[(i_c14)] * placeholder_d_shared_local1[(j_c14)]));
      T_matmul_NN_local[((((i_c14 * 2) + j_c14) + 8))] = (T_matmul_NN_local[((((i_c14 * 2) + j_c14) + 8))] + (placeholder_shared_local1[(i_c14)] * placeholder_d_shared_local1[((j_c14 + 2))]));
      T_matmul_NN_local[((((i_c14 * 2) + j_c14) + 4))] = (T_matmul_NN_local[((((i_c14 * 2) + j_c14) + 4))] + (placeholder_shared_local1[((i_c14 + 2))] * placeholder_d_shared_local1[(j_c14)]));
      T_matmul_NN_local[((((i_c14 * 2) + j_c14) + 12))] = (T_matmul_NN_local[((((i_c14 * 2) + j_c14) + 12))] + (placeholder_shared_local1[((i_c14 + 2))] * placeholder_d_shared_local1[((j_c14 + 2))]));
    }
  }
  for (int ax015 = 0; ax015 < 2; ++ax015) {
    placeholder_shared_local1[(ax015)] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax015 * 8)) + 519))];
    placeholder_shared_local1[((ax015 + 2))] = placeholder_shared[((((((int)threadIdx.x) * 16) + (ax015 * 8)) + 775))];
  }
  for (int ax115 = 0; ax115 < 2; ++ax115) {
    placeholder_d_shared_local1[(ax115)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax115) + 960))];
    placeholder_d_shared_local1[((ax115 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax115) + 992))];
  }
  for (int i_c15 = 0; i_c15 < 2; ++i_c15) {
    for (int j_c15 = 0; j_c15 < 2; ++j_c15) {
      T_matmul_NN_local[(((i_c15 * 2) + j_c15))] = (T_matmul_NN_local[(((i_c15 * 2) + j_c15))] + (placeholder_shared_local1[(i_c15)] * placeholder_d_shared_local1[(j_c15)]));
      T_matmul_NN_local[((((i_c15 * 2) + j_c15) + 8))] = (T_matmul_NN_local[((((i_c15 * 2) + j_c15) + 8))] + (placeholder_shared_local1[(i_c15)] * placeholder_d_shared_local1[((j_c15 + 2))]));
      T_matmul_NN_local[((((i_c15 * 2) + j_c15) + 4))] = (T_matmul_NN_local[((((i_c15 * 2) + j_c15) + 4))] + (placeholder_shared_local1[((i_c15 + 2))] * placeholder_d_shared_local1[(j_c15)]));
      T_matmul_NN_local[((((i_c15 * 2) + j_c15) + 12))] = (T_matmul_NN_local[((((i_c15 * 2) + j_c15) + 12))] + (placeholder_shared_local1[((i_c15 + 2))] * placeholder_d_shared_local1[((j_c15 + 2))]));
    }
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 2; ++j_inner_inner_inner) {
    for (int i_inner_inner_inner = 0; i_inner_inner_inner < 2; ++i_inner_inner_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 2048)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner))] = T_matmul_NN_local[(((i_inner_inner_inner * 2) + j_inner_inner_inner))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 2048)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 8))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 2048)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32768))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 4))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.x) * 2048)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 32800))] = T_matmul_NN_local[((((i_inner_inner_inner * 2) + j_inner_inner_inner) + 12))];
    }
  }
}

