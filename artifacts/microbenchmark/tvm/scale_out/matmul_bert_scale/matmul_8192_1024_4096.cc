//64_128_1_8_8_1
//8192_1024_4096
//dim3 grid(64, 128, 1);
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
  float T_matmul_NN_local[64];
  __shared__ float placeholder_shared[4096];
  __shared__ float placeholder_d_shared[1024];
  float placeholder_shared_local[16];
  float placeholder_d_shared_local[4];
  float placeholder_shared_local1[16];
  float placeholder_d_shared_local1[4];
  for (int vthread_s = 0; vthread_s < 16; ++vthread_s) {
    T_matmul_NN_local[(vthread_s)] = 0.000000e+00f;
    T_matmul_NN_local[((vthread_s + 16))] = 0.000000e+00f;
    T_matmul_NN_local[((vthread_s + 32))] = 0.000000e+00f;
    T_matmul_NN_local[((vthread_s + 48))] = 0.000000e+00f;
  }
  for (int ax0_inner = 0; ax0_inner < 16; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 16) {
        placeholder_shared[(((((((int)threadIdx.y) * 256) + (ax0_inner * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.y) * 16384)) + (ax0_inner * 1024)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
    for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
      placeholder_d_shared[(((((((int)threadIdx.y) * 64) + (ax0_inner1 * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[((((((((int)threadIdx.y) * 8192) + (ax0_inner1 * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 63; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner2 = 0; ax0_inner2 < 16; ++ax0_inner2) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 16) {
          if ((((k_outer_outer * 16) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) < 1008) {
            placeholder_shared[((((((((k_outer_outer + 1) & 1) * 2048) + (((int)threadIdx.y) * 256)) + (ax0_inner2 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[((((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.y) * 16384)) + (ax0_inner2 * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) + 16))];
          }
        }
      }
    }
    for (int ax0_inner3 = 0; ax0_inner3 < 2; ++ax0_inner3) {
      for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
        placeholder_d_shared[((((((((k_outer_outer + 1) & 1) * 512) + (((int)threadIdx.y) * 64)) + (ax0_inner3 * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[((((((((k_outer_outer * 65536) + (((int)threadIdx.y) * 8192)) + (ax0_inner3 * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 65536))];
      }
    }
    for (int vthread_s1 = 0; vthread_s1 < 16; ++vthread_s1) {
      placeholder_shared_local[(vthread_s1)] = placeholder_shared[(((((k_outer_outer & 1) * 2048) + (vthread_s1 * 128)) + (((int)threadIdx.x) * 16)))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 8))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 16))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 24))];
    for (int vthread_s2 = 0; vthread_s2 < 16; ++vthread_s2) {
      T_matmul_NN_local[(vthread_s2)] = (T_matmul_NN_local[(vthread_s2)] + (placeholder_shared_local[(vthread_s2)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s2 + 16))] = (T_matmul_NN_local[((vthread_s2 + 16))] + (placeholder_shared_local[(vthread_s2)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s2 + 32))] = (T_matmul_NN_local[((vthread_s2 + 32))] + (placeholder_shared_local[(vthread_s2)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s2 + 48))] = (T_matmul_NN_local[((vthread_s2 + 48))] + (placeholder_shared_local[(vthread_s2)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s3 = 0; vthread_s3 < 16; ++vthread_s3) {
      placeholder_shared_local[(vthread_s3)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s3 * 128)) + (((int)threadIdx.x) * 16)) + 1))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 32))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 40))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 48))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 56))];
    for (int vthread_s4 = 0; vthread_s4 < 16; ++vthread_s4) {
      T_matmul_NN_local[(vthread_s4)] = (T_matmul_NN_local[(vthread_s4)] + (placeholder_shared_local[(vthread_s4)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s4 + 16))] = (T_matmul_NN_local[((vthread_s4 + 16))] + (placeholder_shared_local[(vthread_s4)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s4 + 32))] = (T_matmul_NN_local[((vthread_s4 + 32))] + (placeholder_shared_local[(vthread_s4)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s4 + 48))] = (T_matmul_NN_local[((vthread_s4 + 48))] + (placeholder_shared_local[(vthread_s4)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s5 = 0; vthread_s5 < 16; ++vthread_s5) {
      placeholder_shared_local[(vthread_s5)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s5 * 128)) + (((int)threadIdx.x) * 16)) + 2))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 64))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 72))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 80))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 88))];
    for (int vthread_s6 = 0; vthread_s6 < 16; ++vthread_s6) {
      T_matmul_NN_local[(vthread_s6)] = (T_matmul_NN_local[(vthread_s6)] + (placeholder_shared_local[(vthread_s6)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s6 + 16))] = (T_matmul_NN_local[((vthread_s6 + 16))] + (placeholder_shared_local[(vthread_s6)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s6 + 32))] = (T_matmul_NN_local[((vthread_s6 + 32))] + (placeholder_shared_local[(vthread_s6)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s6 + 48))] = (T_matmul_NN_local[((vthread_s6 + 48))] + (placeholder_shared_local[(vthread_s6)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s7 = 0; vthread_s7 < 16; ++vthread_s7) {
      placeholder_shared_local[(vthread_s7)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s7 * 128)) + (((int)threadIdx.x) * 16)) + 3))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 96))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 104))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 112))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 120))];
    for (int vthread_s8 = 0; vthread_s8 < 16; ++vthread_s8) {
      T_matmul_NN_local[(vthread_s8)] = (T_matmul_NN_local[(vthread_s8)] + (placeholder_shared_local[(vthread_s8)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s8 + 16))] = (T_matmul_NN_local[((vthread_s8 + 16))] + (placeholder_shared_local[(vthread_s8)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s8 + 32))] = (T_matmul_NN_local[((vthread_s8 + 32))] + (placeholder_shared_local[(vthread_s8)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s8 + 48))] = (T_matmul_NN_local[((vthread_s8 + 48))] + (placeholder_shared_local[(vthread_s8)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s9 = 0; vthread_s9 < 16; ++vthread_s9) {
      placeholder_shared_local[(vthread_s9)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s9 * 128)) + (((int)threadIdx.x) * 16)) + 4))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 128))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 136))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 144))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 152))];
    for (int vthread_s10 = 0; vthread_s10 < 16; ++vthread_s10) {
      T_matmul_NN_local[(vthread_s10)] = (T_matmul_NN_local[(vthread_s10)] + (placeholder_shared_local[(vthread_s10)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s10 + 16))] = (T_matmul_NN_local[((vthread_s10 + 16))] + (placeholder_shared_local[(vthread_s10)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s10 + 32))] = (T_matmul_NN_local[((vthread_s10 + 32))] + (placeholder_shared_local[(vthread_s10)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s10 + 48))] = (T_matmul_NN_local[((vthread_s10 + 48))] + (placeholder_shared_local[(vthread_s10)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s11 = 0; vthread_s11 < 16; ++vthread_s11) {
      placeholder_shared_local[(vthread_s11)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s11 * 128)) + (((int)threadIdx.x) * 16)) + 5))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 160))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 168))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 176))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 184))];
    for (int vthread_s12 = 0; vthread_s12 < 16; ++vthread_s12) {
      T_matmul_NN_local[(vthread_s12)] = (T_matmul_NN_local[(vthread_s12)] + (placeholder_shared_local[(vthread_s12)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s12 + 16))] = (T_matmul_NN_local[((vthread_s12 + 16))] + (placeholder_shared_local[(vthread_s12)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s12 + 32))] = (T_matmul_NN_local[((vthread_s12 + 32))] + (placeholder_shared_local[(vthread_s12)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s12 + 48))] = (T_matmul_NN_local[((vthread_s12 + 48))] + (placeholder_shared_local[(vthread_s12)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s13 = 0; vthread_s13 < 16; ++vthread_s13) {
      placeholder_shared_local[(vthread_s13)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s13 * 128)) + (((int)threadIdx.x) * 16)) + 6))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 192))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 200))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 208))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 216))];
    for (int vthread_s14 = 0; vthread_s14 < 16; ++vthread_s14) {
      T_matmul_NN_local[(vthread_s14)] = (T_matmul_NN_local[(vthread_s14)] + (placeholder_shared_local[(vthread_s14)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s14 + 16))] = (T_matmul_NN_local[((vthread_s14 + 16))] + (placeholder_shared_local[(vthread_s14)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s14 + 32))] = (T_matmul_NN_local[((vthread_s14 + 32))] + (placeholder_shared_local[(vthread_s14)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s14 + 48))] = (T_matmul_NN_local[((vthread_s14 + 48))] + (placeholder_shared_local[(vthread_s14)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s15 = 0; vthread_s15 < 16; ++vthread_s15) {
      placeholder_shared_local[(vthread_s15)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s15 * 128)) + (((int)threadIdx.x) * 16)) + 7))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 224))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 232))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 240))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 248))];
    for (int vthread_s16 = 0; vthread_s16 < 16; ++vthread_s16) {
      T_matmul_NN_local[(vthread_s16)] = (T_matmul_NN_local[(vthread_s16)] + (placeholder_shared_local[(vthread_s16)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s16 + 16))] = (T_matmul_NN_local[((vthread_s16 + 16))] + (placeholder_shared_local[(vthread_s16)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s16 + 32))] = (T_matmul_NN_local[((vthread_s16 + 32))] + (placeholder_shared_local[(vthread_s16)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s16 + 48))] = (T_matmul_NN_local[((vthread_s16 + 48))] + (placeholder_shared_local[(vthread_s16)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s17 = 0; vthread_s17 < 16; ++vthread_s17) {
      placeholder_shared_local[(vthread_s17)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s17 * 128)) + (((int)threadIdx.x) * 16)) + 8))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 256))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 264))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 272))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 280))];
    for (int vthread_s18 = 0; vthread_s18 < 16; ++vthread_s18) {
      T_matmul_NN_local[(vthread_s18)] = (T_matmul_NN_local[(vthread_s18)] + (placeholder_shared_local[(vthread_s18)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s18 + 16))] = (T_matmul_NN_local[((vthread_s18 + 16))] + (placeholder_shared_local[(vthread_s18)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s18 + 32))] = (T_matmul_NN_local[((vthread_s18 + 32))] + (placeholder_shared_local[(vthread_s18)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s18 + 48))] = (T_matmul_NN_local[((vthread_s18 + 48))] + (placeholder_shared_local[(vthread_s18)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s19 = 0; vthread_s19 < 16; ++vthread_s19) {
      placeholder_shared_local[(vthread_s19)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s19 * 128)) + (((int)threadIdx.x) * 16)) + 9))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 288))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 296))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 304))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 312))];
    for (int vthread_s20 = 0; vthread_s20 < 16; ++vthread_s20) {
      T_matmul_NN_local[(vthread_s20)] = (T_matmul_NN_local[(vthread_s20)] + (placeholder_shared_local[(vthread_s20)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s20 + 16))] = (T_matmul_NN_local[((vthread_s20 + 16))] + (placeholder_shared_local[(vthread_s20)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s20 + 32))] = (T_matmul_NN_local[((vthread_s20 + 32))] + (placeholder_shared_local[(vthread_s20)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s20 + 48))] = (T_matmul_NN_local[((vthread_s20 + 48))] + (placeholder_shared_local[(vthread_s20)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s21 = 0; vthread_s21 < 16; ++vthread_s21) {
      placeholder_shared_local[(vthread_s21)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s21 * 128)) + (((int)threadIdx.x) * 16)) + 10))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 320))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 328))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 336))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 344))];
    for (int vthread_s22 = 0; vthread_s22 < 16; ++vthread_s22) {
      T_matmul_NN_local[(vthread_s22)] = (T_matmul_NN_local[(vthread_s22)] + (placeholder_shared_local[(vthread_s22)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s22 + 16))] = (T_matmul_NN_local[((vthread_s22 + 16))] + (placeholder_shared_local[(vthread_s22)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s22 + 32))] = (T_matmul_NN_local[((vthread_s22 + 32))] + (placeholder_shared_local[(vthread_s22)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s22 + 48))] = (T_matmul_NN_local[((vthread_s22 + 48))] + (placeholder_shared_local[(vthread_s22)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s23 = 0; vthread_s23 < 16; ++vthread_s23) {
      placeholder_shared_local[(vthread_s23)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s23 * 128)) + (((int)threadIdx.x) * 16)) + 11))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 352))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 360))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 368))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 376))];
    for (int vthread_s24 = 0; vthread_s24 < 16; ++vthread_s24) {
      T_matmul_NN_local[(vthread_s24)] = (T_matmul_NN_local[(vthread_s24)] + (placeholder_shared_local[(vthread_s24)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s24 + 16))] = (T_matmul_NN_local[((vthread_s24 + 16))] + (placeholder_shared_local[(vthread_s24)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s24 + 32))] = (T_matmul_NN_local[((vthread_s24 + 32))] + (placeholder_shared_local[(vthread_s24)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s24 + 48))] = (T_matmul_NN_local[((vthread_s24 + 48))] + (placeholder_shared_local[(vthread_s24)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s25 = 0; vthread_s25 < 16; ++vthread_s25) {
      placeholder_shared_local[(vthread_s25)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s25 * 128)) + (((int)threadIdx.x) * 16)) + 12))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 384))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 392))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 400))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 408))];
    for (int vthread_s26 = 0; vthread_s26 < 16; ++vthread_s26) {
      T_matmul_NN_local[(vthread_s26)] = (T_matmul_NN_local[(vthread_s26)] + (placeholder_shared_local[(vthread_s26)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s26 + 16))] = (T_matmul_NN_local[((vthread_s26 + 16))] + (placeholder_shared_local[(vthread_s26)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s26 + 32))] = (T_matmul_NN_local[((vthread_s26 + 32))] + (placeholder_shared_local[(vthread_s26)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s26 + 48))] = (T_matmul_NN_local[((vthread_s26 + 48))] + (placeholder_shared_local[(vthread_s26)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s27 = 0; vthread_s27 < 16; ++vthread_s27) {
      placeholder_shared_local[(vthread_s27)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s27 * 128)) + (((int)threadIdx.x) * 16)) + 13))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 416))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 424))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 432))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 440))];
    for (int vthread_s28 = 0; vthread_s28 < 16; ++vthread_s28) {
      T_matmul_NN_local[(vthread_s28)] = (T_matmul_NN_local[(vthread_s28)] + (placeholder_shared_local[(vthread_s28)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s28 + 16))] = (T_matmul_NN_local[((vthread_s28 + 16))] + (placeholder_shared_local[(vthread_s28)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s28 + 32))] = (T_matmul_NN_local[((vthread_s28 + 32))] + (placeholder_shared_local[(vthread_s28)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s28 + 48))] = (T_matmul_NN_local[((vthread_s28 + 48))] + (placeholder_shared_local[(vthread_s28)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s29 = 0; vthread_s29 < 16; ++vthread_s29) {
      placeholder_shared_local[(vthread_s29)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s29 * 128)) + (((int)threadIdx.x) * 16)) + 14))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 448))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 456))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 464))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 472))];
    for (int vthread_s30 = 0; vthread_s30 < 16; ++vthread_s30) {
      T_matmul_NN_local[(vthread_s30)] = (T_matmul_NN_local[(vthread_s30)] + (placeholder_shared_local[(vthread_s30)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s30 + 16))] = (T_matmul_NN_local[((vthread_s30 + 16))] + (placeholder_shared_local[(vthread_s30)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s30 + 32))] = (T_matmul_NN_local[((vthread_s30 + 32))] + (placeholder_shared_local[(vthread_s30)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s30 + 48))] = (T_matmul_NN_local[((vthread_s30 + 48))] + (placeholder_shared_local[(vthread_s30)] * placeholder_d_shared_local[(3)]));
    }
    for (int vthread_s31 = 0; vthread_s31 < 16; ++vthread_s31) {
      placeholder_shared_local[(vthread_s31)] = placeholder_shared[((((((k_outer_outer & 1) * 2048) + (vthread_s31 * 128)) + (((int)threadIdx.x) * 16)) + 15))];
    }
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 480))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 488))];
    placeholder_d_shared_local[(2)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 496))];
    placeholder_d_shared_local[(3)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + ((int)threadIdx.y)) + 504))];
    for (int vthread_s32 = 0; vthread_s32 < 16; ++vthread_s32) {
      T_matmul_NN_local[(vthread_s32)] = (T_matmul_NN_local[(vthread_s32)] + (placeholder_shared_local[(vthread_s32)] * placeholder_d_shared_local[(0)]));
      T_matmul_NN_local[((vthread_s32 + 16))] = (T_matmul_NN_local[((vthread_s32 + 16))] + (placeholder_shared_local[(vthread_s32)] * placeholder_d_shared_local[(1)]));
      T_matmul_NN_local[((vthread_s32 + 32))] = (T_matmul_NN_local[((vthread_s32 + 32))] + (placeholder_shared_local[(vthread_s32)] * placeholder_d_shared_local[(2)]));
      T_matmul_NN_local[((vthread_s32 + 48))] = (T_matmul_NN_local[((vthread_s32 + 48))] + (placeholder_shared_local[(vthread_s32)] * placeholder_d_shared_local[(3)]));
    }
  }
  __syncthreads();
  for (int vthread_s33 = 0; vthread_s33 < 16; ++vthread_s33) {
    placeholder_shared_local1[(vthread_s33)] = placeholder_shared[((((vthread_s33 * 128) + (((int)threadIdx.x) * 16)) + 2048))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 512))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 520))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 528))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 536))];
  for (int vthread_s34 = 0; vthread_s34 < 16; ++vthread_s34) {
    T_matmul_NN_local[(vthread_s34)] = (T_matmul_NN_local[(vthread_s34)] + (placeholder_shared_local1[(vthread_s34)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s34 + 16))] = (T_matmul_NN_local[((vthread_s34 + 16))] + (placeholder_shared_local1[(vthread_s34)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s34 + 32))] = (T_matmul_NN_local[((vthread_s34 + 32))] + (placeholder_shared_local1[(vthread_s34)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s34 + 48))] = (T_matmul_NN_local[((vthread_s34 + 48))] + (placeholder_shared_local1[(vthread_s34)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s35 = 0; vthread_s35 < 16; ++vthread_s35) {
    placeholder_shared_local1[(vthread_s35)] = placeholder_shared[((((vthread_s35 * 128) + (((int)threadIdx.x) * 16)) + 2049))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 544))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 552))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 560))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 568))];
  for (int vthread_s36 = 0; vthread_s36 < 16; ++vthread_s36) {
    T_matmul_NN_local[(vthread_s36)] = (T_matmul_NN_local[(vthread_s36)] + (placeholder_shared_local1[(vthread_s36)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s36 + 16))] = (T_matmul_NN_local[((vthread_s36 + 16))] + (placeholder_shared_local1[(vthread_s36)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s36 + 32))] = (T_matmul_NN_local[((vthread_s36 + 32))] + (placeholder_shared_local1[(vthread_s36)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s36 + 48))] = (T_matmul_NN_local[((vthread_s36 + 48))] + (placeholder_shared_local1[(vthread_s36)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s37 = 0; vthread_s37 < 16; ++vthread_s37) {
    placeholder_shared_local1[(vthread_s37)] = placeholder_shared[((((vthread_s37 * 128) + (((int)threadIdx.x) * 16)) + 2050))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 576))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 584))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 592))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 600))];
  for (int vthread_s38 = 0; vthread_s38 < 16; ++vthread_s38) {
    T_matmul_NN_local[(vthread_s38)] = (T_matmul_NN_local[(vthread_s38)] + (placeholder_shared_local1[(vthread_s38)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s38 + 16))] = (T_matmul_NN_local[((vthread_s38 + 16))] + (placeholder_shared_local1[(vthread_s38)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s38 + 32))] = (T_matmul_NN_local[((vthread_s38 + 32))] + (placeholder_shared_local1[(vthread_s38)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s38 + 48))] = (T_matmul_NN_local[((vthread_s38 + 48))] + (placeholder_shared_local1[(vthread_s38)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s39 = 0; vthread_s39 < 16; ++vthread_s39) {
    placeholder_shared_local1[(vthread_s39)] = placeholder_shared[((((vthread_s39 * 128) + (((int)threadIdx.x) * 16)) + 2051))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 608))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 616))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 624))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 632))];
  for (int vthread_s40 = 0; vthread_s40 < 16; ++vthread_s40) {
    T_matmul_NN_local[(vthread_s40)] = (T_matmul_NN_local[(vthread_s40)] + (placeholder_shared_local1[(vthread_s40)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s40 + 16))] = (T_matmul_NN_local[((vthread_s40 + 16))] + (placeholder_shared_local1[(vthread_s40)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s40 + 32))] = (T_matmul_NN_local[((vthread_s40 + 32))] + (placeholder_shared_local1[(vthread_s40)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s40 + 48))] = (T_matmul_NN_local[((vthread_s40 + 48))] + (placeholder_shared_local1[(vthread_s40)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s41 = 0; vthread_s41 < 16; ++vthread_s41) {
    placeholder_shared_local1[(vthread_s41)] = placeholder_shared[((((vthread_s41 * 128) + (((int)threadIdx.x) * 16)) + 2052))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 640))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 648))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 656))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 664))];
  for (int vthread_s42 = 0; vthread_s42 < 16; ++vthread_s42) {
    T_matmul_NN_local[(vthread_s42)] = (T_matmul_NN_local[(vthread_s42)] + (placeholder_shared_local1[(vthread_s42)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s42 + 16))] = (T_matmul_NN_local[((vthread_s42 + 16))] + (placeholder_shared_local1[(vthread_s42)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s42 + 32))] = (T_matmul_NN_local[((vthread_s42 + 32))] + (placeholder_shared_local1[(vthread_s42)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s42 + 48))] = (T_matmul_NN_local[((vthread_s42 + 48))] + (placeholder_shared_local1[(vthread_s42)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s43 = 0; vthread_s43 < 16; ++vthread_s43) {
    placeholder_shared_local1[(vthread_s43)] = placeholder_shared[((((vthread_s43 * 128) + (((int)threadIdx.x) * 16)) + 2053))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 672))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 680))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 688))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 696))];
  for (int vthread_s44 = 0; vthread_s44 < 16; ++vthread_s44) {
    T_matmul_NN_local[(vthread_s44)] = (T_matmul_NN_local[(vthread_s44)] + (placeholder_shared_local1[(vthread_s44)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s44 + 16))] = (T_matmul_NN_local[((vthread_s44 + 16))] + (placeholder_shared_local1[(vthread_s44)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s44 + 32))] = (T_matmul_NN_local[((vthread_s44 + 32))] + (placeholder_shared_local1[(vthread_s44)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s44 + 48))] = (T_matmul_NN_local[((vthread_s44 + 48))] + (placeholder_shared_local1[(vthread_s44)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s45 = 0; vthread_s45 < 16; ++vthread_s45) {
    placeholder_shared_local1[(vthread_s45)] = placeholder_shared[((((vthread_s45 * 128) + (((int)threadIdx.x) * 16)) + 2054))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 704))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 712))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 720))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 728))];
  for (int vthread_s46 = 0; vthread_s46 < 16; ++vthread_s46) {
    T_matmul_NN_local[(vthread_s46)] = (T_matmul_NN_local[(vthread_s46)] + (placeholder_shared_local1[(vthread_s46)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s46 + 16))] = (T_matmul_NN_local[((vthread_s46 + 16))] + (placeholder_shared_local1[(vthread_s46)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s46 + 32))] = (T_matmul_NN_local[((vthread_s46 + 32))] + (placeholder_shared_local1[(vthread_s46)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s46 + 48))] = (T_matmul_NN_local[((vthread_s46 + 48))] + (placeholder_shared_local1[(vthread_s46)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s47 = 0; vthread_s47 < 16; ++vthread_s47) {
    placeholder_shared_local1[(vthread_s47)] = placeholder_shared[((((vthread_s47 * 128) + (((int)threadIdx.x) * 16)) + 2055))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 736))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 744))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 752))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 760))];
  for (int vthread_s48 = 0; vthread_s48 < 16; ++vthread_s48) {
    T_matmul_NN_local[(vthread_s48)] = (T_matmul_NN_local[(vthread_s48)] + (placeholder_shared_local1[(vthread_s48)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s48 + 16))] = (T_matmul_NN_local[((vthread_s48 + 16))] + (placeholder_shared_local1[(vthread_s48)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s48 + 32))] = (T_matmul_NN_local[((vthread_s48 + 32))] + (placeholder_shared_local1[(vthread_s48)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s48 + 48))] = (T_matmul_NN_local[((vthread_s48 + 48))] + (placeholder_shared_local1[(vthread_s48)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s49 = 0; vthread_s49 < 16; ++vthread_s49) {
    placeholder_shared_local1[(vthread_s49)] = placeholder_shared[((((vthread_s49 * 128) + (((int)threadIdx.x) * 16)) + 2056))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 768))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 776))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 784))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 792))];
  for (int vthread_s50 = 0; vthread_s50 < 16; ++vthread_s50) {
    T_matmul_NN_local[(vthread_s50)] = (T_matmul_NN_local[(vthread_s50)] + (placeholder_shared_local1[(vthread_s50)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s50 + 16))] = (T_matmul_NN_local[((vthread_s50 + 16))] + (placeholder_shared_local1[(vthread_s50)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s50 + 32))] = (T_matmul_NN_local[((vthread_s50 + 32))] + (placeholder_shared_local1[(vthread_s50)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s50 + 48))] = (T_matmul_NN_local[((vthread_s50 + 48))] + (placeholder_shared_local1[(vthread_s50)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s51 = 0; vthread_s51 < 16; ++vthread_s51) {
    placeholder_shared_local1[(vthread_s51)] = placeholder_shared[((((vthread_s51 * 128) + (((int)threadIdx.x) * 16)) + 2057))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 800))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 808))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 816))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 824))];
  for (int vthread_s52 = 0; vthread_s52 < 16; ++vthread_s52) {
    T_matmul_NN_local[(vthread_s52)] = (T_matmul_NN_local[(vthread_s52)] + (placeholder_shared_local1[(vthread_s52)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s52 + 16))] = (T_matmul_NN_local[((vthread_s52 + 16))] + (placeholder_shared_local1[(vthread_s52)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s52 + 32))] = (T_matmul_NN_local[((vthread_s52 + 32))] + (placeholder_shared_local1[(vthread_s52)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s52 + 48))] = (T_matmul_NN_local[((vthread_s52 + 48))] + (placeholder_shared_local1[(vthread_s52)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s53 = 0; vthread_s53 < 16; ++vthread_s53) {
    placeholder_shared_local1[(vthread_s53)] = placeholder_shared[((((vthread_s53 * 128) + (((int)threadIdx.x) * 16)) + 2058))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 832))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 840))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 848))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 856))];
  for (int vthread_s54 = 0; vthread_s54 < 16; ++vthread_s54) {
    T_matmul_NN_local[(vthread_s54)] = (T_matmul_NN_local[(vthread_s54)] + (placeholder_shared_local1[(vthread_s54)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s54 + 16))] = (T_matmul_NN_local[((vthread_s54 + 16))] + (placeholder_shared_local1[(vthread_s54)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s54 + 32))] = (T_matmul_NN_local[((vthread_s54 + 32))] + (placeholder_shared_local1[(vthread_s54)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s54 + 48))] = (T_matmul_NN_local[((vthread_s54 + 48))] + (placeholder_shared_local1[(vthread_s54)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s55 = 0; vthread_s55 < 16; ++vthread_s55) {
    placeholder_shared_local1[(vthread_s55)] = placeholder_shared[((((vthread_s55 * 128) + (((int)threadIdx.x) * 16)) + 2059))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 864))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 872))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 880))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 888))];
  for (int vthread_s56 = 0; vthread_s56 < 16; ++vthread_s56) {
    T_matmul_NN_local[(vthread_s56)] = (T_matmul_NN_local[(vthread_s56)] + (placeholder_shared_local1[(vthread_s56)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s56 + 16))] = (T_matmul_NN_local[((vthread_s56 + 16))] + (placeholder_shared_local1[(vthread_s56)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s56 + 32))] = (T_matmul_NN_local[((vthread_s56 + 32))] + (placeholder_shared_local1[(vthread_s56)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s56 + 48))] = (T_matmul_NN_local[((vthread_s56 + 48))] + (placeholder_shared_local1[(vthread_s56)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s57 = 0; vthread_s57 < 16; ++vthread_s57) {
    placeholder_shared_local1[(vthread_s57)] = placeholder_shared[((((vthread_s57 * 128) + (((int)threadIdx.x) * 16)) + 2060))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 896))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 904))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 912))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 920))];
  for (int vthread_s58 = 0; vthread_s58 < 16; ++vthread_s58) {
    T_matmul_NN_local[(vthread_s58)] = (T_matmul_NN_local[(vthread_s58)] + (placeholder_shared_local1[(vthread_s58)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s58 + 16))] = (T_matmul_NN_local[((vthread_s58 + 16))] + (placeholder_shared_local1[(vthread_s58)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s58 + 32))] = (T_matmul_NN_local[((vthread_s58 + 32))] + (placeholder_shared_local1[(vthread_s58)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s58 + 48))] = (T_matmul_NN_local[((vthread_s58 + 48))] + (placeholder_shared_local1[(vthread_s58)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s59 = 0; vthread_s59 < 16; ++vthread_s59) {
    placeholder_shared_local1[(vthread_s59)] = placeholder_shared[((((vthread_s59 * 128) + (((int)threadIdx.x) * 16)) + 2061))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 928))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 936))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 944))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 952))];
  for (int vthread_s60 = 0; vthread_s60 < 16; ++vthread_s60) {
    T_matmul_NN_local[(vthread_s60)] = (T_matmul_NN_local[(vthread_s60)] + (placeholder_shared_local1[(vthread_s60)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s60 + 16))] = (T_matmul_NN_local[((vthread_s60 + 16))] + (placeholder_shared_local1[(vthread_s60)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s60 + 32))] = (T_matmul_NN_local[((vthread_s60 + 32))] + (placeholder_shared_local1[(vthread_s60)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s60 + 48))] = (T_matmul_NN_local[((vthread_s60 + 48))] + (placeholder_shared_local1[(vthread_s60)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s61 = 0; vthread_s61 < 16; ++vthread_s61) {
    placeholder_shared_local1[(vthread_s61)] = placeholder_shared[((((vthread_s61 * 128) + (((int)threadIdx.x) * 16)) + 2062))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 960))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 968))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 976))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 984))];
  for (int vthread_s62 = 0; vthread_s62 < 16; ++vthread_s62) {
    T_matmul_NN_local[(vthread_s62)] = (T_matmul_NN_local[(vthread_s62)] + (placeholder_shared_local1[(vthread_s62)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s62 + 16))] = (T_matmul_NN_local[((vthread_s62 + 16))] + (placeholder_shared_local1[(vthread_s62)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s62 + 32))] = (T_matmul_NN_local[((vthread_s62 + 32))] + (placeholder_shared_local1[(vthread_s62)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s62 + 48))] = (T_matmul_NN_local[((vthread_s62 + 48))] + (placeholder_shared_local1[(vthread_s62)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s63 = 0; vthread_s63 < 16; ++vthread_s63) {
    placeholder_shared_local1[(vthread_s63)] = placeholder_shared[((((vthread_s63 * 128) + (((int)threadIdx.x) * 16)) + 2063))];
  }
  placeholder_d_shared_local1[(0)] = placeholder_d_shared[((((int)threadIdx.y) + 992))];
  placeholder_d_shared_local1[(1)] = placeholder_d_shared[((((int)threadIdx.y) + 1000))];
  placeholder_d_shared_local1[(2)] = placeholder_d_shared[((((int)threadIdx.y) + 1008))];
  placeholder_d_shared_local1[(3)] = placeholder_d_shared[((((int)threadIdx.y) + 1016))];
  for (int vthread_s64 = 0; vthread_s64 < 16; ++vthread_s64) {
    T_matmul_NN_local[(vthread_s64)] = (T_matmul_NN_local[(vthread_s64)] + (placeholder_shared_local1[(vthread_s64)] * placeholder_d_shared_local1[(0)]));
    T_matmul_NN_local[((vthread_s64 + 16))] = (T_matmul_NN_local[((vthread_s64 + 16))] + (placeholder_shared_local1[(vthread_s64)] * placeholder_d_shared_local1[(1)]));
    T_matmul_NN_local[((vthread_s64 + 32))] = (T_matmul_NN_local[((vthread_s64 + 32))] + (placeholder_shared_local1[(vthread_s64)] * placeholder_d_shared_local1[(2)]));
    T_matmul_NN_local[((vthread_s64 + 48))] = (T_matmul_NN_local[((vthread_s64 + 48))] + (placeholder_shared_local1[(vthread_s64)] * placeholder_d_shared_local1[(3)]));
  }
  for (int vthread_s65 = 0; vthread_s65 < 16; ++vthread_s65) {
    T_matmul_NN[((((((((int)blockIdx.x) * 524288) + (vthread_s65 * 32768)) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.y)))] = T_matmul_NN_local[(vthread_s65)];
    T_matmul_NN[(((((((((int)blockIdx.x) * 524288) + (vthread_s65 * 32768)) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.y)) + 8))] = T_matmul_NN_local[((vthread_s65 + 16))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 524288) + (vthread_s65 * 32768)) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.y)) + 16))] = T_matmul_NN_local[((vthread_s65 + 32))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 524288) + (vthread_s65 * 32768)) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.y)) + 24))] = T_matmul_NN_local[((vthread_s65 + 48))];
  }
}

