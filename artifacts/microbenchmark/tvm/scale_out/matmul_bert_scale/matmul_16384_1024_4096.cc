//256_16_1_16_16_1
//16384_1024_4096
//dim3 grid(256, 16, 1);
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
  __shared__ float placeholder_shared[1024];
  __shared__ float placeholder_d_shared[4096];
  float placeholder_shared_local[4];
  float placeholder_d_shared_local[16];
  float placeholder_shared_local1[4];
  float placeholder_d_shared_local1[16];
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
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 8) {
        placeholder_shared[(((((((int)threadIdx.y) * 32) + (ax0_inner * 8)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 4096)) + (ax0_inner * 1024)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
    for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
      if (((int)threadIdx.y) < 8) {
        placeholder_d_shared[(((((((int)threadIdx.y) * 256) + (ax1_outer * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[((((((((int)threadIdx.y) * 4096) + (((int)blockIdx.y) * 256)) + (ax1_outer * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
      }
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
    for (int ax1_outer1 = 0; ax1_outer1 < 4; ++ax1_outer1) {
      for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
        if (((int)threadIdx.y) < 8) {
          placeholder_d_shared[((((((((k_outer_outer + 1) & 1) * 2048) + (((int)threadIdx.y) * 256)) + (ax1_outer1 * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[((((((((k_outer_outer * 32768) + (((int)threadIdx.y) * 4096)) + (((int)blockIdx.y) * 256)) + (ax1_outer1 * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 32768))];
        }
      }
    }
    placeholder_shared_local[(0)] = placeholder_shared[((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 128))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 256))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 384))];
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      placeholder_d_shared_local[(ax1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax1))];
      placeholder_d_shared_local[((ax1 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax1) + 64))];
      placeholder_d_shared_local[((ax1 + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax1) + 128))];
      placeholder_d_shared_local[((ax1 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax1) + 192))];
    }
    for (int j_c = 0; j_c < 4; ++j_c) {
      T_matmul_NN_local[(j_c)] = (T_matmul_NN_local[(j_c)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c)]));
      T_matmul_NN_local[((j_c + 16))] = (T_matmul_NN_local[((j_c + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c + 4))]));
      T_matmul_NN_local[((j_c + 32))] = (T_matmul_NN_local[((j_c + 32))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c + 8))]));
      T_matmul_NN_local[((j_c + 48))] = (T_matmul_NN_local[((j_c + 48))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c + 12))]));
      T_matmul_NN_local[((j_c + 4))] = (T_matmul_NN_local[((j_c + 4))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c)]));
      T_matmul_NN_local[((j_c + 20))] = (T_matmul_NN_local[((j_c + 20))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c + 4))]));
      T_matmul_NN_local[((j_c + 36))] = (T_matmul_NN_local[((j_c + 36))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c + 8))]));
      T_matmul_NN_local[((j_c + 52))] = (T_matmul_NN_local[((j_c + 52))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c + 12))]));
      T_matmul_NN_local[((j_c + 8))] = (T_matmul_NN_local[((j_c + 8))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c)]));
      T_matmul_NN_local[((j_c + 24))] = (T_matmul_NN_local[((j_c + 24))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c + 4))]));
      T_matmul_NN_local[((j_c + 40))] = (T_matmul_NN_local[((j_c + 40))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c + 8))]));
      T_matmul_NN_local[((j_c + 56))] = (T_matmul_NN_local[((j_c + 56))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c + 12))]));
      T_matmul_NN_local[((j_c + 12))] = (T_matmul_NN_local[((j_c + 12))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c)]));
      T_matmul_NN_local[((j_c + 28))] = (T_matmul_NN_local[((j_c + 28))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c + 4))]));
      T_matmul_NN_local[((j_c + 44))] = (T_matmul_NN_local[((j_c + 44))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c + 8))]));
      T_matmul_NN_local[((j_c + 60))] = (T_matmul_NN_local[((j_c + 60))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c + 12))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 1))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 129))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 257))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 385))];
    for (int ax11 = 0; ax11 < 4; ++ax11) {
      placeholder_d_shared_local[(ax11)] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax11) + 256))];
      placeholder_d_shared_local[((ax11 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax11) + 320))];
      placeholder_d_shared_local[((ax11 + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax11) + 384))];
      placeholder_d_shared_local[((ax11 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax11) + 448))];
    }
    for (int j_c1 = 0; j_c1 < 4; ++j_c1) {
      T_matmul_NN_local[(j_c1)] = (T_matmul_NN_local[(j_c1)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c1)]));
      T_matmul_NN_local[((j_c1 + 16))] = (T_matmul_NN_local[((j_c1 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c1 + 4))]));
      T_matmul_NN_local[((j_c1 + 32))] = (T_matmul_NN_local[((j_c1 + 32))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c1 + 8))]));
      T_matmul_NN_local[((j_c1 + 48))] = (T_matmul_NN_local[((j_c1 + 48))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c1 + 12))]));
      T_matmul_NN_local[((j_c1 + 4))] = (T_matmul_NN_local[((j_c1 + 4))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c1)]));
      T_matmul_NN_local[((j_c1 + 20))] = (T_matmul_NN_local[((j_c1 + 20))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c1 + 4))]));
      T_matmul_NN_local[((j_c1 + 36))] = (T_matmul_NN_local[((j_c1 + 36))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c1 + 8))]));
      T_matmul_NN_local[((j_c1 + 52))] = (T_matmul_NN_local[((j_c1 + 52))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c1 + 12))]));
      T_matmul_NN_local[((j_c1 + 8))] = (T_matmul_NN_local[((j_c1 + 8))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c1)]));
      T_matmul_NN_local[((j_c1 + 24))] = (T_matmul_NN_local[((j_c1 + 24))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c1 + 4))]));
      T_matmul_NN_local[((j_c1 + 40))] = (T_matmul_NN_local[((j_c1 + 40))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c1 + 8))]));
      T_matmul_NN_local[((j_c1 + 56))] = (T_matmul_NN_local[((j_c1 + 56))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c1 + 12))]));
      T_matmul_NN_local[((j_c1 + 12))] = (T_matmul_NN_local[((j_c1 + 12))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c1)]));
      T_matmul_NN_local[((j_c1 + 28))] = (T_matmul_NN_local[((j_c1 + 28))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c1 + 4))]));
      T_matmul_NN_local[((j_c1 + 44))] = (T_matmul_NN_local[((j_c1 + 44))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c1 + 8))]));
      T_matmul_NN_local[((j_c1 + 60))] = (T_matmul_NN_local[((j_c1 + 60))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c1 + 12))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 2))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 130))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 258))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 386))];
    for (int ax12 = 0; ax12 < 4; ++ax12) {
      placeholder_d_shared_local[(ax12)] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax12) + 512))];
      placeholder_d_shared_local[((ax12 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax12) + 576))];
      placeholder_d_shared_local[((ax12 + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax12) + 640))];
      placeholder_d_shared_local[((ax12 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax12) + 704))];
    }
    for (int j_c2 = 0; j_c2 < 4; ++j_c2) {
      T_matmul_NN_local[(j_c2)] = (T_matmul_NN_local[(j_c2)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c2)]));
      T_matmul_NN_local[((j_c2 + 16))] = (T_matmul_NN_local[((j_c2 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c2 + 4))]));
      T_matmul_NN_local[((j_c2 + 32))] = (T_matmul_NN_local[((j_c2 + 32))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c2 + 8))]));
      T_matmul_NN_local[((j_c2 + 48))] = (T_matmul_NN_local[((j_c2 + 48))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c2 + 12))]));
      T_matmul_NN_local[((j_c2 + 4))] = (T_matmul_NN_local[((j_c2 + 4))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c2)]));
      T_matmul_NN_local[((j_c2 + 20))] = (T_matmul_NN_local[((j_c2 + 20))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c2 + 4))]));
      T_matmul_NN_local[((j_c2 + 36))] = (T_matmul_NN_local[((j_c2 + 36))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c2 + 8))]));
      T_matmul_NN_local[((j_c2 + 52))] = (T_matmul_NN_local[((j_c2 + 52))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c2 + 12))]));
      T_matmul_NN_local[((j_c2 + 8))] = (T_matmul_NN_local[((j_c2 + 8))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c2)]));
      T_matmul_NN_local[((j_c2 + 24))] = (T_matmul_NN_local[((j_c2 + 24))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c2 + 4))]));
      T_matmul_NN_local[((j_c2 + 40))] = (T_matmul_NN_local[((j_c2 + 40))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c2 + 8))]));
      T_matmul_NN_local[((j_c2 + 56))] = (T_matmul_NN_local[((j_c2 + 56))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c2 + 12))]));
      T_matmul_NN_local[((j_c2 + 12))] = (T_matmul_NN_local[((j_c2 + 12))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c2)]));
      T_matmul_NN_local[((j_c2 + 28))] = (T_matmul_NN_local[((j_c2 + 28))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c2 + 4))]));
      T_matmul_NN_local[((j_c2 + 44))] = (T_matmul_NN_local[((j_c2 + 44))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c2 + 8))]));
      T_matmul_NN_local[((j_c2 + 60))] = (T_matmul_NN_local[((j_c2 + 60))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c2 + 12))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 3))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 131))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 259))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 387))];
    for (int ax13 = 0; ax13 < 4; ++ax13) {
      placeholder_d_shared_local[(ax13)] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax13) + 768))];
      placeholder_d_shared_local[((ax13 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax13) + 832))];
      placeholder_d_shared_local[((ax13 + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax13) + 896))];
      placeholder_d_shared_local[((ax13 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax13) + 960))];
    }
    for (int j_c3 = 0; j_c3 < 4; ++j_c3) {
      T_matmul_NN_local[(j_c3)] = (T_matmul_NN_local[(j_c3)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c3)]));
      T_matmul_NN_local[((j_c3 + 16))] = (T_matmul_NN_local[((j_c3 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c3 + 4))]));
      T_matmul_NN_local[((j_c3 + 32))] = (T_matmul_NN_local[((j_c3 + 32))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c3 + 8))]));
      T_matmul_NN_local[((j_c3 + 48))] = (T_matmul_NN_local[((j_c3 + 48))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c3 + 12))]));
      T_matmul_NN_local[((j_c3 + 4))] = (T_matmul_NN_local[((j_c3 + 4))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c3)]));
      T_matmul_NN_local[((j_c3 + 20))] = (T_matmul_NN_local[((j_c3 + 20))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c3 + 4))]));
      T_matmul_NN_local[((j_c3 + 36))] = (T_matmul_NN_local[((j_c3 + 36))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c3 + 8))]));
      T_matmul_NN_local[((j_c3 + 52))] = (T_matmul_NN_local[((j_c3 + 52))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c3 + 12))]));
      T_matmul_NN_local[((j_c3 + 8))] = (T_matmul_NN_local[((j_c3 + 8))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c3)]));
      T_matmul_NN_local[((j_c3 + 24))] = (T_matmul_NN_local[((j_c3 + 24))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c3 + 4))]));
      T_matmul_NN_local[((j_c3 + 40))] = (T_matmul_NN_local[((j_c3 + 40))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c3 + 8))]));
      T_matmul_NN_local[((j_c3 + 56))] = (T_matmul_NN_local[((j_c3 + 56))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c3 + 12))]));
      T_matmul_NN_local[((j_c3 + 12))] = (T_matmul_NN_local[((j_c3 + 12))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c3)]));
      T_matmul_NN_local[((j_c3 + 28))] = (T_matmul_NN_local[((j_c3 + 28))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c3 + 4))]));
      T_matmul_NN_local[((j_c3 + 44))] = (T_matmul_NN_local[((j_c3 + 44))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c3 + 8))]));
      T_matmul_NN_local[((j_c3 + 60))] = (T_matmul_NN_local[((j_c3 + 60))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c3 + 12))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 4))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 132))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 260))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 388))];
    for (int ax14 = 0; ax14 < 4; ++ax14) {
      placeholder_d_shared_local[(ax14)] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax14) + 1024))];
      placeholder_d_shared_local[((ax14 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax14) + 1088))];
      placeholder_d_shared_local[((ax14 + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax14) + 1152))];
      placeholder_d_shared_local[((ax14 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax14) + 1216))];
    }
    for (int j_c4 = 0; j_c4 < 4; ++j_c4) {
      T_matmul_NN_local[(j_c4)] = (T_matmul_NN_local[(j_c4)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c4)]));
      T_matmul_NN_local[((j_c4 + 16))] = (T_matmul_NN_local[((j_c4 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c4 + 4))]));
      T_matmul_NN_local[((j_c4 + 32))] = (T_matmul_NN_local[((j_c4 + 32))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c4 + 8))]));
      T_matmul_NN_local[((j_c4 + 48))] = (T_matmul_NN_local[((j_c4 + 48))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c4 + 12))]));
      T_matmul_NN_local[((j_c4 + 4))] = (T_matmul_NN_local[((j_c4 + 4))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c4)]));
      T_matmul_NN_local[((j_c4 + 20))] = (T_matmul_NN_local[((j_c4 + 20))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c4 + 4))]));
      T_matmul_NN_local[((j_c4 + 36))] = (T_matmul_NN_local[((j_c4 + 36))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c4 + 8))]));
      T_matmul_NN_local[((j_c4 + 52))] = (T_matmul_NN_local[((j_c4 + 52))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c4 + 12))]));
      T_matmul_NN_local[((j_c4 + 8))] = (T_matmul_NN_local[((j_c4 + 8))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c4)]));
      T_matmul_NN_local[((j_c4 + 24))] = (T_matmul_NN_local[((j_c4 + 24))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c4 + 4))]));
      T_matmul_NN_local[((j_c4 + 40))] = (T_matmul_NN_local[((j_c4 + 40))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c4 + 8))]));
      T_matmul_NN_local[((j_c4 + 56))] = (T_matmul_NN_local[((j_c4 + 56))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c4 + 12))]));
      T_matmul_NN_local[((j_c4 + 12))] = (T_matmul_NN_local[((j_c4 + 12))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c4)]));
      T_matmul_NN_local[((j_c4 + 28))] = (T_matmul_NN_local[((j_c4 + 28))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c4 + 4))]));
      T_matmul_NN_local[((j_c4 + 44))] = (T_matmul_NN_local[((j_c4 + 44))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c4 + 8))]));
      T_matmul_NN_local[((j_c4 + 60))] = (T_matmul_NN_local[((j_c4 + 60))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c4 + 12))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 5))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 133))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 261))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 389))];
    for (int ax15 = 0; ax15 < 4; ++ax15) {
      placeholder_d_shared_local[(ax15)] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax15) + 1280))];
      placeholder_d_shared_local[((ax15 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax15) + 1344))];
      placeholder_d_shared_local[((ax15 + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax15) + 1408))];
      placeholder_d_shared_local[((ax15 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax15) + 1472))];
    }
    for (int j_c5 = 0; j_c5 < 4; ++j_c5) {
      T_matmul_NN_local[(j_c5)] = (T_matmul_NN_local[(j_c5)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c5)]));
      T_matmul_NN_local[((j_c5 + 16))] = (T_matmul_NN_local[((j_c5 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c5 + 4))]));
      T_matmul_NN_local[((j_c5 + 32))] = (T_matmul_NN_local[((j_c5 + 32))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c5 + 8))]));
      T_matmul_NN_local[((j_c5 + 48))] = (T_matmul_NN_local[((j_c5 + 48))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c5 + 12))]));
      T_matmul_NN_local[((j_c5 + 4))] = (T_matmul_NN_local[((j_c5 + 4))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c5)]));
      T_matmul_NN_local[((j_c5 + 20))] = (T_matmul_NN_local[((j_c5 + 20))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c5 + 4))]));
      T_matmul_NN_local[((j_c5 + 36))] = (T_matmul_NN_local[((j_c5 + 36))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c5 + 8))]));
      T_matmul_NN_local[((j_c5 + 52))] = (T_matmul_NN_local[((j_c5 + 52))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c5 + 12))]));
      T_matmul_NN_local[((j_c5 + 8))] = (T_matmul_NN_local[((j_c5 + 8))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c5)]));
      T_matmul_NN_local[((j_c5 + 24))] = (T_matmul_NN_local[((j_c5 + 24))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c5 + 4))]));
      T_matmul_NN_local[((j_c5 + 40))] = (T_matmul_NN_local[((j_c5 + 40))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c5 + 8))]));
      T_matmul_NN_local[((j_c5 + 56))] = (T_matmul_NN_local[((j_c5 + 56))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c5 + 12))]));
      T_matmul_NN_local[((j_c5 + 12))] = (T_matmul_NN_local[((j_c5 + 12))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c5)]));
      T_matmul_NN_local[((j_c5 + 28))] = (T_matmul_NN_local[((j_c5 + 28))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c5 + 4))]));
      T_matmul_NN_local[((j_c5 + 44))] = (T_matmul_NN_local[((j_c5 + 44))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c5 + 8))]));
      T_matmul_NN_local[((j_c5 + 60))] = (T_matmul_NN_local[((j_c5 + 60))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c5 + 12))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 6))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 134))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 262))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 390))];
    for (int ax16 = 0; ax16 < 4; ++ax16) {
      placeholder_d_shared_local[(ax16)] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax16) + 1536))];
      placeholder_d_shared_local[((ax16 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax16) + 1600))];
      placeholder_d_shared_local[((ax16 + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax16) + 1664))];
      placeholder_d_shared_local[((ax16 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax16) + 1728))];
    }
    for (int j_c6 = 0; j_c6 < 4; ++j_c6) {
      T_matmul_NN_local[(j_c6)] = (T_matmul_NN_local[(j_c6)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c6)]));
      T_matmul_NN_local[((j_c6 + 16))] = (T_matmul_NN_local[((j_c6 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c6 + 4))]));
      T_matmul_NN_local[((j_c6 + 32))] = (T_matmul_NN_local[((j_c6 + 32))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c6 + 8))]));
      T_matmul_NN_local[((j_c6 + 48))] = (T_matmul_NN_local[((j_c6 + 48))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c6 + 12))]));
      T_matmul_NN_local[((j_c6 + 4))] = (T_matmul_NN_local[((j_c6 + 4))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c6)]));
      T_matmul_NN_local[((j_c6 + 20))] = (T_matmul_NN_local[((j_c6 + 20))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c6 + 4))]));
      T_matmul_NN_local[((j_c6 + 36))] = (T_matmul_NN_local[((j_c6 + 36))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c6 + 8))]));
      T_matmul_NN_local[((j_c6 + 52))] = (T_matmul_NN_local[((j_c6 + 52))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c6 + 12))]));
      T_matmul_NN_local[((j_c6 + 8))] = (T_matmul_NN_local[((j_c6 + 8))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c6)]));
      T_matmul_NN_local[((j_c6 + 24))] = (T_matmul_NN_local[((j_c6 + 24))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c6 + 4))]));
      T_matmul_NN_local[((j_c6 + 40))] = (T_matmul_NN_local[((j_c6 + 40))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c6 + 8))]));
      T_matmul_NN_local[((j_c6 + 56))] = (T_matmul_NN_local[((j_c6 + 56))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c6 + 12))]));
      T_matmul_NN_local[((j_c6 + 12))] = (T_matmul_NN_local[((j_c6 + 12))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c6)]));
      T_matmul_NN_local[((j_c6 + 28))] = (T_matmul_NN_local[((j_c6 + 28))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c6 + 4))]));
      T_matmul_NN_local[((j_c6 + 44))] = (T_matmul_NN_local[((j_c6 + 44))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c6 + 8))]));
      T_matmul_NN_local[((j_c6 + 60))] = (T_matmul_NN_local[((j_c6 + 60))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c6 + 12))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 7))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 135))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 263))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.x) * 8)) + 391))];
    for (int ax17 = 0; ax17 < 4; ++ax17) {
      placeholder_d_shared_local[(ax17)] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax17) + 1792))];
      placeholder_d_shared_local[((ax17 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax17) + 1856))];
      placeholder_d_shared_local[((ax17 + 8))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax17) + 1920))];
      placeholder_d_shared_local[((ax17 + 12))] = placeholder_d_shared[((((((k_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 4)) + ax17) + 1984))];
    }
    for (int j_c7 = 0; j_c7 < 4; ++j_c7) {
      T_matmul_NN_local[(j_c7)] = (T_matmul_NN_local[(j_c7)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c7)]));
      T_matmul_NN_local[((j_c7 + 16))] = (T_matmul_NN_local[((j_c7 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c7 + 4))]));
      T_matmul_NN_local[((j_c7 + 32))] = (T_matmul_NN_local[((j_c7 + 32))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c7 + 8))]));
      T_matmul_NN_local[((j_c7 + 48))] = (T_matmul_NN_local[((j_c7 + 48))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c7 + 12))]));
      T_matmul_NN_local[((j_c7 + 4))] = (T_matmul_NN_local[((j_c7 + 4))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c7)]));
      T_matmul_NN_local[((j_c7 + 20))] = (T_matmul_NN_local[((j_c7 + 20))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c7 + 4))]));
      T_matmul_NN_local[((j_c7 + 36))] = (T_matmul_NN_local[((j_c7 + 36))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c7 + 8))]));
      T_matmul_NN_local[((j_c7 + 52))] = (T_matmul_NN_local[((j_c7 + 52))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c7 + 12))]));
      T_matmul_NN_local[((j_c7 + 8))] = (T_matmul_NN_local[((j_c7 + 8))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c7)]));
      T_matmul_NN_local[((j_c7 + 24))] = (T_matmul_NN_local[((j_c7 + 24))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c7 + 4))]));
      T_matmul_NN_local[((j_c7 + 40))] = (T_matmul_NN_local[((j_c7 + 40))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c7 + 8))]));
      T_matmul_NN_local[((j_c7 + 56))] = (T_matmul_NN_local[((j_c7 + 56))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c7 + 12))]));
      T_matmul_NN_local[((j_c7 + 12))] = (T_matmul_NN_local[((j_c7 + 12))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c7)]));
      T_matmul_NN_local[((j_c7 + 28))] = (T_matmul_NN_local[((j_c7 + 28))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c7 + 4))]));
      T_matmul_NN_local[((j_c7 + 44))] = (T_matmul_NN_local[((j_c7 + 44))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c7 + 8))]));
      T_matmul_NN_local[((j_c7 + 60))] = (T_matmul_NN_local[((j_c7 + 60))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c7 + 12))]));
    }
  }
  __syncthreads();
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 512))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 640))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 768))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 896))];
  for (int ax18 = 0; ax18 < 4; ++ax18) {
    placeholder_d_shared_local1[(ax18)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax18) + 2048))];
    placeholder_d_shared_local1[((ax18 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax18) + 2112))];
    placeholder_d_shared_local1[((ax18 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax18) + 2176))];
    placeholder_d_shared_local1[((ax18 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax18) + 2240))];
  }
  for (int j_c8 = 0; j_c8 < 4; ++j_c8) {
    T_matmul_NN_local[(j_c8)] = (T_matmul_NN_local[(j_c8)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c8)]));
    T_matmul_NN_local[((j_c8 + 16))] = (T_matmul_NN_local[((j_c8 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c8 + 4))]));
    T_matmul_NN_local[((j_c8 + 32))] = (T_matmul_NN_local[((j_c8 + 32))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c8 + 8))]));
    T_matmul_NN_local[((j_c8 + 48))] = (T_matmul_NN_local[((j_c8 + 48))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c8 + 12))]));
    T_matmul_NN_local[((j_c8 + 4))] = (T_matmul_NN_local[((j_c8 + 4))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c8)]));
    T_matmul_NN_local[((j_c8 + 20))] = (T_matmul_NN_local[((j_c8 + 20))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c8 + 4))]));
    T_matmul_NN_local[((j_c8 + 36))] = (T_matmul_NN_local[((j_c8 + 36))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c8 + 8))]));
    T_matmul_NN_local[((j_c8 + 52))] = (T_matmul_NN_local[((j_c8 + 52))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c8 + 12))]));
    T_matmul_NN_local[((j_c8 + 8))] = (T_matmul_NN_local[((j_c8 + 8))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c8)]));
    T_matmul_NN_local[((j_c8 + 24))] = (T_matmul_NN_local[((j_c8 + 24))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c8 + 4))]));
    T_matmul_NN_local[((j_c8 + 40))] = (T_matmul_NN_local[((j_c8 + 40))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c8 + 8))]));
    T_matmul_NN_local[((j_c8 + 56))] = (T_matmul_NN_local[((j_c8 + 56))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c8 + 12))]));
    T_matmul_NN_local[((j_c8 + 12))] = (T_matmul_NN_local[((j_c8 + 12))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c8)]));
    T_matmul_NN_local[((j_c8 + 28))] = (T_matmul_NN_local[((j_c8 + 28))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c8 + 4))]));
    T_matmul_NN_local[((j_c8 + 44))] = (T_matmul_NN_local[((j_c8 + 44))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c8 + 8))]));
    T_matmul_NN_local[((j_c8 + 60))] = (T_matmul_NN_local[((j_c8 + 60))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c8 + 12))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 513))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 641))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 769))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 897))];
  for (int ax19 = 0; ax19 < 4; ++ax19) {
    placeholder_d_shared_local1[(ax19)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax19) + 2304))];
    placeholder_d_shared_local1[((ax19 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax19) + 2368))];
    placeholder_d_shared_local1[((ax19 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax19) + 2432))];
    placeholder_d_shared_local1[((ax19 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax19) + 2496))];
  }
  for (int j_c9 = 0; j_c9 < 4; ++j_c9) {
    T_matmul_NN_local[(j_c9)] = (T_matmul_NN_local[(j_c9)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c9)]));
    T_matmul_NN_local[((j_c9 + 16))] = (T_matmul_NN_local[((j_c9 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c9 + 4))]));
    T_matmul_NN_local[((j_c9 + 32))] = (T_matmul_NN_local[((j_c9 + 32))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c9 + 8))]));
    T_matmul_NN_local[((j_c9 + 48))] = (T_matmul_NN_local[((j_c9 + 48))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c9 + 12))]));
    T_matmul_NN_local[((j_c9 + 4))] = (T_matmul_NN_local[((j_c9 + 4))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c9)]));
    T_matmul_NN_local[((j_c9 + 20))] = (T_matmul_NN_local[((j_c9 + 20))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c9 + 4))]));
    T_matmul_NN_local[((j_c9 + 36))] = (T_matmul_NN_local[((j_c9 + 36))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c9 + 8))]));
    T_matmul_NN_local[((j_c9 + 52))] = (T_matmul_NN_local[((j_c9 + 52))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c9 + 12))]));
    T_matmul_NN_local[((j_c9 + 8))] = (T_matmul_NN_local[((j_c9 + 8))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c9)]));
    T_matmul_NN_local[((j_c9 + 24))] = (T_matmul_NN_local[((j_c9 + 24))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c9 + 4))]));
    T_matmul_NN_local[((j_c9 + 40))] = (T_matmul_NN_local[((j_c9 + 40))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c9 + 8))]));
    T_matmul_NN_local[((j_c9 + 56))] = (T_matmul_NN_local[((j_c9 + 56))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c9 + 12))]));
    T_matmul_NN_local[((j_c9 + 12))] = (T_matmul_NN_local[((j_c9 + 12))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c9)]));
    T_matmul_NN_local[((j_c9 + 28))] = (T_matmul_NN_local[((j_c9 + 28))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c9 + 4))]));
    T_matmul_NN_local[((j_c9 + 44))] = (T_matmul_NN_local[((j_c9 + 44))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c9 + 8))]));
    T_matmul_NN_local[((j_c9 + 60))] = (T_matmul_NN_local[((j_c9 + 60))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c9 + 12))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 514))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 642))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 770))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 898))];
  for (int ax110 = 0; ax110 < 4; ++ax110) {
    placeholder_d_shared_local1[(ax110)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax110) + 2560))];
    placeholder_d_shared_local1[((ax110 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax110) + 2624))];
    placeholder_d_shared_local1[((ax110 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax110) + 2688))];
    placeholder_d_shared_local1[((ax110 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax110) + 2752))];
  }
  for (int j_c10 = 0; j_c10 < 4; ++j_c10) {
    T_matmul_NN_local[(j_c10)] = (T_matmul_NN_local[(j_c10)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c10)]));
    T_matmul_NN_local[((j_c10 + 16))] = (T_matmul_NN_local[((j_c10 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c10 + 4))]));
    T_matmul_NN_local[((j_c10 + 32))] = (T_matmul_NN_local[((j_c10 + 32))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c10 + 8))]));
    T_matmul_NN_local[((j_c10 + 48))] = (T_matmul_NN_local[((j_c10 + 48))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c10 + 12))]));
    T_matmul_NN_local[((j_c10 + 4))] = (T_matmul_NN_local[((j_c10 + 4))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c10)]));
    T_matmul_NN_local[((j_c10 + 20))] = (T_matmul_NN_local[((j_c10 + 20))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c10 + 4))]));
    T_matmul_NN_local[((j_c10 + 36))] = (T_matmul_NN_local[((j_c10 + 36))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c10 + 8))]));
    T_matmul_NN_local[((j_c10 + 52))] = (T_matmul_NN_local[((j_c10 + 52))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c10 + 12))]));
    T_matmul_NN_local[((j_c10 + 8))] = (T_matmul_NN_local[((j_c10 + 8))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c10)]));
    T_matmul_NN_local[((j_c10 + 24))] = (T_matmul_NN_local[((j_c10 + 24))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c10 + 4))]));
    T_matmul_NN_local[((j_c10 + 40))] = (T_matmul_NN_local[((j_c10 + 40))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c10 + 8))]));
    T_matmul_NN_local[((j_c10 + 56))] = (T_matmul_NN_local[((j_c10 + 56))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c10 + 12))]));
    T_matmul_NN_local[((j_c10 + 12))] = (T_matmul_NN_local[((j_c10 + 12))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c10)]));
    T_matmul_NN_local[((j_c10 + 28))] = (T_matmul_NN_local[((j_c10 + 28))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c10 + 4))]));
    T_matmul_NN_local[((j_c10 + 44))] = (T_matmul_NN_local[((j_c10 + 44))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c10 + 8))]));
    T_matmul_NN_local[((j_c10 + 60))] = (T_matmul_NN_local[((j_c10 + 60))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c10 + 12))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 515))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 643))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 771))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 899))];
  for (int ax111 = 0; ax111 < 4; ++ax111) {
    placeholder_d_shared_local1[(ax111)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax111) + 2816))];
    placeholder_d_shared_local1[((ax111 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax111) + 2880))];
    placeholder_d_shared_local1[((ax111 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax111) + 2944))];
    placeholder_d_shared_local1[((ax111 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax111) + 3008))];
  }
  for (int j_c11 = 0; j_c11 < 4; ++j_c11) {
    T_matmul_NN_local[(j_c11)] = (T_matmul_NN_local[(j_c11)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c11)]));
    T_matmul_NN_local[((j_c11 + 16))] = (T_matmul_NN_local[((j_c11 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c11 + 4))]));
    T_matmul_NN_local[((j_c11 + 32))] = (T_matmul_NN_local[((j_c11 + 32))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c11 + 8))]));
    T_matmul_NN_local[((j_c11 + 48))] = (T_matmul_NN_local[((j_c11 + 48))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c11 + 12))]));
    T_matmul_NN_local[((j_c11 + 4))] = (T_matmul_NN_local[((j_c11 + 4))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c11)]));
    T_matmul_NN_local[((j_c11 + 20))] = (T_matmul_NN_local[((j_c11 + 20))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c11 + 4))]));
    T_matmul_NN_local[((j_c11 + 36))] = (T_matmul_NN_local[((j_c11 + 36))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c11 + 8))]));
    T_matmul_NN_local[((j_c11 + 52))] = (T_matmul_NN_local[((j_c11 + 52))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c11 + 12))]));
    T_matmul_NN_local[((j_c11 + 8))] = (T_matmul_NN_local[((j_c11 + 8))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c11)]));
    T_matmul_NN_local[((j_c11 + 24))] = (T_matmul_NN_local[((j_c11 + 24))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c11 + 4))]));
    T_matmul_NN_local[((j_c11 + 40))] = (T_matmul_NN_local[((j_c11 + 40))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c11 + 8))]));
    T_matmul_NN_local[((j_c11 + 56))] = (T_matmul_NN_local[((j_c11 + 56))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c11 + 12))]));
    T_matmul_NN_local[((j_c11 + 12))] = (T_matmul_NN_local[((j_c11 + 12))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c11)]));
    T_matmul_NN_local[((j_c11 + 28))] = (T_matmul_NN_local[((j_c11 + 28))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c11 + 4))]));
    T_matmul_NN_local[((j_c11 + 44))] = (T_matmul_NN_local[((j_c11 + 44))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c11 + 8))]));
    T_matmul_NN_local[((j_c11 + 60))] = (T_matmul_NN_local[((j_c11 + 60))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c11 + 12))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 516))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 644))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 772))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 900))];
  for (int ax112 = 0; ax112 < 4; ++ax112) {
    placeholder_d_shared_local1[(ax112)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax112) + 3072))];
    placeholder_d_shared_local1[((ax112 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax112) + 3136))];
    placeholder_d_shared_local1[((ax112 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax112) + 3200))];
    placeholder_d_shared_local1[((ax112 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax112) + 3264))];
  }
  for (int j_c12 = 0; j_c12 < 4; ++j_c12) {
    T_matmul_NN_local[(j_c12)] = (T_matmul_NN_local[(j_c12)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c12)]));
    T_matmul_NN_local[((j_c12 + 16))] = (T_matmul_NN_local[((j_c12 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c12 + 4))]));
    T_matmul_NN_local[((j_c12 + 32))] = (T_matmul_NN_local[((j_c12 + 32))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c12 + 8))]));
    T_matmul_NN_local[((j_c12 + 48))] = (T_matmul_NN_local[((j_c12 + 48))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c12 + 12))]));
    T_matmul_NN_local[((j_c12 + 4))] = (T_matmul_NN_local[((j_c12 + 4))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c12)]));
    T_matmul_NN_local[((j_c12 + 20))] = (T_matmul_NN_local[((j_c12 + 20))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c12 + 4))]));
    T_matmul_NN_local[((j_c12 + 36))] = (T_matmul_NN_local[((j_c12 + 36))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c12 + 8))]));
    T_matmul_NN_local[((j_c12 + 52))] = (T_matmul_NN_local[((j_c12 + 52))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c12 + 12))]));
    T_matmul_NN_local[((j_c12 + 8))] = (T_matmul_NN_local[((j_c12 + 8))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c12)]));
    T_matmul_NN_local[((j_c12 + 24))] = (T_matmul_NN_local[((j_c12 + 24))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c12 + 4))]));
    T_matmul_NN_local[((j_c12 + 40))] = (T_matmul_NN_local[((j_c12 + 40))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c12 + 8))]));
    T_matmul_NN_local[((j_c12 + 56))] = (T_matmul_NN_local[((j_c12 + 56))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c12 + 12))]));
    T_matmul_NN_local[((j_c12 + 12))] = (T_matmul_NN_local[((j_c12 + 12))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c12)]));
    T_matmul_NN_local[((j_c12 + 28))] = (T_matmul_NN_local[((j_c12 + 28))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c12 + 4))]));
    T_matmul_NN_local[((j_c12 + 44))] = (T_matmul_NN_local[((j_c12 + 44))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c12 + 8))]));
    T_matmul_NN_local[((j_c12 + 60))] = (T_matmul_NN_local[((j_c12 + 60))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c12 + 12))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 517))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 645))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 773))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 901))];
  for (int ax113 = 0; ax113 < 4; ++ax113) {
    placeholder_d_shared_local1[(ax113)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax113) + 3328))];
    placeholder_d_shared_local1[((ax113 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax113) + 3392))];
    placeholder_d_shared_local1[((ax113 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax113) + 3456))];
    placeholder_d_shared_local1[((ax113 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax113) + 3520))];
  }
  for (int j_c13 = 0; j_c13 < 4; ++j_c13) {
    T_matmul_NN_local[(j_c13)] = (T_matmul_NN_local[(j_c13)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c13)]));
    T_matmul_NN_local[((j_c13 + 16))] = (T_matmul_NN_local[((j_c13 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c13 + 4))]));
    T_matmul_NN_local[((j_c13 + 32))] = (T_matmul_NN_local[((j_c13 + 32))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c13 + 8))]));
    T_matmul_NN_local[((j_c13 + 48))] = (T_matmul_NN_local[((j_c13 + 48))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c13 + 12))]));
    T_matmul_NN_local[((j_c13 + 4))] = (T_matmul_NN_local[((j_c13 + 4))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c13)]));
    T_matmul_NN_local[((j_c13 + 20))] = (T_matmul_NN_local[((j_c13 + 20))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c13 + 4))]));
    T_matmul_NN_local[((j_c13 + 36))] = (T_matmul_NN_local[((j_c13 + 36))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c13 + 8))]));
    T_matmul_NN_local[((j_c13 + 52))] = (T_matmul_NN_local[((j_c13 + 52))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c13 + 12))]));
    T_matmul_NN_local[((j_c13 + 8))] = (T_matmul_NN_local[((j_c13 + 8))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c13)]));
    T_matmul_NN_local[((j_c13 + 24))] = (T_matmul_NN_local[((j_c13 + 24))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c13 + 4))]));
    T_matmul_NN_local[((j_c13 + 40))] = (T_matmul_NN_local[((j_c13 + 40))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c13 + 8))]));
    T_matmul_NN_local[((j_c13 + 56))] = (T_matmul_NN_local[((j_c13 + 56))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c13 + 12))]));
    T_matmul_NN_local[((j_c13 + 12))] = (T_matmul_NN_local[((j_c13 + 12))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c13)]));
    T_matmul_NN_local[((j_c13 + 28))] = (T_matmul_NN_local[((j_c13 + 28))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c13 + 4))]));
    T_matmul_NN_local[((j_c13 + 44))] = (T_matmul_NN_local[((j_c13 + 44))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c13 + 8))]));
    T_matmul_NN_local[((j_c13 + 60))] = (T_matmul_NN_local[((j_c13 + 60))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c13 + 12))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 518))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 646))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 774))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 902))];
  for (int ax114 = 0; ax114 < 4; ++ax114) {
    placeholder_d_shared_local1[(ax114)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax114) + 3584))];
    placeholder_d_shared_local1[((ax114 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax114) + 3648))];
    placeholder_d_shared_local1[((ax114 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax114) + 3712))];
    placeholder_d_shared_local1[((ax114 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax114) + 3776))];
  }
  for (int j_c14 = 0; j_c14 < 4; ++j_c14) {
    T_matmul_NN_local[(j_c14)] = (T_matmul_NN_local[(j_c14)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c14)]));
    T_matmul_NN_local[((j_c14 + 16))] = (T_matmul_NN_local[((j_c14 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c14 + 4))]));
    T_matmul_NN_local[((j_c14 + 32))] = (T_matmul_NN_local[((j_c14 + 32))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c14 + 8))]));
    T_matmul_NN_local[((j_c14 + 48))] = (T_matmul_NN_local[((j_c14 + 48))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c14 + 12))]));
    T_matmul_NN_local[((j_c14 + 4))] = (T_matmul_NN_local[((j_c14 + 4))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c14)]));
    T_matmul_NN_local[((j_c14 + 20))] = (T_matmul_NN_local[((j_c14 + 20))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c14 + 4))]));
    T_matmul_NN_local[((j_c14 + 36))] = (T_matmul_NN_local[((j_c14 + 36))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c14 + 8))]));
    T_matmul_NN_local[((j_c14 + 52))] = (T_matmul_NN_local[((j_c14 + 52))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c14 + 12))]));
    T_matmul_NN_local[((j_c14 + 8))] = (T_matmul_NN_local[((j_c14 + 8))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c14)]));
    T_matmul_NN_local[((j_c14 + 24))] = (T_matmul_NN_local[((j_c14 + 24))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c14 + 4))]));
    T_matmul_NN_local[((j_c14 + 40))] = (T_matmul_NN_local[((j_c14 + 40))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c14 + 8))]));
    T_matmul_NN_local[((j_c14 + 56))] = (T_matmul_NN_local[((j_c14 + 56))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c14 + 12))]));
    T_matmul_NN_local[((j_c14 + 12))] = (T_matmul_NN_local[((j_c14 + 12))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c14)]));
    T_matmul_NN_local[((j_c14 + 28))] = (T_matmul_NN_local[((j_c14 + 28))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c14 + 4))]));
    T_matmul_NN_local[((j_c14 + 44))] = (T_matmul_NN_local[((j_c14 + 44))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c14 + 8))]));
    T_matmul_NN_local[((j_c14 + 60))] = (T_matmul_NN_local[((j_c14 + 60))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c14 + 12))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 519))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 647))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 775))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 8) + 903))];
  for (int ax115 = 0; ax115 < 4; ++ax115) {
    placeholder_d_shared_local1[(ax115)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax115) + 3840))];
    placeholder_d_shared_local1[((ax115 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax115) + 3904))];
    placeholder_d_shared_local1[((ax115 + 8))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax115) + 3968))];
    placeholder_d_shared_local1[((ax115 + 12))] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax115) + 4032))];
  }
  for (int j_c15 = 0; j_c15 < 4; ++j_c15) {
    T_matmul_NN_local[(j_c15)] = (T_matmul_NN_local[(j_c15)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c15)]));
    T_matmul_NN_local[((j_c15 + 16))] = (T_matmul_NN_local[((j_c15 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c15 + 4))]));
    T_matmul_NN_local[((j_c15 + 32))] = (T_matmul_NN_local[((j_c15 + 32))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c15 + 8))]));
    T_matmul_NN_local[((j_c15 + 48))] = (T_matmul_NN_local[((j_c15 + 48))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c15 + 12))]));
    T_matmul_NN_local[((j_c15 + 4))] = (T_matmul_NN_local[((j_c15 + 4))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c15)]));
    T_matmul_NN_local[((j_c15 + 20))] = (T_matmul_NN_local[((j_c15 + 20))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c15 + 4))]));
    T_matmul_NN_local[((j_c15 + 36))] = (T_matmul_NN_local[((j_c15 + 36))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c15 + 8))]));
    T_matmul_NN_local[((j_c15 + 52))] = (T_matmul_NN_local[((j_c15 + 52))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c15 + 12))]));
    T_matmul_NN_local[((j_c15 + 8))] = (T_matmul_NN_local[((j_c15 + 8))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c15)]));
    T_matmul_NN_local[((j_c15 + 24))] = (T_matmul_NN_local[((j_c15 + 24))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c15 + 4))]));
    T_matmul_NN_local[((j_c15 + 40))] = (T_matmul_NN_local[((j_c15 + 40))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c15 + 8))]));
    T_matmul_NN_local[((j_c15 + 56))] = (T_matmul_NN_local[((j_c15 + 56))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c15 + 12))]));
    T_matmul_NN_local[((j_c15 + 12))] = (T_matmul_NN_local[((j_c15 + 12))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c15)]));
    T_matmul_NN_local[((j_c15 + 28))] = (T_matmul_NN_local[((j_c15 + 28))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c15 + 4))]));
    T_matmul_NN_local[((j_c15 + 44))] = (T_matmul_NN_local[((j_c15 + 44))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c15 + 8))]));
    T_matmul_NN_local[((j_c15 + 60))] = (T_matmul_NN_local[((j_c15 + 60))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c15 + 12))]));
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 4; ++j_inner_inner_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner))] = T_matmul_NN_local[(j_inner_inner_inner)];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 64))] = T_matmul_NN_local[((j_inner_inner_inner + 16))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 128))] = T_matmul_NN_local[((j_inner_inner_inner + 32))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 192))] = T_matmul_NN_local[((j_inner_inner_inner + 48))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 65536))] = T_matmul_NN_local[((j_inner_inner_inner + 4))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 65600))] = T_matmul_NN_local[((j_inner_inner_inner + 20))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 65664))] = T_matmul_NN_local[((j_inner_inner_inner + 36))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 65728))] = T_matmul_NN_local[((j_inner_inner_inner + 52))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 131072))] = T_matmul_NN_local[((j_inner_inner_inner + 8))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 131136))] = T_matmul_NN_local[((j_inner_inner_inner + 24))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 131200))] = T_matmul_NN_local[((j_inner_inner_inner + 40))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 131264))] = T_matmul_NN_local[((j_inner_inner_inner + 56))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 196608))] = T_matmul_NN_local[((j_inner_inner_inner + 12))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 196672))] = T_matmul_NN_local[((j_inner_inner_inner + 28))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 196736))] = T_matmul_NN_local[((j_inner_inner_inner + 44))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 196800))] = T_matmul_NN_local[((j_inner_inner_inner + 60))];
  }
}

