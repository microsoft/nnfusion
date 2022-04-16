//4096_32_1_4_4_1
//65536_1024_1024
//dim3 grid(4096, 32, 1);
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
  float T_matmul_NN_local[32];
  __shared__ float placeholder_shared[512];
  __shared__ float placeholder_d_shared[1024];
  float placeholder_shared_local[4];
  float placeholder_d_shared_local[8];
  float placeholder_shared_local1[4];
  float placeholder_d_shared_local1[8];
  for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
    T_matmul_NN_local[(j_c_init)] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 8))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 16))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 24))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 2))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 10))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 18))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 26))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 4))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 12))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 20))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 28))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 6))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 14))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 22))] = 0.000000e+00f;
    T_matmul_NN_local[((j_c_init + 30))] = 0.000000e+00f;
  }
  for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      placeholder_shared[(((((((int)threadIdx.y) * 64) + (ax0_inner * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.y) * 4096)) + (ax0_inner * 1024)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
    }
  }
  for (int ax0_inner1 = 0; ax0_inner1 < 4; ++ax0_inner1) {
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
        placeholder_d_shared[((((((((int)threadIdx.y) * 128) + (ax0_inner1 * 32)) + (ax1_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[(((((((((int)threadIdx.y) * 4096) + (ax0_inner1 * 1024)) + (((int)blockIdx.y) * 32)) + (ax1_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 63; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner2 = 0; ax0_inner2 < 4; ++ax0_inner2) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        placeholder_shared[((((((((k_outer_outer + 1) & 1) * 256) + (((int)threadIdx.y) * 64)) + (ax0_inner2 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2))] = placeholder[((((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.y) * 4096)) + (ax0_inner2 * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) + 16))];
      }
    }
    for (int ax0_inner3 = 0; ax0_inner3 < 4; ++ax0_inner3) {
      for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
        for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
          placeholder_d_shared[(((((((((k_outer_outer + 1) & 1) * 512) + (((int)threadIdx.y) * 128)) + (ax0_inner3 * 32)) + (ax1_outer1 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[(((((((((k_outer_outer * 16384) + (((int)threadIdx.y) * 4096)) + (ax0_inner3 * 1024)) + (((int)blockIdx.y) * 32)) + (ax1_outer1 * 16)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 16384))];
        }
      }
    }
    placeholder_shared_local[(0)] = placeholder_shared[((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 64))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 128))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 192))];
    for (int ax1 = 0; ax1 < 2; ++ax1) {
      placeholder_d_shared_local[(ax1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax1))];
      placeholder_d_shared_local[((ax1 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax1) + 8))];
      placeholder_d_shared_local[((ax1 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax1) + 16))];
      placeholder_d_shared_local[((ax1 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax1) + 24))];
    }
    for (int j_c = 0; j_c < 2; ++j_c) {
      T_matmul_NN_local[(j_c)] = (T_matmul_NN_local[(j_c)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c)]));
      T_matmul_NN_local[((j_c + 8))] = (T_matmul_NN_local[((j_c + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c + 2))]));
      T_matmul_NN_local[((j_c + 16))] = (T_matmul_NN_local[((j_c + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c + 4))]));
      T_matmul_NN_local[((j_c + 24))] = (T_matmul_NN_local[((j_c + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c + 6))]));
      T_matmul_NN_local[((j_c + 2))] = (T_matmul_NN_local[((j_c + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c)]));
      T_matmul_NN_local[((j_c + 10))] = (T_matmul_NN_local[((j_c + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c + 2))]));
      T_matmul_NN_local[((j_c + 18))] = (T_matmul_NN_local[((j_c + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c + 4))]));
      T_matmul_NN_local[((j_c + 26))] = (T_matmul_NN_local[((j_c + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c + 6))]));
      T_matmul_NN_local[((j_c + 4))] = (T_matmul_NN_local[((j_c + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c)]));
      T_matmul_NN_local[((j_c + 12))] = (T_matmul_NN_local[((j_c + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c + 2))]));
      T_matmul_NN_local[((j_c + 20))] = (T_matmul_NN_local[((j_c + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c + 4))]));
      T_matmul_NN_local[((j_c + 28))] = (T_matmul_NN_local[((j_c + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c + 6))]));
      T_matmul_NN_local[((j_c + 6))] = (T_matmul_NN_local[((j_c + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c)]));
      T_matmul_NN_local[((j_c + 14))] = (T_matmul_NN_local[((j_c + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c + 2))]));
      T_matmul_NN_local[((j_c + 22))] = (T_matmul_NN_local[((j_c + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c + 4))]));
      T_matmul_NN_local[((j_c + 30))] = (T_matmul_NN_local[((j_c + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 1))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 65))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 129))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 193))];
    for (int ax11 = 0; ax11 < 2; ++ax11) {
      placeholder_d_shared_local[(ax11)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax11) + 32))];
      placeholder_d_shared_local[((ax11 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax11) + 40))];
      placeholder_d_shared_local[((ax11 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax11) + 48))];
      placeholder_d_shared_local[((ax11 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax11) + 56))];
    }
    for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
      T_matmul_NN_local[(j_c1)] = (T_matmul_NN_local[(j_c1)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c1)]));
      T_matmul_NN_local[((j_c1 + 8))] = (T_matmul_NN_local[((j_c1 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c1 + 2))]));
      T_matmul_NN_local[((j_c1 + 16))] = (T_matmul_NN_local[((j_c1 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c1 + 4))]));
      T_matmul_NN_local[((j_c1 + 24))] = (T_matmul_NN_local[((j_c1 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c1 + 6))]));
      T_matmul_NN_local[((j_c1 + 2))] = (T_matmul_NN_local[((j_c1 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c1)]));
      T_matmul_NN_local[((j_c1 + 10))] = (T_matmul_NN_local[((j_c1 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c1 + 2))]));
      T_matmul_NN_local[((j_c1 + 18))] = (T_matmul_NN_local[((j_c1 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c1 + 4))]));
      T_matmul_NN_local[((j_c1 + 26))] = (T_matmul_NN_local[((j_c1 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c1 + 6))]));
      T_matmul_NN_local[((j_c1 + 4))] = (T_matmul_NN_local[((j_c1 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c1)]));
      T_matmul_NN_local[((j_c1 + 12))] = (T_matmul_NN_local[((j_c1 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c1 + 2))]));
      T_matmul_NN_local[((j_c1 + 20))] = (T_matmul_NN_local[((j_c1 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c1 + 4))]));
      T_matmul_NN_local[((j_c1 + 28))] = (T_matmul_NN_local[((j_c1 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c1 + 6))]));
      T_matmul_NN_local[((j_c1 + 6))] = (T_matmul_NN_local[((j_c1 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c1)]));
      T_matmul_NN_local[((j_c1 + 14))] = (T_matmul_NN_local[((j_c1 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c1 + 2))]));
      T_matmul_NN_local[((j_c1 + 22))] = (T_matmul_NN_local[((j_c1 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c1 + 4))]));
      T_matmul_NN_local[((j_c1 + 30))] = (T_matmul_NN_local[((j_c1 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c1 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 2))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 66))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 130))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 194))];
    for (int ax12 = 0; ax12 < 2; ++ax12) {
      placeholder_d_shared_local[(ax12)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax12) + 64))];
      placeholder_d_shared_local[((ax12 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax12) + 72))];
      placeholder_d_shared_local[((ax12 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax12) + 80))];
      placeholder_d_shared_local[((ax12 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax12) + 88))];
    }
    for (int j_c2 = 0; j_c2 < 2; ++j_c2) {
      T_matmul_NN_local[(j_c2)] = (T_matmul_NN_local[(j_c2)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c2)]));
      T_matmul_NN_local[((j_c2 + 8))] = (T_matmul_NN_local[((j_c2 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c2 + 2))]));
      T_matmul_NN_local[((j_c2 + 16))] = (T_matmul_NN_local[((j_c2 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c2 + 4))]));
      T_matmul_NN_local[((j_c2 + 24))] = (T_matmul_NN_local[((j_c2 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c2 + 6))]));
      T_matmul_NN_local[((j_c2 + 2))] = (T_matmul_NN_local[((j_c2 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c2)]));
      T_matmul_NN_local[((j_c2 + 10))] = (T_matmul_NN_local[((j_c2 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c2 + 2))]));
      T_matmul_NN_local[((j_c2 + 18))] = (T_matmul_NN_local[((j_c2 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c2 + 4))]));
      T_matmul_NN_local[((j_c2 + 26))] = (T_matmul_NN_local[((j_c2 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c2 + 6))]));
      T_matmul_NN_local[((j_c2 + 4))] = (T_matmul_NN_local[((j_c2 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c2)]));
      T_matmul_NN_local[((j_c2 + 12))] = (T_matmul_NN_local[((j_c2 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c2 + 2))]));
      T_matmul_NN_local[((j_c2 + 20))] = (T_matmul_NN_local[((j_c2 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c2 + 4))]));
      T_matmul_NN_local[((j_c2 + 28))] = (T_matmul_NN_local[((j_c2 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c2 + 6))]));
      T_matmul_NN_local[((j_c2 + 6))] = (T_matmul_NN_local[((j_c2 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c2)]));
      T_matmul_NN_local[((j_c2 + 14))] = (T_matmul_NN_local[((j_c2 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c2 + 2))]));
      T_matmul_NN_local[((j_c2 + 22))] = (T_matmul_NN_local[((j_c2 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c2 + 4))]));
      T_matmul_NN_local[((j_c2 + 30))] = (T_matmul_NN_local[((j_c2 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c2 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 3))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 67))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 131))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 195))];
    for (int ax13 = 0; ax13 < 2; ++ax13) {
      placeholder_d_shared_local[(ax13)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax13) + 96))];
      placeholder_d_shared_local[((ax13 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax13) + 104))];
      placeholder_d_shared_local[((ax13 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax13) + 112))];
      placeholder_d_shared_local[((ax13 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax13) + 120))];
    }
    for (int j_c3 = 0; j_c3 < 2; ++j_c3) {
      T_matmul_NN_local[(j_c3)] = (T_matmul_NN_local[(j_c3)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c3)]));
      T_matmul_NN_local[((j_c3 + 8))] = (T_matmul_NN_local[((j_c3 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c3 + 2))]));
      T_matmul_NN_local[((j_c3 + 16))] = (T_matmul_NN_local[((j_c3 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c3 + 4))]));
      T_matmul_NN_local[((j_c3 + 24))] = (T_matmul_NN_local[((j_c3 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c3 + 6))]));
      T_matmul_NN_local[((j_c3 + 2))] = (T_matmul_NN_local[((j_c3 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c3)]));
      T_matmul_NN_local[((j_c3 + 10))] = (T_matmul_NN_local[((j_c3 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c3 + 2))]));
      T_matmul_NN_local[((j_c3 + 18))] = (T_matmul_NN_local[((j_c3 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c3 + 4))]));
      T_matmul_NN_local[((j_c3 + 26))] = (T_matmul_NN_local[((j_c3 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c3 + 6))]));
      T_matmul_NN_local[((j_c3 + 4))] = (T_matmul_NN_local[((j_c3 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c3)]));
      T_matmul_NN_local[((j_c3 + 12))] = (T_matmul_NN_local[((j_c3 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c3 + 2))]));
      T_matmul_NN_local[((j_c3 + 20))] = (T_matmul_NN_local[((j_c3 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c3 + 4))]));
      T_matmul_NN_local[((j_c3 + 28))] = (T_matmul_NN_local[((j_c3 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c3 + 6))]));
      T_matmul_NN_local[((j_c3 + 6))] = (T_matmul_NN_local[((j_c3 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c3)]));
      T_matmul_NN_local[((j_c3 + 14))] = (T_matmul_NN_local[((j_c3 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c3 + 2))]));
      T_matmul_NN_local[((j_c3 + 22))] = (T_matmul_NN_local[((j_c3 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c3 + 4))]));
      T_matmul_NN_local[((j_c3 + 30))] = (T_matmul_NN_local[((j_c3 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c3 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 4))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 68))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 132))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 196))];
    for (int ax14 = 0; ax14 < 2; ++ax14) {
      placeholder_d_shared_local[(ax14)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax14) + 128))];
      placeholder_d_shared_local[((ax14 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax14) + 136))];
      placeholder_d_shared_local[((ax14 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax14) + 144))];
      placeholder_d_shared_local[((ax14 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax14) + 152))];
    }
    for (int j_c4 = 0; j_c4 < 2; ++j_c4) {
      T_matmul_NN_local[(j_c4)] = (T_matmul_NN_local[(j_c4)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c4)]));
      T_matmul_NN_local[((j_c4 + 8))] = (T_matmul_NN_local[((j_c4 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c4 + 2))]));
      T_matmul_NN_local[((j_c4 + 16))] = (T_matmul_NN_local[((j_c4 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c4 + 4))]));
      T_matmul_NN_local[((j_c4 + 24))] = (T_matmul_NN_local[((j_c4 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c4 + 6))]));
      T_matmul_NN_local[((j_c4 + 2))] = (T_matmul_NN_local[((j_c4 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c4)]));
      T_matmul_NN_local[((j_c4 + 10))] = (T_matmul_NN_local[((j_c4 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c4 + 2))]));
      T_matmul_NN_local[((j_c4 + 18))] = (T_matmul_NN_local[((j_c4 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c4 + 4))]));
      T_matmul_NN_local[((j_c4 + 26))] = (T_matmul_NN_local[((j_c4 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c4 + 6))]));
      T_matmul_NN_local[((j_c4 + 4))] = (T_matmul_NN_local[((j_c4 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c4)]));
      T_matmul_NN_local[((j_c4 + 12))] = (T_matmul_NN_local[((j_c4 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c4 + 2))]));
      T_matmul_NN_local[((j_c4 + 20))] = (T_matmul_NN_local[((j_c4 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c4 + 4))]));
      T_matmul_NN_local[((j_c4 + 28))] = (T_matmul_NN_local[((j_c4 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c4 + 6))]));
      T_matmul_NN_local[((j_c4 + 6))] = (T_matmul_NN_local[((j_c4 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c4)]));
      T_matmul_NN_local[((j_c4 + 14))] = (T_matmul_NN_local[((j_c4 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c4 + 2))]));
      T_matmul_NN_local[((j_c4 + 22))] = (T_matmul_NN_local[((j_c4 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c4 + 4))]));
      T_matmul_NN_local[((j_c4 + 30))] = (T_matmul_NN_local[((j_c4 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c4 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 5))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 69))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 133))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 197))];
    for (int ax15 = 0; ax15 < 2; ++ax15) {
      placeholder_d_shared_local[(ax15)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax15) + 160))];
      placeholder_d_shared_local[((ax15 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax15) + 168))];
      placeholder_d_shared_local[((ax15 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax15) + 176))];
      placeholder_d_shared_local[((ax15 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax15) + 184))];
    }
    for (int j_c5 = 0; j_c5 < 2; ++j_c5) {
      T_matmul_NN_local[(j_c5)] = (T_matmul_NN_local[(j_c5)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c5)]));
      T_matmul_NN_local[((j_c5 + 8))] = (T_matmul_NN_local[((j_c5 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c5 + 2))]));
      T_matmul_NN_local[((j_c5 + 16))] = (T_matmul_NN_local[((j_c5 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c5 + 4))]));
      T_matmul_NN_local[((j_c5 + 24))] = (T_matmul_NN_local[((j_c5 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c5 + 6))]));
      T_matmul_NN_local[((j_c5 + 2))] = (T_matmul_NN_local[((j_c5 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c5)]));
      T_matmul_NN_local[((j_c5 + 10))] = (T_matmul_NN_local[((j_c5 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c5 + 2))]));
      T_matmul_NN_local[((j_c5 + 18))] = (T_matmul_NN_local[((j_c5 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c5 + 4))]));
      T_matmul_NN_local[((j_c5 + 26))] = (T_matmul_NN_local[((j_c5 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c5 + 6))]));
      T_matmul_NN_local[((j_c5 + 4))] = (T_matmul_NN_local[((j_c5 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c5)]));
      T_matmul_NN_local[((j_c5 + 12))] = (T_matmul_NN_local[((j_c5 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c5 + 2))]));
      T_matmul_NN_local[((j_c5 + 20))] = (T_matmul_NN_local[((j_c5 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c5 + 4))]));
      T_matmul_NN_local[((j_c5 + 28))] = (T_matmul_NN_local[((j_c5 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c5 + 6))]));
      T_matmul_NN_local[((j_c5 + 6))] = (T_matmul_NN_local[((j_c5 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c5)]));
      T_matmul_NN_local[((j_c5 + 14))] = (T_matmul_NN_local[((j_c5 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c5 + 2))]));
      T_matmul_NN_local[((j_c5 + 22))] = (T_matmul_NN_local[((j_c5 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c5 + 4))]));
      T_matmul_NN_local[((j_c5 + 30))] = (T_matmul_NN_local[((j_c5 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c5 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 6))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 70))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 134))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 198))];
    for (int ax16 = 0; ax16 < 2; ++ax16) {
      placeholder_d_shared_local[(ax16)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax16) + 192))];
      placeholder_d_shared_local[((ax16 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax16) + 200))];
      placeholder_d_shared_local[((ax16 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax16) + 208))];
      placeholder_d_shared_local[((ax16 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax16) + 216))];
    }
    for (int j_c6 = 0; j_c6 < 2; ++j_c6) {
      T_matmul_NN_local[(j_c6)] = (T_matmul_NN_local[(j_c6)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c6)]));
      T_matmul_NN_local[((j_c6 + 8))] = (T_matmul_NN_local[((j_c6 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c6 + 2))]));
      T_matmul_NN_local[((j_c6 + 16))] = (T_matmul_NN_local[((j_c6 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c6 + 4))]));
      T_matmul_NN_local[((j_c6 + 24))] = (T_matmul_NN_local[((j_c6 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c6 + 6))]));
      T_matmul_NN_local[((j_c6 + 2))] = (T_matmul_NN_local[((j_c6 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c6)]));
      T_matmul_NN_local[((j_c6 + 10))] = (T_matmul_NN_local[((j_c6 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c6 + 2))]));
      T_matmul_NN_local[((j_c6 + 18))] = (T_matmul_NN_local[((j_c6 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c6 + 4))]));
      T_matmul_NN_local[((j_c6 + 26))] = (T_matmul_NN_local[((j_c6 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c6 + 6))]));
      T_matmul_NN_local[((j_c6 + 4))] = (T_matmul_NN_local[((j_c6 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c6)]));
      T_matmul_NN_local[((j_c6 + 12))] = (T_matmul_NN_local[((j_c6 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c6 + 2))]));
      T_matmul_NN_local[((j_c6 + 20))] = (T_matmul_NN_local[((j_c6 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c6 + 4))]));
      T_matmul_NN_local[((j_c6 + 28))] = (T_matmul_NN_local[((j_c6 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c6 + 6))]));
      T_matmul_NN_local[((j_c6 + 6))] = (T_matmul_NN_local[((j_c6 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c6)]));
      T_matmul_NN_local[((j_c6 + 14))] = (T_matmul_NN_local[((j_c6 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c6 + 2))]));
      T_matmul_NN_local[((j_c6 + 22))] = (T_matmul_NN_local[((j_c6 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c6 + 4))]));
      T_matmul_NN_local[((j_c6 + 30))] = (T_matmul_NN_local[((j_c6 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c6 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 7))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 71))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 135))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 199))];
    for (int ax17 = 0; ax17 < 2; ++ax17) {
      placeholder_d_shared_local[(ax17)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax17) + 224))];
      placeholder_d_shared_local[((ax17 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax17) + 232))];
      placeholder_d_shared_local[((ax17 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax17) + 240))];
      placeholder_d_shared_local[((ax17 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax17) + 248))];
    }
    for (int j_c7 = 0; j_c7 < 2; ++j_c7) {
      T_matmul_NN_local[(j_c7)] = (T_matmul_NN_local[(j_c7)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c7)]));
      T_matmul_NN_local[((j_c7 + 8))] = (T_matmul_NN_local[((j_c7 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c7 + 2))]));
      T_matmul_NN_local[((j_c7 + 16))] = (T_matmul_NN_local[((j_c7 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c7 + 4))]));
      T_matmul_NN_local[((j_c7 + 24))] = (T_matmul_NN_local[((j_c7 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c7 + 6))]));
      T_matmul_NN_local[((j_c7 + 2))] = (T_matmul_NN_local[((j_c7 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c7)]));
      T_matmul_NN_local[((j_c7 + 10))] = (T_matmul_NN_local[((j_c7 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c7 + 2))]));
      T_matmul_NN_local[((j_c7 + 18))] = (T_matmul_NN_local[((j_c7 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c7 + 4))]));
      T_matmul_NN_local[((j_c7 + 26))] = (T_matmul_NN_local[((j_c7 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c7 + 6))]));
      T_matmul_NN_local[((j_c7 + 4))] = (T_matmul_NN_local[((j_c7 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c7)]));
      T_matmul_NN_local[((j_c7 + 12))] = (T_matmul_NN_local[((j_c7 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c7 + 2))]));
      T_matmul_NN_local[((j_c7 + 20))] = (T_matmul_NN_local[((j_c7 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c7 + 4))]));
      T_matmul_NN_local[((j_c7 + 28))] = (T_matmul_NN_local[((j_c7 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c7 + 6))]));
      T_matmul_NN_local[((j_c7 + 6))] = (T_matmul_NN_local[((j_c7 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c7)]));
      T_matmul_NN_local[((j_c7 + 14))] = (T_matmul_NN_local[((j_c7 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c7 + 2))]));
      T_matmul_NN_local[((j_c7 + 22))] = (T_matmul_NN_local[((j_c7 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c7 + 4))]));
      T_matmul_NN_local[((j_c7 + 30))] = (T_matmul_NN_local[((j_c7 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c7 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 8))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 72))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 136))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 200))];
    for (int ax18 = 0; ax18 < 2; ++ax18) {
      placeholder_d_shared_local[(ax18)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax18) + 256))];
      placeholder_d_shared_local[((ax18 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax18) + 264))];
      placeholder_d_shared_local[((ax18 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax18) + 272))];
      placeholder_d_shared_local[((ax18 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax18) + 280))];
    }
    for (int j_c8 = 0; j_c8 < 2; ++j_c8) {
      T_matmul_NN_local[(j_c8)] = (T_matmul_NN_local[(j_c8)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c8)]));
      T_matmul_NN_local[((j_c8 + 8))] = (T_matmul_NN_local[((j_c8 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c8 + 2))]));
      T_matmul_NN_local[((j_c8 + 16))] = (T_matmul_NN_local[((j_c8 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c8 + 4))]));
      T_matmul_NN_local[((j_c8 + 24))] = (T_matmul_NN_local[((j_c8 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c8 + 6))]));
      T_matmul_NN_local[((j_c8 + 2))] = (T_matmul_NN_local[((j_c8 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c8)]));
      T_matmul_NN_local[((j_c8 + 10))] = (T_matmul_NN_local[((j_c8 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c8 + 2))]));
      T_matmul_NN_local[((j_c8 + 18))] = (T_matmul_NN_local[((j_c8 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c8 + 4))]));
      T_matmul_NN_local[((j_c8 + 26))] = (T_matmul_NN_local[((j_c8 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c8 + 6))]));
      T_matmul_NN_local[((j_c8 + 4))] = (T_matmul_NN_local[((j_c8 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c8)]));
      T_matmul_NN_local[((j_c8 + 12))] = (T_matmul_NN_local[((j_c8 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c8 + 2))]));
      T_matmul_NN_local[((j_c8 + 20))] = (T_matmul_NN_local[((j_c8 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c8 + 4))]));
      T_matmul_NN_local[((j_c8 + 28))] = (T_matmul_NN_local[((j_c8 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c8 + 6))]));
      T_matmul_NN_local[((j_c8 + 6))] = (T_matmul_NN_local[((j_c8 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c8)]));
      T_matmul_NN_local[((j_c8 + 14))] = (T_matmul_NN_local[((j_c8 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c8 + 2))]));
      T_matmul_NN_local[((j_c8 + 22))] = (T_matmul_NN_local[((j_c8 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c8 + 4))]));
      T_matmul_NN_local[((j_c8 + 30))] = (T_matmul_NN_local[((j_c8 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c8 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 9))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 73))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 137))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 201))];
    for (int ax19 = 0; ax19 < 2; ++ax19) {
      placeholder_d_shared_local[(ax19)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax19) + 288))];
      placeholder_d_shared_local[((ax19 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax19) + 296))];
      placeholder_d_shared_local[((ax19 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax19) + 304))];
      placeholder_d_shared_local[((ax19 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax19) + 312))];
    }
    for (int j_c9 = 0; j_c9 < 2; ++j_c9) {
      T_matmul_NN_local[(j_c9)] = (T_matmul_NN_local[(j_c9)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c9)]));
      T_matmul_NN_local[((j_c9 + 8))] = (T_matmul_NN_local[((j_c9 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c9 + 2))]));
      T_matmul_NN_local[((j_c9 + 16))] = (T_matmul_NN_local[((j_c9 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c9 + 4))]));
      T_matmul_NN_local[((j_c9 + 24))] = (T_matmul_NN_local[((j_c9 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c9 + 6))]));
      T_matmul_NN_local[((j_c9 + 2))] = (T_matmul_NN_local[((j_c9 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c9)]));
      T_matmul_NN_local[((j_c9 + 10))] = (T_matmul_NN_local[((j_c9 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c9 + 2))]));
      T_matmul_NN_local[((j_c9 + 18))] = (T_matmul_NN_local[((j_c9 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c9 + 4))]));
      T_matmul_NN_local[((j_c9 + 26))] = (T_matmul_NN_local[((j_c9 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c9 + 6))]));
      T_matmul_NN_local[((j_c9 + 4))] = (T_matmul_NN_local[((j_c9 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c9)]));
      T_matmul_NN_local[((j_c9 + 12))] = (T_matmul_NN_local[((j_c9 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c9 + 2))]));
      T_matmul_NN_local[((j_c9 + 20))] = (T_matmul_NN_local[((j_c9 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c9 + 4))]));
      T_matmul_NN_local[((j_c9 + 28))] = (T_matmul_NN_local[((j_c9 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c9 + 6))]));
      T_matmul_NN_local[((j_c9 + 6))] = (T_matmul_NN_local[((j_c9 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c9)]));
      T_matmul_NN_local[((j_c9 + 14))] = (T_matmul_NN_local[((j_c9 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c9 + 2))]));
      T_matmul_NN_local[((j_c9 + 22))] = (T_matmul_NN_local[((j_c9 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c9 + 4))]));
      T_matmul_NN_local[((j_c9 + 30))] = (T_matmul_NN_local[((j_c9 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c9 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 10))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 74))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 138))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 202))];
    for (int ax110 = 0; ax110 < 2; ++ax110) {
      placeholder_d_shared_local[(ax110)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax110) + 320))];
      placeholder_d_shared_local[((ax110 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax110) + 328))];
      placeholder_d_shared_local[((ax110 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax110) + 336))];
      placeholder_d_shared_local[((ax110 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax110) + 344))];
    }
    for (int j_c10 = 0; j_c10 < 2; ++j_c10) {
      T_matmul_NN_local[(j_c10)] = (T_matmul_NN_local[(j_c10)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c10)]));
      T_matmul_NN_local[((j_c10 + 8))] = (T_matmul_NN_local[((j_c10 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c10 + 2))]));
      T_matmul_NN_local[((j_c10 + 16))] = (T_matmul_NN_local[((j_c10 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c10 + 4))]));
      T_matmul_NN_local[((j_c10 + 24))] = (T_matmul_NN_local[((j_c10 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c10 + 6))]));
      T_matmul_NN_local[((j_c10 + 2))] = (T_matmul_NN_local[((j_c10 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c10)]));
      T_matmul_NN_local[((j_c10 + 10))] = (T_matmul_NN_local[((j_c10 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c10 + 2))]));
      T_matmul_NN_local[((j_c10 + 18))] = (T_matmul_NN_local[((j_c10 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c10 + 4))]));
      T_matmul_NN_local[((j_c10 + 26))] = (T_matmul_NN_local[((j_c10 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c10 + 6))]));
      T_matmul_NN_local[((j_c10 + 4))] = (T_matmul_NN_local[((j_c10 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c10)]));
      T_matmul_NN_local[((j_c10 + 12))] = (T_matmul_NN_local[((j_c10 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c10 + 2))]));
      T_matmul_NN_local[((j_c10 + 20))] = (T_matmul_NN_local[((j_c10 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c10 + 4))]));
      T_matmul_NN_local[((j_c10 + 28))] = (T_matmul_NN_local[((j_c10 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c10 + 6))]));
      T_matmul_NN_local[((j_c10 + 6))] = (T_matmul_NN_local[((j_c10 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c10)]));
      T_matmul_NN_local[((j_c10 + 14))] = (T_matmul_NN_local[((j_c10 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c10 + 2))]));
      T_matmul_NN_local[((j_c10 + 22))] = (T_matmul_NN_local[((j_c10 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c10 + 4))]));
      T_matmul_NN_local[((j_c10 + 30))] = (T_matmul_NN_local[((j_c10 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c10 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 11))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 75))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 139))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 203))];
    for (int ax111 = 0; ax111 < 2; ++ax111) {
      placeholder_d_shared_local[(ax111)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax111) + 352))];
      placeholder_d_shared_local[((ax111 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax111) + 360))];
      placeholder_d_shared_local[((ax111 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax111) + 368))];
      placeholder_d_shared_local[((ax111 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax111) + 376))];
    }
    for (int j_c11 = 0; j_c11 < 2; ++j_c11) {
      T_matmul_NN_local[(j_c11)] = (T_matmul_NN_local[(j_c11)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c11)]));
      T_matmul_NN_local[((j_c11 + 8))] = (T_matmul_NN_local[((j_c11 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c11 + 2))]));
      T_matmul_NN_local[((j_c11 + 16))] = (T_matmul_NN_local[((j_c11 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c11 + 4))]));
      T_matmul_NN_local[((j_c11 + 24))] = (T_matmul_NN_local[((j_c11 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c11 + 6))]));
      T_matmul_NN_local[((j_c11 + 2))] = (T_matmul_NN_local[((j_c11 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c11)]));
      T_matmul_NN_local[((j_c11 + 10))] = (T_matmul_NN_local[((j_c11 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c11 + 2))]));
      T_matmul_NN_local[((j_c11 + 18))] = (T_matmul_NN_local[((j_c11 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c11 + 4))]));
      T_matmul_NN_local[((j_c11 + 26))] = (T_matmul_NN_local[((j_c11 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c11 + 6))]));
      T_matmul_NN_local[((j_c11 + 4))] = (T_matmul_NN_local[((j_c11 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c11)]));
      T_matmul_NN_local[((j_c11 + 12))] = (T_matmul_NN_local[((j_c11 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c11 + 2))]));
      T_matmul_NN_local[((j_c11 + 20))] = (T_matmul_NN_local[((j_c11 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c11 + 4))]));
      T_matmul_NN_local[((j_c11 + 28))] = (T_matmul_NN_local[((j_c11 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c11 + 6))]));
      T_matmul_NN_local[((j_c11 + 6))] = (T_matmul_NN_local[((j_c11 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c11)]));
      T_matmul_NN_local[((j_c11 + 14))] = (T_matmul_NN_local[((j_c11 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c11 + 2))]));
      T_matmul_NN_local[((j_c11 + 22))] = (T_matmul_NN_local[((j_c11 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c11 + 4))]));
      T_matmul_NN_local[((j_c11 + 30))] = (T_matmul_NN_local[((j_c11 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c11 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 12))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 76))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 140))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 204))];
    for (int ax112 = 0; ax112 < 2; ++ax112) {
      placeholder_d_shared_local[(ax112)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax112) + 384))];
      placeholder_d_shared_local[((ax112 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax112) + 392))];
      placeholder_d_shared_local[((ax112 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax112) + 400))];
      placeholder_d_shared_local[((ax112 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax112) + 408))];
    }
    for (int j_c12 = 0; j_c12 < 2; ++j_c12) {
      T_matmul_NN_local[(j_c12)] = (T_matmul_NN_local[(j_c12)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c12)]));
      T_matmul_NN_local[((j_c12 + 8))] = (T_matmul_NN_local[((j_c12 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c12 + 2))]));
      T_matmul_NN_local[((j_c12 + 16))] = (T_matmul_NN_local[((j_c12 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c12 + 4))]));
      T_matmul_NN_local[((j_c12 + 24))] = (T_matmul_NN_local[((j_c12 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c12 + 6))]));
      T_matmul_NN_local[((j_c12 + 2))] = (T_matmul_NN_local[((j_c12 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c12)]));
      T_matmul_NN_local[((j_c12 + 10))] = (T_matmul_NN_local[((j_c12 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c12 + 2))]));
      T_matmul_NN_local[((j_c12 + 18))] = (T_matmul_NN_local[((j_c12 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c12 + 4))]));
      T_matmul_NN_local[((j_c12 + 26))] = (T_matmul_NN_local[((j_c12 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c12 + 6))]));
      T_matmul_NN_local[((j_c12 + 4))] = (T_matmul_NN_local[((j_c12 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c12)]));
      T_matmul_NN_local[((j_c12 + 12))] = (T_matmul_NN_local[((j_c12 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c12 + 2))]));
      T_matmul_NN_local[((j_c12 + 20))] = (T_matmul_NN_local[((j_c12 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c12 + 4))]));
      T_matmul_NN_local[((j_c12 + 28))] = (T_matmul_NN_local[((j_c12 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c12 + 6))]));
      T_matmul_NN_local[((j_c12 + 6))] = (T_matmul_NN_local[((j_c12 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c12)]));
      T_matmul_NN_local[((j_c12 + 14))] = (T_matmul_NN_local[((j_c12 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c12 + 2))]));
      T_matmul_NN_local[((j_c12 + 22))] = (T_matmul_NN_local[((j_c12 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c12 + 4))]));
      T_matmul_NN_local[((j_c12 + 30))] = (T_matmul_NN_local[((j_c12 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c12 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 13))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 77))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 141))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 205))];
    for (int ax113 = 0; ax113 < 2; ++ax113) {
      placeholder_d_shared_local[(ax113)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax113) + 416))];
      placeholder_d_shared_local[((ax113 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax113) + 424))];
      placeholder_d_shared_local[((ax113 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax113) + 432))];
      placeholder_d_shared_local[((ax113 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax113) + 440))];
    }
    for (int j_c13 = 0; j_c13 < 2; ++j_c13) {
      T_matmul_NN_local[(j_c13)] = (T_matmul_NN_local[(j_c13)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c13)]));
      T_matmul_NN_local[((j_c13 + 8))] = (T_matmul_NN_local[((j_c13 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c13 + 2))]));
      T_matmul_NN_local[((j_c13 + 16))] = (T_matmul_NN_local[((j_c13 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c13 + 4))]));
      T_matmul_NN_local[((j_c13 + 24))] = (T_matmul_NN_local[((j_c13 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c13 + 6))]));
      T_matmul_NN_local[((j_c13 + 2))] = (T_matmul_NN_local[((j_c13 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c13)]));
      T_matmul_NN_local[((j_c13 + 10))] = (T_matmul_NN_local[((j_c13 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c13 + 2))]));
      T_matmul_NN_local[((j_c13 + 18))] = (T_matmul_NN_local[((j_c13 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c13 + 4))]));
      T_matmul_NN_local[((j_c13 + 26))] = (T_matmul_NN_local[((j_c13 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c13 + 6))]));
      T_matmul_NN_local[((j_c13 + 4))] = (T_matmul_NN_local[((j_c13 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c13)]));
      T_matmul_NN_local[((j_c13 + 12))] = (T_matmul_NN_local[((j_c13 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c13 + 2))]));
      T_matmul_NN_local[((j_c13 + 20))] = (T_matmul_NN_local[((j_c13 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c13 + 4))]));
      T_matmul_NN_local[((j_c13 + 28))] = (T_matmul_NN_local[((j_c13 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c13 + 6))]));
      T_matmul_NN_local[((j_c13 + 6))] = (T_matmul_NN_local[((j_c13 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c13)]));
      T_matmul_NN_local[((j_c13 + 14))] = (T_matmul_NN_local[((j_c13 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c13 + 2))]));
      T_matmul_NN_local[((j_c13 + 22))] = (T_matmul_NN_local[((j_c13 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c13 + 4))]));
      T_matmul_NN_local[((j_c13 + 30))] = (T_matmul_NN_local[((j_c13 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c13 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 14))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 78))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 142))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 206))];
    for (int ax114 = 0; ax114 < 2; ++ax114) {
      placeholder_d_shared_local[(ax114)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax114) + 448))];
      placeholder_d_shared_local[((ax114 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax114) + 456))];
      placeholder_d_shared_local[((ax114 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax114) + 464))];
      placeholder_d_shared_local[((ax114 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax114) + 472))];
    }
    for (int j_c14 = 0; j_c14 < 2; ++j_c14) {
      T_matmul_NN_local[(j_c14)] = (T_matmul_NN_local[(j_c14)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c14)]));
      T_matmul_NN_local[((j_c14 + 8))] = (T_matmul_NN_local[((j_c14 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c14 + 2))]));
      T_matmul_NN_local[((j_c14 + 16))] = (T_matmul_NN_local[((j_c14 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c14 + 4))]));
      T_matmul_NN_local[((j_c14 + 24))] = (T_matmul_NN_local[((j_c14 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c14 + 6))]));
      T_matmul_NN_local[((j_c14 + 2))] = (T_matmul_NN_local[((j_c14 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c14)]));
      T_matmul_NN_local[((j_c14 + 10))] = (T_matmul_NN_local[((j_c14 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c14 + 2))]));
      T_matmul_NN_local[((j_c14 + 18))] = (T_matmul_NN_local[((j_c14 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c14 + 4))]));
      T_matmul_NN_local[((j_c14 + 26))] = (T_matmul_NN_local[((j_c14 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c14 + 6))]));
      T_matmul_NN_local[((j_c14 + 4))] = (T_matmul_NN_local[((j_c14 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c14)]));
      T_matmul_NN_local[((j_c14 + 12))] = (T_matmul_NN_local[((j_c14 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c14 + 2))]));
      T_matmul_NN_local[((j_c14 + 20))] = (T_matmul_NN_local[((j_c14 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c14 + 4))]));
      T_matmul_NN_local[((j_c14 + 28))] = (T_matmul_NN_local[((j_c14 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c14 + 6))]));
      T_matmul_NN_local[((j_c14 + 6))] = (T_matmul_NN_local[((j_c14 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c14)]));
      T_matmul_NN_local[((j_c14 + 14))] = (T_matmul_NN_local[((j_c14 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c14 + 2))]));
      T_matmul_NN_local[((j_c14 + 22))] = (T_matmul_NN_local[((j_c14 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c14 + 4))]));
      T_matmul_NN_local[((j_c14 + 30))] = (T_matmul_NN_local[((j_c14 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c14 + 6))]));
    }
    placeholder_shared_local[(0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 15))];
    placeholder_shared_local[(1)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 79))];
    placeholder_shared_local[(2)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 143))];
    placeholder_shared_local[(3)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + 207))];
    for (int ax115 = 0; ax115 < 2; ++ax115) {
      placeholder_d_shared_local[(ax115)] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax115) + 480))];
      placeholder_d_shared_local[((ax115 + 2))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax115) + 488))];
      placeholder_d_shared_local[((ax115 + 4))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax115) + 496))];
      placeholder_d_shared_local[((ax115 + 6))] = placeholder_d_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 2)) + ax115) + 504))];
    }
    for (int j_c15 = 0; j_c15 < 2; ++j_c15) {
      T_matmul_NN_local[(j_c15)] = (T_matmul_NN_local[(j_c15)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(j_c15)]));
      T_matmul_NN_local[((j_c15 + 8))] = (T_matmul_NN_local[((j_c15 + 8))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c15 + 2))]));
      T_matmul_NN_local[((j_c15 + 16))] = (T_matmul_NN_local[((j_c15 + 16))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c15 + 4))]));
      T_matmul_NN_local[((j_c15 + 24))] = (T_matmul_NN_local[((j_c15 + 24))] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[((j_c15 + 6))]));
      T_matmul_NN_local[((j_c15 + 2))] = (T_matmul_NN_local[((j_c15 + 2))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(j_c15)]));
      T_matmul_NN_local[((j_c15 + 10))] = (T_matmul_NN_local[((j_c15 + 10))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c15 + 2))]));
      T_matmul_NN_local[((j_c15 + 18))] = (T_matmul_NN_local[((j_c15 + 18))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c15 + 4))]));
      T_matmul_NN_local[((j_c15 + 26))] = (T_matmul_NN_local[((j_c15 + 26))] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[((j_c15 + 6))]));
      T_matmul_NN_local[((j_c15 + 4))] = (T_matmul_NN_local[((j_c15 + 4))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(j_c15)]));
      T_matmul_NN_local[((j_c15 + 12))] = (T_matmul_NN_local[((j_c15 + 12))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c15 + 2))]));
      T_matmul_NN_local[((j_c15 + 20))] = (T_matmul_NN_local[((j_c15 + 20))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c15 + 4))]));
      T_matmul_NN_local[((j_c15 + 28))] = (T_matmul_NN_local[((j_c15 + 28))] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[((j_c15 + 6))]));
      T_matmul_NN_local[((j_c15 + 6))] = (T_matmul_NN_local[((j_c15 + 6))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(j_c15)]));
      T_matmul_NN_local[((j_c15 + 14))] = (T_matmul_NN_local[((j_c15 + 14))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c15 + 2))]));
      T_matmul_NN_local[((j_c15 + 22))] = (T_matmul_NN_local[((j_c15 + 22))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c15 + 4))]));
      T_matmul_NN_local[((j_c15 + 30))] = (T_matmul_NN_local[((j_c15 + 30))] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[((j_c15 + 6))]));
    }
  }
  __syncthreads();
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 256))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 320))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 384))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 448))];
  for (int ax116 = 0; ax116 < 2; ++ax116) {
    placeholder_d_shared_local1[(ax116)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax116) + 512))];
    placeholder_d_shared_local1[((ax116 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax116) + 520))];
    placeholder_d_shared_local1[((ax116 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax116) + 528))];
    placeholder_d_shared_local1[((ax116 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax116) + 536))];
  }
  for (int j_c16 = 0; j_c16 < 2; ++j_c16) {
    T_matmul_NN_local[(j_c16)] = (T_matmul_NN_local[(j_c16)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c16)]));
    T_matmul_NN_local[((j_c16 + 8))] = (T_matmul_NN_local[((j_c16 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c16 + 2))]));
    T_matmul_NN_local[((j_c16 + 16))] = (T_matmul_NN_local[((j_c16 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c16 + 4))]));
    T_matmul_NN_local[((j_c16 + 24))] = (T_matmul_NN_local[((j_c16 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c16 + 6))]));
    T_matmul_NN_local[((j_c16 + 2))] = (T_matmul_NN_local[((j_c16 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c16)]));
    T_matmul_NN_local[((j_c16 + 10))] = (T_matmul_NN_local[((j_c16 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c16 + 2))]));
    T_matmul_NN_local[((j_c16 + 18))] = (T_matmul_NN_local[((j_c16 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c16 + 4))]));
    T_matmul_NN_local[((j_c16 + 26))] = (T_matmul_NN_local[((j_c16 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c16 + 6))]));
    T_matmul_NN_local[((j_c16 + 4))] = (T_matmul_NN_local[((j_c16 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c16)]));
    T_matmul_NN_local[((j_c16 + 12))] = (T_matmul_NN_local[((j_c16 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c16 + 2))]));
    T_matmul_NN_local[((j_c16 + 20))] = (T_matmul_NN_local[((j_c16 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c16 + 4))]));
    T_matmul_NN_local[((j_c16 + 28))] = (T_matmul_NN_local[((j_c16 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c16 + 6))]));
    T_matmul_NN_local[((j_c16 + 6))] = (T_matmul_NN_local[((j_c16 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c16)]));
    T_matmul_NN_local[((j_c16 + 14))] = (T_matmul_NN_local[((j_c16 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c16 + 2))]));
    T_matmul_NN_local[((j_c16 + 22))] = (T_matmul_NN_local[((j_c16 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c16 + 4))]));
    T_matmul_NN_local[((j_c16 + 30))] = (T_matmul_NN_local[((j_c16 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c16 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 257))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 321))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 385))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 449))];
  for (int ax117 = 0; ax117 < 2; ++ax117) {
    placeholder_d_shared_local1[(ax117)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax117) + 544))];
    placeholder_d_shared_local1[((ax117 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax117) + 552))];
    placeholder_d_shared_local1[((ax117 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax117) + 560))];
    placeholder_d_shared_local1[((ax117 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax117) + 568))];
  }
  for (int j_c17 = 0; j_c17 < 2; ++j_c17) {
    T_matmul_NN_local[(j_c17)] = (T_matmul_NN_local[(j_c17)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c17)]));
    T_matmul_NN_local[((j_c17 + 8))] = (T_matmul_NN_local[((j_c17 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c17 + 2))]));
    T_matmul_NN_local[((j_c17 + 16))] = (T_matmul_NN_local[((j_c17 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c17 + 4))]));
    T_matmul_NN_local[((j_c17 + 24))] = (T_matmul_NN_local[((j_c17 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c17 + 6))]));
    T_matmul_NN_local[((j_c17 + 2))] = (T_matmul_NN_local[((j_c17 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c17)]));
    T_matmul_NN_local[((j_c17 + 10))] = (T_matmul_NN_local[((j_c17 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c17 + 2))]));
    T_matmul_NN_local[((j_c17 + 18))] = (T_matmul_NN_local[((j_c17 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c17 + 4))]));
    T_matmul_NN_local[((j_c17 + 26))] = (T_matmul_NN_local[((j_c17 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c17 + 6))]));
    T_matmul_NN_local[((j_c17 + 4))] = (T_matmul_NN_local[((j_c17 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c17)]));
    T_matmul_NN_local[((j_c17 + 12))] = (T_matmul_NN_local[((j_c17 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c17 + 2))]));
    T_matmul_NN_local[((j_c17 + 20))] = (T_matmul_NN_local[((j_c17 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c17 + 4))]));
    T_matmul_NN_local[((j_c17 + 28))] = (T_matmul_NN_local[((j_c17 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c17 + 6))]));
    T_matmul_NN_local[((j_c17 + 6))] = (T_matmul_NN_local[((j_c17 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c17)]));
    T_matmul_NN_local[((j_c17 + 14))] = (T_matmul_NN_local[((j_c17 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c17 + 2))]));
    T_matmul_NN_local[((j_c17 + 22))] = (T_matmul_NN_local[((j_c17 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c17 + 4))]));
    T_matmul_NN_local[((j_c17 + 30))] = (T_matmul_NN_local[((j_c17 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c17 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 258))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 322))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 386))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 450))];
  for (int ax118 = 0; ax118 < 2; ++ax118) {
    placeholder_d_shared_local1[(ax118)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax118) + 576))];
    placeholder_d_shared_local1[((ax118 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax118) + 584))];
    placeholder_d_shared_local1[((ax118 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax118) + 592))];
    placeholder_d_shared_local1[((ax118 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax118) + 600))];
  }
  for (int j_c18 = 0; j_c18 < 2; ++j_c18) {
    T_matmul_NN_local[(j_c18)] = (T_matmul_NN_local[(j_c18)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c18)]));
    T_matmul_NN_local[((j_c18 + 8))] = (T_matmul_NN_local[((j_c18 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c18 + 2))]));
    T_matmul_NN_local[((j_c18 + 16))] = (T_matmul_NN_local[((j_c18 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c18 + 4))]));
    T_matmul_NN_local[((j_c18 + 24))] = (T_matmul_NN_local[((j_c18 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c18 + 6))]));
    T_matmul_NN_local[((j_c18 + 2))] = (T_matmul_NN_local[((j_c18 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c18)]));
    T_matmul_NN_local[((j_c18 + 10))] = (T_matmul_NN_local[((j_c18 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c18 + 2))]));
    T_matmul_NN_local[((j_c18 + 18))] = (T_matmul_NN_local[((j_c18 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c18 + 4))]));
    T_matmul_NN_local[((j_c18 + 26))] = (T_matmul_NN_local[((j_c18 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c18 + 6))]));
    T_matmul_NN_local[((j_c18 + 4))] = (T_matmul_NN_local[((j_c18 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c18)]));
    T_matmul_NN_local[((j_c18 + 12))] = (T_matmul_NN_local[((j_c18 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c18 + 2))]));
    T_matmul_NN_local[((j_c18 + 20))] = (T_matmul_NN_local[((j_c18 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c18 + 4))]));
    T_matmul_NN_local[((j_c18 + 28))] = (T_matmul_NN_local[((j_c18 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c18 + 6))]));
    T_matmul_NN_local[((j_c18 + 6))] = (T_matmul_NN_local[((j_c18 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c18)]));
    T_matmul_NN_local[((j_c18 + 14))] = (T_matmul_NN_local[((j_c18 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c18 + 2))]));
    T_matmul_NN_local[((j_c18 + 22))] = (T_matmul_NN_local[((j_c18 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c18 + 4))]));
    T_matmul_NN_local[((j_c18 + 30))] = (T_matmul_NN_local[((j_c18 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c18 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 259))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 323))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 387))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 451))];
  for (int ax119 = 0; ax119 < 2; ++ax119) {
    placeholder_d_shared_local1[(ax119)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax119) + 608))];
    placeholder_d_shared_local1[((ax119 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax119) + 616))];
    placeholder_d_shared_local1[((ax119 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax119) + 624))];
    placeholder_d_shared_local1[((ax119 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax119) + 632))];
  }
  for (int j_c19 = 0; j_c19 < 2; ++j_c19) {
    T_matmul_NN_local[(j_c19)] = (T_matmul_NN_local[(j_c19)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c19)]));
    T_matmul_NN_local[((j_c19 + 8))] = (T_matmul_NN_local[((j_c19 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c19 + 2))]));
    T_matmul_NN_local[((j_c19 + 16))] = (T_matmul_NN_local[((j_c19 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c19 + 4))]));
    T_matmul_NN_local[((j_c19 + 24))] = (T_matmul_NN_local[((j_c19 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c19 + 6))]));
    T_matmul_NN_local[((j_c19 + 2))] = (T_matmul_NN_local[((j_c19 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c19)]));
    T_matmul_NN_local[((j_c19 + 10))] = (T_matmul_NN_local[((j_c19 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c19 + 2))]));
    T_matmul_NN_local[((j_c19 + 18))] = (T_matmul_NN_local[((j_c19 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c19 + 4))]));
    T_matmul_NN_local[((j_c19 + 26))] = (T_matmul_NN_local[((j_c19 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c19 + 6))]));
    T_matmul_NN_local[((j_c19 + 4))] = (T_matmul_NN_local[((j_c19 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c19)]));
    T_matmul_NN_local[((j_c19 + 12))] = (T_matmul_NN_local[((j_c19 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c19 + 2))]));
    T_matmul_NN_local[((j_c19 + 20))] = (T_matmul_NN_local[((j_c19 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c19 + 4))]));
    T_matmul_NN_local[((j_c19 + 28))] = (T_matmul_NN_local[((j_c19 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c19 + 6))]));
    T_matmul_NN_local[((j_c19 + 6))] = (T_matmul_NN_local[((j_c19 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c19)]));
    T_matmul_NN_local[((j_c19 + 14))] = (T_matmul_NN_local[((j_c19 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c19 + 2))]));
    T_matmul_NN_local[((j_c19 + 22))] = (T_matmul_NN_local[((j_c19 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c19 + 4))]));
    T_matmul_NN_local[((j_c19 + 30))] = (T_matmul_NN_local[((j_c19 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c19 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 260))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 324))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 388))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 452))];
  for (int ax120 = 0; ax120 < 2; ++ax120) {
    placeholder_d_shared_local1[(ax120)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax120) + 640))];
    placeholder_d_shared_local1[((ax120 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax120) + 648))];
    placeholder_d_shared_local1[((ax120 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax120) + 656))];
    placeholder_d_shared_local1[((ax120 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax120) + 664))];
  }
  for (int j_c20 = 0; j_c20 < 2; ++j_c20) {
    T_matmul_NN_local[(j_c20)] = (T_matmul_NN_local[(j_c20)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c20)]));
    T_matmul_NN_local[((j_c20 + 8))] = (T_matmul_NN_local[((j_c20 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c20 + 2))]));
    T_matmul_NN_local[((j_c20 + 16))] = (T_matmul_NN_local[((j_c20 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c20 + 4))]));
    T_matmul_NN_local[((j_c20 + 24))] = (T_matmul_NN_local[((j_c20 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c20 + 6))]));
    T_matmul_NN_local[((j_c20 + 2))] = (T_matmul_NN_local[((j_c20 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c20)]));
    T_matmul_NN_local[((j_c20 + 10))] = (T_matmul_NN_local[((j_c20 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c20 + 2))]));
    T_matmul_NN_local[((j_c20 + 18))] = (T_matmul_NN_local[((j_c20 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c20 + 4))]));
    T_matmul_NN_local[((j_c20 + 26))] = (T_matmul_NN_local[((j_c20 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c20 + 6))]));
    T_matmul_NN_local[((j_c20 + 4))] = (T_matmul_NN_local[((j_c20 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c20)]));
    T_matmul_NN_local[((j_c20 + 12))] = (T_matmul_NN_local[((j_c20 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c20 + 2))]));
    T_matmul_NN_local[((j_c20 + 20))] = (T_matmul_NN_local[((j_c20 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c20 + 4))]));
    T_matmul_NN_local[((j_c20 + 28))] = (T_matmul_NN_local[((j_c20 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c20 + 6))]));
    T_matmul_NN_local[((j_c20 + 6))] = (T_matmul_NN_local[((j_c20 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c20)]));
    T_matmul_NN_local[((j_c20 + 14))] = (T_matmul_NN_local[((j_c20 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c20 + 2))]));
    T_matmul_NN_local[((j_c20 + 22))] = (T_matmul_NN_local[((j_c20 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c20 + 4))]));
    T_matmul_NN_local[((j_c20 + 30))] = (T_matmul_NN_local[((j_c20 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c20 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 261))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 325))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 389))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 453))];
  for (int ax121 = 0; ax121 < 2; ++ax121) {
    placeholder_d_shared_local1[(ax121)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax121) + 672))];
    placeholder_d_shared_local1[((ax121 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax121) + 680))];
    placeholder_d_shared_local1[((ax121 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax121) + 688))];
    placeholder_d_shared_local1[((ax121 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax121) + 696))];
  }
  for (int j_c21 = 0; j_c21 < 2; ++j_c21) {
    T_matmul_NN_local[(j_c21)] = (T_matmul_NN_local[(j_c21)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c21)]));
    T_matmul_NN_local[((j_c21 + 8))] = (T_matmul_NN_local[((j_c21 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c21 + 2))]));
    T_matmul_NN_local[((j_c21 + 16))] = (T_matmul_NN_local[((j_c21 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c21 + 4))]));
    T_matmul_NN_local[((j_c21 + 24))] = (T_matmul_NN_local[((j_c21 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c21 + 6))]));
    T_matmul_NN_local[((j_c21 + 2))] = (T_matmul_NN_local[((j_c21 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c21)]));
    T_matmul_NN_local[((j_c21 + 10))] = (T_matmul_NN_local[((j_c21 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c21 + 2))]));
    T_matmul_NN_local[((j_c21 + 18))] = (T_matmul_NN_local[((j_c21 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c21 + 4))]));
    T_matmul_NN_local[((j_c21 + 26))] = (T_matmul_NN_local[((j_c21 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c21 + 6))]));
    T_matmul_NN_local[((j_c21 + 4))] = (T_matmul_NN_local[((j_c21 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c21)]));
    T_matmul_NN_local[((j_c21 + 12))] = (T_matmul_NN_local[((j_c21 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c21 + 2))]));
    T_matmul_NN_local[((j_c21 + 20))] = (T_matmul_NN_local[((j_c21 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c21 + 4))]));
    T_matmul_NN_local[((j_c21 + 28))] = (T_matmul_NN_local[((j_c21 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c21 + 6))]));
    T_matmul_NN_local[((j_c21 + 6))] = (T_matmul_NN_local[((j_c21 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c21)]));
    T_matmul_NN_local[((j_c21 + 14))] = (T_matmul_NN_local[((j_c21 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c21 + 2))]));
    T_matmul_NN_local[((j_c21 + 22))] = (T_matmul_NN_local[((j_c21 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c21 + 4))]));
    T_matmul_NN_local[((j_c21 + 30))] = (T_matmul_NN_local[((j_c21 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c21 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 262))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 326))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 390))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 454))];
  for (int ax122 = 0; ax122 < 2; ++ax122) {
    placeholder_d_shared_local1[(ax122)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax122) + 704))];
    placeholder_d_shared_local1[((ax122 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax122) + 712))];
    placeholder_d_shared_local1[((ax122 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax122) + 720))];
    placeholder_d_shared_local1[((ax122 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax122) + 728))];
  }
  for (int j_c22 = 0; j_c22 < 2; ++j_c22) {
    T_matmul_NN_local[(j_c22)] = (T_matmul_NN_local[(j_c22)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c22)]));
    T_matmul_NN_local[((j_c22 + 8))] = (T_matmul_NN_local[((j_c22 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c22 + 2))]));
    T_matmul_NN_local[((j_c22 + 16))] = (T_matmul_NN_local[((j_c22 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c22 + 4))]));
    T_matmul_NN_local[((j_c22 + 24))] = (T_matmul_NN_local[((j_c22 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c22 + 6))]));
    T_matmul_NN_local[((j_c22 + 2))] = (T_matmul_NN_local[((j_c22 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c22)]));
    T_matmul_NN_local[((j_c22 + 10))] = (T_matmul_NN_local[((j_c22 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c22 + 2))]));
    T_matmul_NN_local[((j_c22 + 18))] = (T_matmul_NN_local[((j_c22 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c22 + 4))]));
    T_matmul_NN_local[((j_c22 + 26))] = (T_matmul_NN_local[((j_c22 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c22 + 6))]));
    T_matmul_NN_local[((j_c22 + 4))] = (T_matmul_NN_local[((j_c22 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c22)]));
    T_matmul_NN_local[((j_c22 + 12))] = (T_matmul_NN_local[((j_c22 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c22 + 2))]));
    T_matmul_NN_local[((j_c22 + 20))] = (T_matmul_NN_local[((j_c22 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c22 + 4))]));
    T_matmul_NN_local[((j_c22 + 28))] = (T_matmul_NN_local[((j_c22 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c22 + 6))]));
    T_matmul_NN_local[((j_c22 + 6))] = (T_matmul_NN_local[((j_c22 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c22)]));
    T_matmul_NN_local[((j_c22 + 14))] = (T_matmul_NN_local[((j_c22 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c22 + 2))]));
    T_matmul_NN_local[((j_c22 + 22))] = (T_matmul_NN_local[((j_c22 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c22 + 4))]));
    T_matmul_NN_local[((j_c22 + 30))] = (T_matmul_NN_local[((j_c22 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c22 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 263))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 327))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 391))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 455))];
  for (int ax123 = 0; ax123 < 2; ++ax123) {
    placeholder_d_shared_local1[(ax123)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax123) + 736))];
    placeholder_d_shared_local1[((ax123 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax123) + 744))];
    placeholder_d_shared_local1[((ax123 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax123) + 752))];
    placeholder_d_shared_local1[((ax123 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax123) + 760))];
  }
  for (int j_c23 = 0; j_c23 < 2; ++j_c23) {
    T_matmul_NN_local[(j_c23)] = (T_matmul_NN_local[(j_c23)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c23)]));
    T_matmul_NN_local[((j_c23 + 8))] = (T_matmul_NN_local[((j_c23 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c23 + 2))]));
    T_matmul_NN_local[((j_c23 + 16))] = (T_matmul_NN_local[((j_c23 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c23 + 4))]));
    T_matmul_NN_local[((j_c23 + 24))] = (T_matmul_NN_local[((j_c23 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c23 + 6))]));
    T_matmul_NN_local[((j_c23 + 2))] = (T_matmul_NN_local[((j_c23 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c23)]));
    T_matmul_NN_local[((j_c23 + 10))] = (T_matmul_NN_local[((j_c23 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c23 + 2))]));
    T_matmul_NN_local[((j_c23 + 18))] = (T_matmul_NN_local[((j_c23 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c23 + 4))]));
    T_matmul_NN_local[((j_c23 + 26))] = (T_matmul_NN_local[((j_c23 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c23 + 6))]));
    T_matmul_NN_local[((j_c23 + 4))] = (T_matmul_NN_local[((j_c23 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c23)]));
    T_matmul_NN_local[((j_c23 + 12))] = (T_matmul_NN_local[((j_c23 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c23 + 2))]));
    T_matmul_NN_local[((j_c23 + 20))] = (T_matmul_NN_local[((j_c23 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c23 + 4))]));
    T_matmul_NN_local[((j_c23 + 28))] = (T_matmul_NN_local[((j_c23 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c23 + 6))]));
    T_matmul_NN_local[((j_c23 + 6))] = (T_matmul_NN_local[((j_c23 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c23)]));
    T_matmul_NN_local[((j_c23 + 14))] = (T_matmul_NN_local[((j_c23 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c23 + 2))]));
    T_matmul_NN_local[((j_c23 + 22))] = (T_matmul_NN_local[((j_c23 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c23 + 4))]));
    T_matmul_NN_local[((j_c23 + 30))] = (T_matmul_NN_local[((j_c23 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c23 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 264))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 328))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 392))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 456))];
  for (int ax124 = 0; ax124 < 2; ++ax124) {
    placeholder_d_shared_local1[(ax124)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax124) + 768))];
    placeholder_d_shared_local1[((ax124 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax124) + 776))];
    placeholder_d_shared_local1[((ax124 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax124) + 784))];
    placeholder_d_shared_local1[((ax124 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax124) + 792))];
  }
  for (int j_c24 = 0; j_c24 < 2; ++j_c24) {
    T_matmul_NN_local[(j_c24)] = (T_matmul_NN_local[(j_c24)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c24)]));
    T_matmul_NN_local[((j_c24 + 8))] = (T_matmul_NN_local[((j_c24 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c24 + 2))]));
    T_matmul_NN_local[((j_c24 + 16))] = (T_matmul_NN_local[((j_c24 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c24 + 4))]));
    T_matmul_NN_local[((j_c24 + 24))] = (T_matmul_NN_local[((j_c24 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c24 + 6))]));
    T_matmul_NN_local[((j_c24 + 2))] = (T_matmul_NN_local[((j_c24 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c24)]));
    T_matmul_NN_local[((j_c24 + 10))] = (T_matmul_NN_local[((j_c24 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c24 + 2))]));
    T_matmul_NN_local[((j_c24 + 18))] = (T_matmul_NN_local[((j_c24 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c24 + 4))]));
    T_matmul_NN_local[((j_c24 + 26))] = (T_matmul_NN_local[((j_c24 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c24 + 6))]));
    T_matmul_NN_local[((j_c24 + 4))] = (T_matmul_NN_local[((j_c24 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c24)]));
    T_matmul_NN_local[((j_c24 + 12))] = (T_matmul_NN_local[((j_c24 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c24 + 2))]));
    T_matmul_NN_local[((j_c24 + 20))] = (T_matmul_NN_local[((j_c24 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c24 + 4))]));
    T_matmul_NN_local[((j_c24 + 28))] = (T_matmul_NN_local[((j_c24 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c24 + 6))]));
    T_matmul_NN_local[((j_c24 + 6))] = (T_matmul_NN_local[((j_c24 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c24)]));
    T_matmul_NN_local[((j_c24 + 14))] = (T_matmul_NN_local[((j_c24 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c24 + 2))]));
    T_matmul_NN_local[((j_c24 + 22))] = (T_matmul_NN_local[((j_c24 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c24 + 4))]));
    T_matmul_NN_local[((j_c24 + 30))] = (T_matmul_NN_local[((j_c24 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c24 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 265))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 329))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 393))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 457))];
  for (int ax125 = 0; ax125 < 2; ++ax125) {
    placeholder_d_shared_local1[(ax125)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax125) + 800))];
    placeholder_d_shared_local1[((ax125 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax125) + 808))];
    placeholder_d_shared_local1[((ax125 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax125) + 816))];
    placeholder_d_shared_local1[((ax125 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax125) + 824))];
  }
  for (int j_c25 = 0; j_c25 < 2; ++j_c25) {
    T_matmul_NN_local[(j_c25)] = (T_matmul_NN_local[(j_c25)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c25)]));
    T_matmul_NN_local[((j_c25 + 8))] = (T_matmul_NN_local[((j_c25 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c25 + 2))]));
    T_matmul_NN_local[((j_c25 + 16))] = (T_matmul_NN_local[((j_c25 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c25 + 4))]));
    T_matmul_NN_local[((j_c25 + 24))] = (T_matmul_NN_local[((j_c25 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c25 + 6))]));
    T_matmul_NN_local[((j_c25 + 2))] = (T_matmul_NN_local[((j_c25 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c25)]));
    T_matmul_NN_local[((j_c25 + 10))] = (T_matmul_NN_local[((j_c25 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c25 + 2))]));
    T_matmul_NN_local[((j_c25 + 18))] = (T_matmul_NN_local[((j_c25 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c25 + 4))]));
    T_matmul_NN_local[((j_c25 + 26))] = (T_matmul_NN_local[((j_c25 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c25 + 6))]));
    T_matmul_NN_local[((j_c25 + 4))] = (T_matmul_NN_local[((j_c25 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c25)]));
    T_matmul_NN_local[((j_c25 + 12))] = (T_matmul_NN_local[((j_c25 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c25 + 2))]));
    T_matmul_NN_local[((j_c25 + 20))] = (T_matmul_NN_local[((j_c25 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c25 + 4))]));
    T_matmul_NN_local[((j_c25 + 28))] = (T_matmul_NN_local[((j_c25 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c25 + 6))]));
    T_matmul_NN_local[((j_c25 + 6))] = (T_matmul_NN_local[((j_c25 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c25)]));
    T_matmul_NN_local[((j_c25 + 14))] = (T_matmul_NN_local[((j_c25 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c25 + 2))]));
    T_matmul_NN_local[((j_c25 + 22))] = (T_matmul_NN_local[((j_c25 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c25 + 4))]));
    T_matmul_NN_local[((j_c25 + 30))] = (T_matmul_NN_local[((j_c25 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c25 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 266))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 330))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 394))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 458))];
  for (int ax126 = 0; ax126 < 2; ++ax126) {
    placeholder_d_shared_local1[(ax126)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax126) + 832))];
    placeholder_d_shared_local1[((ax126 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax126) + 840))];
    placeholder_d_shared_local1[((ax126 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax126) + 848))];
    placeholder_d_shared_local1[((ax126 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax126) + 856))];
  }
  for (int j_c26 = 0; j_c26 < 2; ++j_c26) {
    T_matmul_NN_local[(j_c26)] = (T_matmul_NN_local[(j_c26)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c26)]));
    T_matmul_NN_local[((j_c26 + 8))] = (T_matmul_NN_local[((j_c26 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c26 + 2))]));
    T_matmul_NN_local[((j_c26 + 16))] = (T_matmul_NN_local[((j_c26 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c26 + 4))]));
    T_matmul_NN_local[((j_c26 + 24))] = (T_matmul_NN_local[((j_c26 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c26 + 6))]));
    T_matmul_NN_local[((j_c26 + 2))] = (T_matmul_NN_local[((j_c26 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c26)]));
    T_matmul_NN_local[((j_c26 + 10))] = (T_matmul_NN_local[((j_c26 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c26 + 2))]));
    T_matmul_NN_local[((j_c26 + 18))] = (T_matmul_NN_local[((j_c26 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c26 + 4))]));
    T_matmul_NN_local[((j_c26 + 26))] = (T_matmul_NN_local[((j_c26 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c26 + 6))]));
    T_matmul_NN_local[((j_c26 + 4))] = (T_matmul_NN_local[((j_c26 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c26)]));
    T_matmul_NN_local[((j_c26 + 12))] = (T_matmul_NN_local[((j_c26 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c26 + 2))]));
    T_matmul_NN_local[((j_c26 + 20))] = (T_matmul_NN_local[((j_c26 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c26 + 4))]));
    T_matmul_NN_local[((j_c26 + 28))] = (T_matmul_NN_local[((j_c26 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c26 + 6))]));
    T_matmul_NN_local[((j_c26 + 6))] = (T_matmul_NN_local[((j_c26 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c26)]));
    T_matmul_NN_local[((j_c26 + 14))] = (T_matmul_NN_local[((j_c26 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c26 + 2))]));
    T_matmul_NN_local[((j_c26 + 22))] = (T_matmul_NN_local[((j_c26 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c26 + 4))]));
    T_matmul_NN_local[((j_c26 + 30))] = (T_matmul_NN_local[((j_c26 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c26 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 267))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 331))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 395))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 459))];
  for (int ax127 = 0; ax127 < 2; ++ax127) {
    placeholder_d_shared_local1[(ax127)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax127) + 864))];
    placeholder_d_shared_local1[((ax127 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax127) + 872))];
    placeholder_d_shared_local1[((ax127 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax127) + 880))];
    placeholder_d_shared_local1[((ax127 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax127) + 888))];
  }
  for (int j_c27 = 0; j_c27 < 2; ++j_c27) {
    T_matmul_NN_local[(j_c27)] = (T_matmul_NN_local[(j_c27)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c27)]));
    T_matmul_NN_local[((j_c27 + 8))] = (T_matmul_NN_local[((j_c27 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c27 + 2))]));
    T_matmul_NN_local[((j_c27 + 16))] = (T_matmul_NN_local[((j_c27 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c27 + 4))]));
    T_matmul_NN_local[((j_c27 + 24))] = (T_matmul_NN_local[((j_c27 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c27 + 6))]));
    T_matmul_NN_local[((j_c27 + 2))] = (T_matmul_NN_local[((j_c27 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c27)]));
    T_matmul_NN_local[((j_c27 + 10))] = (T_matmul_NN_local[((j_c27 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c27 + 2))]));
    T_matmul_NN_local[((j_c27 + 18))] = (T_matmul_NN_local[((j_c27 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c27 + 4))]));
    T_matmul_NN_local[((j_c27 + 26))] = (T_matmul_NN_local[((j_c27 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c27 + 6))]));
    T_matmul_NN_local[((j_c27 + 4))] = (T_matmul_NN_local[((j_c27 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c27)]));
    T_matmul_NN_local[((j_c27 + 12))] = (T_matmul_NN_local[((j_c27 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c27 + 2))]));
    T_matmul_NN_local[((j_c27 + 20))] = (T_matmul_NN_local[((j_c27 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c27 + 4))]));
    T_matmul_NN_local[((j_c27 + 28))] = (T_matmul_NN_local[((j_c27 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c27 + 6))]));
    T_matmul_NN_local[((j_c27 + 6))] = (T_matmul_NN_local[((j_c27 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c27)]));
    T_matmul_NN_local[((j_c27 + 14))] = (T_matmul_NN_local[((j_c27 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c27 + 2))]));
    T_matmul_NN_local[((j_c27 + 22))] = (T_matmul_NN_local[((j_c27 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c27 + 4))]));
    T_matmul_NN_local[((j_c27 + 30))] = (T_matmul_NN_local[((j_c27 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c27 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 268))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 332))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 396))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 460))];
  for (int ax128 = 0; ax128 < 2; ++ax128) {
    placeholder_d_shared_local1[(ax128)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax128) + 896))];
    placeholder_d_shared_local1[((ax128 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax128) + 904))];
    placeholder_d_shared_local1[((ax128 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax128) + 912))];
    placeholder_d_shared_local1[((ax128 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax128) + 920))];
  }
  for (int j_c28 = 0; j_c28 < 2; ++j_c28) {
    T_matmul_NN_local[(j_c28)] = (T_matmul_NN_local[(j_c28)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c28)]));
    T_matmul_NN_local[((j_c28 + 8))] = (T_matmul_NN_local[((j_c28 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c28 + 2))]));
    T_matmul_NN_local[((j_c28 + 16))] = (T_matmul_NN_local[((j_c28 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c28 + 4))]));
    T_matmul_NN_local[((j_c28 + 24))] = (T_matmul_NN_local[((j_c28 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c28 + 6))]));
    T_matmul_NN_local[((j_c28 + 2))] = (T_matmul_NN_local[((j_c28 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c28)]));
    T_matmul_NN_local[((j_c28 + 10))] = (T_matmul_NN_local[((j_c28 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c28 + 2))]));
    T_matmul_NN_local[((j_c28 + 18))] = (T_matmul_NN_local[((j_c28 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c28 + 4))]));
    T_matmul_NN_local[((j_c28 + 26))] = (T_matmul_NN_local[((j_c28 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c28 + 6))]));
    T_matmul_NN_local[((j_c28 + 4))] = (T_matmul_NN_local[((j_c28 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c28)]));
    T_matmul_NN_local[((j_c28 + 12))] = (T_matmul_NN_local[((j_c28 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c28 + 2))]));
    T_matmul_NN_local[((j_c28 + 20))] = (T_matmul_NN_local[((j_c28 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c28 + 4))]));
    T_matmul_NN_local[((j_c28 + 28))] = (T_matmul_NN_local[((j_c28 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c28 + 6))]));
    T_matmul_NN_local[((j_c28 + 6))] = (T_matmul_NN_local[((j_c28 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c28)]));
    T_matmul_NN_local[((j_c28 + 14))] = (T_matmul_NN_local[((j_c28 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c28 + 2))]));
    T_matmul_NN_local[((j_c28 + 22))] = (T_matmul_NN_local[((j_c28 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c28 + 4))]));
    T_matmul_NN_local[((j_c28 + 30))] = (T_matmul_NN_local[((j_c28 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c28 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 269))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 333))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 397))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 461))];
  for (int ax129 = 0; ax129 < 2; ++ax129) {
    placeholder_d_shared_local1[(ax129)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax129) + 928))];
    placeholder_d_shared_local1[((ax129 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax129) + 936))];
    placeholder_d_shared_local1[((ax129 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax129) + 944))];
    placeholder_d_shared_local1[((ax129 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax129) + 952))];
  }
  for (int j_c29 = 0; j_c29 < 2; ++j_c29) {
    T_matmul_NN_local[(j_c29)] = (T_matmul_NN_local[(j_c29)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c29)]));
    T_matmul_NN_local[((j_c29 + 8))] = (T_matmul_NN_local[((j_c29 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c29 + 2))]));
    T_matmul_NN_local[((j_c29 + 16))] = (T_matmul_NN_local[((j_c29 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c29 + 4))]));
    T_matmul_NN_local[((j_c29 + 24))] = (T_matmul_NN_local[((j_c29 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c29 + 6))]));
    T_matmul_NN_local[((j_c29 + 2))] = (T_matmul_NN_local[((j_c29 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c29)]));
    T_matmul_NN_local[((j_c29 + 10))] = (T_matmul_NN_local[((j_c29 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c29 + 2))]));
    T_matmul_NN_local[((j_c29 + 18))] = (T_matmul_NN_local[((j_c29 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c29 + 4))]));
    T_matmul_NN_local[((j_c29 + 26))] = (T_matmul_NN_local[((j_c29 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c29 + 6))]));
    T_matmul_NN_local[((j_c29 + 4))] = (T_matmul_NN_local[((j_c29 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c29)]));
    T_matmul_NN_local[((j_c29 + 12))] = (T_matmul_NN_local[((j_c29 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c29 + 2))]));
    T_matmul_NN_local[((j_c29 + 20))] = (T_matmul_NN_local[((j_c29 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c29 + 4))]));
    T_matmul_NN_local[((j_c29 + 28))] = (T_matmul_NN_local[((j_c29 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c29 + 6))]));
    T_matmul_NN_local[((j_c29 + 6))] = (T_matmul_NN_local[((j_c29 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c29)]));
    T_matmul_NN_local[((j_c29 + 14))] = (T_matmul_NN_local[((j_c29 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c29 + 2))]));
    T_matmul_NN_local[((j_c29 + 22))] = (T_matmul_NN_local[((j_c29 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c29 + 4))]));
    T_matmul_NN_local[((j_c29 + 30))] = (T_matmul_NN_local[((j_c29 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c29 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 270))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 334))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 398))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 462))];
  for (int ax130 = 0; ax130 < 2; ++ax130) {
    placeholder_d_shared_local1[(ax130)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax130) + 960))];
    placeholder_d_shared_local1[((ax130 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax130) + 968))];
    placeholder_d_shared_local1[((ax130 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax130) + 976))];
    placeholder_d_shared_local1[((ax130 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax130) + 984))];
  }
  for (int j_c30 = 0; j_c30 < 2; ++j_c30) {
    T_matmul_NN_local[(j_c30)] = (T_matmul_NN_local[(j_c30)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c30)]));
    T_matmul_NN_local[((j_c30 + 8))] = (T_matmul_NN_local[((j_c30 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c30 + 2))]));
    T_matmul_NN_local[((j_c30 + 16))] = (T_matmul_NN_local[((j_c30 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c30 + 4))]));
    T_matmul_NN_local[((j_c30 + 24))] = (T_matmul_NN_local[((j_c30 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c30 + 6))]));
    T_matmul_NN_local[((j_c30 + 2))] = (T_matmul_NN_local[((j_c30 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c30)]));
    T_matmul_NN_local[((j_c30 + 10))] = (T_matmul_NN_local[((j_c30 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c30 + 2))]));
    T_matmul_NN_local[((j_c30 + 18))] = (T_matmul_NN_local[((j_c30 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c30 + 4))]));
    T_matmul_NN_local[((j_c30 + 26))] = (T_matmul_NN_local[((j_c30 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c30 + 6))]));
    T_matmul_NN_local[((j_c30 + 4))] = (T_matmul_NN_local[((j_c30 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c30)]));
    T_matmul_NN_local[((j_c30 + 12))] = (T_matmul_NN_local[((j_c30 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c30 + 2))]));
    T_matmul_NN_local[((j_c30 + 20))] = (T_matmul_NN_local[((j_c30 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c30 + 4))]));
    T_matmul_NN_local[((j_c30 + 28))] = (T_matmul_NN_local[((j_c30 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c30 + 6))]));
    T_matmul_NN_local[((j_c30 + 6))] = (T_matmul_NN_local[((j_c30 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c30)]));
    T_matmul_NN_local[((j_c30 + 14))] = (T_matmul_NN_local[((j_c30 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c30 + 2))]));
    T_matmul_NN_local[((j_c30 + 22))] = (T_matmul_NN_local[((j_c30 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c30 + 4))]));
    T_matmul_NN_local[((j_c30 + 30))] = (T_matmul_NN_local[((j_c30 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c30 + 6))]));
  }
  placeholder_shared_local1[(0)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 271))];
  placeholder_shared_local1[(1)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 335))];
  placeholder_shared_local1[(2)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 399))];
  placeholder_shared_local1[(3)] = placeholder_shared[(((((int)threadIdx.x) * 16) + 463))];
  for (int ax131 = 0; ax131 < 2; ++ax131) {
    placeholder_d_shared_local1[(ax131)] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax131) + 992))];
    placeholder_d_shared_local1[((ax131 + 2))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax131) + 1000))];
    placeholder_d_shared_local1[((ax131 + 4))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax131) + 1008))];
    placeholder_d_shared_local1[((ax131 + 6))] = placeholder_d_shared[((((((int)threadIdx.y) * 2) + ax131) + 1016))];
  }
  for (int j_c31 = 0; j_c31 < 2; ++j_c31) {
    T_matmul_NN_local[(j_c31)] = (T_matmul_NN_local[(j_c31)] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[(j_c31)]));
    T_matmul_NN_local[((j_c31 + 8))] = (T_matmul_NN_local[((j_c31 + 8))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c31 + 2))]));
    T_matmul_NN_local[((j_c31 + 16))] = (T_matmul_NN_local[((j_c31 + 16))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c31 + 4))]));
    T_matmul_NN_local[((j_c31 + 24))] = (T_matmul_NN_local[((j_c31 + 24))] + (placeholder_shared_local1[(0)] * placeholder_d_shared_local1[((j_c31 + 6))]));
    T_matmul_NN_local[((j_c31 + 2))] = (T_matmul_NN_local[((j_c31 + 2))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[(j_c31)]));
    T_matmul_NN_local[((j_c31 + 10))] = (T_matmul_NN_local[((j_c31 + 10))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c31 + 2))]));
    T_matmul_NN_local[((j_c31 + 18))] = (T_matmul_NN_local[((j_c31 + 18))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c31 + 4))]));
    T_matmul_NN_local[((j_c31 + 26))] = (T_matmul_NN_local[((j_c31 + 26))] + (placeholder_shared_local1[(1)] * placeholder_d_shared_local1[((j_c31 + 6))]));
    T_matmul_NN_local[((j_c31 + 4))] = (T_matmul_NN_local[((j_c31 + 4))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[(j_c31)]));
    T_matmul_NN_local[((j_c31 + 12))] = (T_matmul_NN_local[((j_c31 + 12))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c31 + 2))]));
    T_matmul_NN_local[((j_c31 + 20))] = (T_matmul_NN_local[((j_c31 + 20))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c31 + 4))]));
    T_matmul_NN_local[((j_c31 + 28))] = (T_matmul_NN_local[((j_c31 + 28))] + (placeholder_shared_local1[(2)] * placeholder_d_shared_local1[((j_c31 + 6))]));
    T_matmul_NN_local[((j_c31 + 6))] = (T_matmul_NN_local[((j_c31 + 6))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[(j_c31)]));
    T_matmul_NN_local[((j_c31 + 14))] = (T_matmul_NN_local[((j_c31 + 14))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c31 + 2))]));
    T_matmul_NN_local[((j_c31 + 22))] = (T_matmul_NN_local[((j_c31 + 22))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c31 + 4))]));
    T_matmul_NN_local[((j_c31 + 30))] = (T_matmul_NN_local[((j_c31 + 30))] + (placeholder_shared_local1[(3)] * placeholder_d_shared_local1[((j_c31 + 6))]));
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 2; ++j_inner_inner_inner) {
    T_matmul_NN[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner))] = T_matmul_NN_local[(j_inner_inner_inner)];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 8))] = T_matmul_NN_local[((j_inner_inner_inner + 8))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 16))] = T_matmul_NN_local[((j_inner_inner_inner + 16))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 24))] = T_matmul_NN_local[((j_inner_inner_inner + 24))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 4096))] = T_matmul_NN_local[((j_inner_inner_inner + 2))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 4104))] = T_matmul_NN_local[((j_inner_inner_inner + 10))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 4112))] = T_matmul_NN_local[((j_inner_inner_inner + 18))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 4120))] = T_matmul_NN_local[((j_inner_inner_inner + 26))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 8192))] = T_matmul_NN_local[((j_inner_inner_inner + 4))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 8200))] = T_matmul_NN_local[((j_inner_inner_inner + 12))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 8208))] = T_matmul_NN_local[((j_inner_inner_inner + 20))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 8216))] = T_matmul_NN_local[((j_inner_inner_inner + 28))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 12288))] = T_matmul_NN_local[((j_inner_inner_inner + 6))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 12296))] = T_matmul_NN_local[((j_inner_inner_inner + 14))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 12304))] = T_matmul_NN_local[((j_inner_inner_inner + 22))];
    T_matmul_NN[(((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 2)) + j_inner_inner_inner) + 12312))] = T_matmul_NN_local[((j_inner_inner_inner + 30))];
  }
}

