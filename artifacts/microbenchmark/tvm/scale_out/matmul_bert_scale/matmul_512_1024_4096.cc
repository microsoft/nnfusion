//2_128_1_8_8_1
//512_1024_4096
//dim3 grid(2, 128, 1);
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
  float T_matmul_NN_local[128];
  __shared__ float placeholder_shared[512];
  __shared__ float placeholder_d_shared[64];
  float placeholder_shared_local[32];
  float placeholder_d_shared_local[4];
  float placeholder_shared_local1[32];
  float placeholder_d_shared_local1[4];
  for (int i_c_init = 0; i_c_init < 16; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
      T_matmul_NN_local[(((i_c_init * 4) + j_c_init))] = 0.000000e+00f;
      T_matmul_NN_local[((((i_c_init * 4) + j_c_init) + 64))] = 0.000000e+00f;
    }
  }
  for (int ax0_inner = 0; ax0_inner < 32; ++ax0_inner) {
    for (int ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
      if (((((int)threadIdx.x) * 4) + ax1_inner_inner) < 1) {
        placeholder_shared[(((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + ax0_inner) + ax1_inner_inner))] = placeholder[((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.y) * 32768)) + (ax0_inner * 1024)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner))];
      }
    }
  }
  for (int ax1_inner_inner1 = 0; ax1_inner_inner1 < 4; ++ax1_inner_inner1) {
    if (((int)threadIdx.y) < 1) {
      placeholder_d_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))] = placeholder1[(((((((int)threadIdx.y) * 4096) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner1))];
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 1023; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_inner1 = 0; ax0_inner1 < 32; ++ax0_inner1) {
      for (int ax1_inner_inner2 = 0; ax1_inner_inner2 < 4; ++ax1_inner_inner2) {
        if (((((int)threadIdx.x) * 4) + ax1_inner_inner2) < 1) {
          placeholder_shared[((((((((k_outer_outer + 1) & 1) * 256) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + ax0_inner1) + ax1_inner_inner2))] = placeholder[((((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.y) * 32768)) + (ax0_inner1 * 1024)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner2) + k_outer_outer) + 1))];
        }
      }
    }
    for (int ax1_inner_inner3 = 0; ax1_inner_inner3 < 4; ++ax1_inner_inner3) {
      if (((int)threadIdx.y) < 1) {
        placeholder_d_shared[(((((((int)threadIdx.y) * 32) + (((k_outer_outer + 1) & 1) * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3))] = placeholder1[(((((((((int)threadIdx.y) * 4096) + (k_outer_outer * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.x) * 4)) + ax1_inner_inner3) + 4096))];
      }
    }
    for (int ax0 = 0; ax0 < 16; ++ax0) {
      placeholder_shared_local[(ax0)] = placeholder_shared[(((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + ax0))];
      placeholder_shared_local[((ax0 + 16))] = placeholder_shared[((((((k_outer_outer & 1) * 256) + (((int)threadIdx.x) * 16)) + ax0) + 128))];
    }
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      placeholder_d_shared_local[(ax1)] = placeholder_d_shared[(((((k_outer_outer & 1) * 32) + (((int)threadIdx.y) * 4)) + ax1))];
    }
    for (int i_c = 0; i_c < 16; ++i_c) {
      for (int j_c = 0; j_c < 4; ++j_c) {
        T_matmul_NN_local[(((i_c * 4) + j_c))] = (T_matmul_NN_local[(((i_c * 4) + j_c))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(j_c)]));
        T_matmul_NN_local[((((i_c * 4) + j_c) + 64))] = (T_matmul_NN_local[((((i_c * 4) + j_c) + 64))] + (placeholder_shared_local[((i_c + 16))] * placeholder_d_shared_local[(j_c)]));
      }
    }
  }
  __syncthreads();
  for (int ax01 = 0; ax01 < 16; ++ax01) {
    placeholder_shared_local1[(ax01)] = placeholder_shared[((((((int)threadIdx.x) * 16) + ax01) + 256))];
    placeholder_shared_local1[((ax01 + 16))] = placeholder_shared[((((((int)threadIdx.x) * 16) + ax01) + 384))];
  }
  for (int ax11 = 0; ax11 < 4; ++ax11) {
    placeholder_d_shared_local1[(ax11)] = placeholder_d_shared[((((((int)threadIdx.y) * 4) + ax11) + 32))];
  }
  for (int i_c1 = 0; i_c1 < 16; ++i_c1) {
    for (int j_c1 = 0; j_c1 < 4; ++j_c1) {
      T_matmul_NN_local[(((i_c1 * 4) + j_c1))] = (T_matmul_NN_local[(((i_c1 * 4) + j_c1))] + (placeholder_shared_local1[(i_c1)] * placeholder_d_shared_local1[(j_c1)]));
      T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 64))] = (T_matmul_NN_local[((((i_c1 * 4) + j_c1) + 64))] + (placeholder_shared_local1[((i_c1 + 16))] * placeholder_d_shared_local1[(j_c1)]));
    }
  }
  for (int j_inner_inner_inner = 0; j_inner_inner_inner < 4; ++j_inner_inner_inner) {
    for (int i_inner_inner_inner = 0; i_inner_inner_inner < 16; ++i_inner_inner_inner) {
      T_matmul_NN[(((((((((int)blockIdx.x) * 1048576) + (((int)threadIdx.x) * 65536)) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner))] = T_matmul_NN_local[(((i_inner_inner_inner * 4) + j_inner_inner_inner))];
      T_matmul_NN[((((((((((int)blockIdx.x) * 1048576) + (((int)threadIdx.x) * 65536)) + (i_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.y) * 4)) + j_inner_inner_inner) + 524288))] = T_matmul_NN_local[((((i_inner_inner_inner * 4) + j_inner_inner_inner) + 64))];
    }
  }
}

