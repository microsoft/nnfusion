//1_4_2048_16_16_1
//128_16_512_512_64
//dim3 grid(1, 4, 2048);
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
extern "C" __global__ void __launch_bounds__(256) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NN) {
  float T_batch_matmul_NN_local[32];
  __shared__ float placeholder_shared[4096];
  __shared__ float placeholder_d_shared[2048];
  float placeholder_shared_local[8];
  float placeholder_d_shared_local[4];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
      T_batch_matmul_NN_local[(((i_c_init * 4) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      #pragma unroll
      for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
        placeholder_shared[(((((((int)threadIdx.y) * 256) + (ax1_inner * 32)) + (((int)threadIdx.x) * 2)) + ax2_inner))] = placeholder[((((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 65536)) + (((int)threadIdx.y) * 4096)) + (ax1_inner * 512)) + (k_outer * 32)) + (((int)threadIdx.x) * 2)) + ax2_inner))];
      }
    }
    #pragma unroll
    for (int ax1_inner1 = 0; ax1_inner1 < 2; ++ax1_inner1) {
      #pragma unroll
      for (int ax2_inner1 = 0; ax2_inner1 < 4; ++ax2_inner1) {
        placeholder_d_shared[(((((((int)threadIdx.y) * 128) + (ax1_inner1 * 64)) + (((int)threadIdx.x) * 4)) + ax2_inner1))] = placeholder1[(((((((((int)blockIdx.z) * 32768) + (k_outer * 2048)) + (((int)threadIdx.y) * 128)) + (ax1_inner1 * 64)) + (((int)threadIdx.x) * 4)) + ax2_inner1))];
      }
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        placeholder_shared_local[(ax1)] = placeholder_shared[((((((int)threadIdx.y) * 256) + (ax1 * 32)) + k_inner))];
      }
      #pragma unroll
      for (int ax2 = 0; ax2 < 4; ++ax2) {
        placeholder_d_shared_local[(ax2)] = placeholder_d_shared[((((k_inner * 64) + (((int)threadIdx.x) * 4)) + ax2))];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 4; ++j_c) {
          T_batch_matmul_NN_local[(((i_c * 4) + j_c))] = (T_batch_matmul_NN_local[(((i_c * 4) + j_c))] + (placeholder_shared_local[(i_c)] * placeholder_d_shared_local[(j_c)]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 4; ++j_inner_inner) {
      T_batch_matmul_NN[(((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 8192)) + (((int)threadIdx.y) * 512)) + (i_inner_inner * 64)) + (((int)threadIdx.x) * 4)) + j_inner_inner))] = T_batch_matmul_NN_local[(((i_inner_inner * 4) + j_inner_inner))];
    }
  }
}

