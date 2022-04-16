//65536_1_1_256_1_1
//128_16_512_64_512
//dim3 grid(65536, 1, 1);
//dim3 block(256, 1, 1);

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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NN) {
  float T_batch_matmul_NN_local[32];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[4096];
  T_batch_matmul_NN_local[(0)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(16)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(1)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(17)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(2)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(18)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(3)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(19)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(4)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(20)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(5)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(21)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(6)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(22)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(7)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(23)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(8)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(24)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(9)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(25)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(10)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(26)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(11)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(27)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(12)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(28)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(13)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(29)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(14)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(30)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(15)] = 0.000000e+00f;
  T_batch_matmul_NN_local[(31)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] = A[((((((((int)blockIdx.x) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)))];
    A_shared[((((int)threadIdx.x) + 256))] = A[(((((((((int)blockIdx.x) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 512))];
    A_shared[((((int)threadIdx.x) + 512))] = A[(((((((((int)blockIdx.x) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1024))];
    A_shared[((((int)threadIdx.x) + 768))] = A[(((((((((int)blockIdx.x) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1536))];
    A_shared[((((int)threadIdx.x) + 1024))] = A[(((((((((int)blockIdx.x) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048))];
    A_shared[((((int)threadIdx.x) + 1280))] = A[(((((((((int)blockIdx.x) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2560))];
    A_shared[((((int)threadIdx.x) + 1536))] = A[(((((((((int)blockIdx.x) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3072))];
    A_shared[((((int)threadIdx.x) + 1792))] = A[(((((((((int)blockIdx.x) >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3584))];
    ((float4*)(B_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(B + (((((((((int)blockIdx.x) >> 5) * 32768) + (k_outer_outer * 16384)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 3) * 128)) + ((((int)threadIdx.x) & 31) * 4)))))[0];
    ((float4*)(B_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(B + ((((((((((int)blockIdx.x) >> 5) * 32768) + (k_outer_outer * 16384)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 3) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 4096))))[0];
    ((float4*)(B_shared + (((((int)threadIdx.x) * 4) + 2048))))[0] = ((float4*)(B + ((((((((((int)blockIdx.x) >> 5) * 32768) + (k_outer_outer * 16384)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 3) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 8192))))[0];
    ((float4*)(B_shared + (((((int)threadIdx.x) * 4) + 3072))))[0] = ((float4*)(B + ((((((((((int)blockIdx.x) >> 5) * 32768) + (k_outer_outer * 16384)) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)blockIdx.x) & 3) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 12288))))[0];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 8; ++i_c_outer_inner) {
        T_batch_matmul_NN_local[((i_c_outer_inner * 2))] = (T_batch_matmul_NN_local[((i_c_outer_inner * 2))] + (A_shared[(((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)))] * B_shared[(((k_outer_inner * 512) + (((int)threadIdx.x) & 63)))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 16))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 16))] + (A_shared[(((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 64))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 1))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 1))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 32))] * B_shared[(((k_outer_inner * 512) + (((int)threadIdx.x) & 63)))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 17))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 17))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 32))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 64))]));
        T_batch_matmul_NN_local[((i_c_outer_inner * 2))] = (T_batch_matmul_NN_local[((i_c_outer_inner * 2))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 1))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 128))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 16))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 16))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 1))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 192))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 1))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 1))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 33))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 128))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 17))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 17))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 33))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 192))]));
        T_batch_matmul_NN_local[((i_c_outer_inner * 2))] = (T_batch_matmul_NN_local[((i_c_outer_inner * 2))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 2))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 256))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 16))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 16))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 2))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 320))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 1))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 1))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 34))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 256))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 17))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 17))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 34))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 320))]));
        T_batch_matmul_NN_local[((i_c_outer_inner * 2))] = (T_batch_matmul_NN_local[((i_c_outer_inner * 2))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 3))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 384))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 16))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 16))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 3))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 448))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 1))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 1))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 35))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 384))]));
        T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 17))] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 2) + 17))] + (A_shared[((((((((int)threadIdx.x) >> 6) * 512) + (i_c_outer_inner * 64)) + (k_outer_inner * 4)) + 35))] * B_shared[((((k_outer_inner * 512) + (((int)threadIdx.x) & 63)) + 448))]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    T_batch_matmul_NN[(((((((((int)blockIdx.x) >> 2) * 32768) + ((((int)threadIdx.x) >> 6) * 8192)) + (i_inner * 512)) + ((((int)blockIdx.x) & 3) * 128)) + (((int)threadIdx.x) & 63)))] = T_batch_matmul_NN_local[(i_inner)];
    T_batch_matmul_NN[((((((((((int)blockIdx.x) >> 2) * 32768) + ((((int)threadIdx.x) >> 6) * 8192)) + (i_inner * 512)) + ((((int)blockIdx.x) & 3) * 128)) + (((int)threadIdx.x) & 63)) + 64))] = T_batch_matmul_NN_local[((i_inner + 16))];
  }
}

