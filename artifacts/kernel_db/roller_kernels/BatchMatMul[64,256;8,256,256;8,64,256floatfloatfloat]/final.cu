
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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[8];
  __shared__ float A_shared[1024];
  __shared__ float B_shared[1024];
  float A_shared_local[4];
  float B_shared_local[2];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] = A[(((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)))];
    A_shared[((((int)threadIdx.x) + 128))] = A[((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 1024))];
    A_shared[((((int)threadIdx.x) + 256))] = A[((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048))];
    A_shared[((((int)threadIdx.x) + 384))] = A[((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 3072))];
    A_shared[((((int)threadIdx.x) + 512))] = A[((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096))];
    A_shared[((((int)threadIdx.x) + 640))] = A[((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 5120))];
    A_shared[((((int)threadIdx.x) + 768))] = A[((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144))];
    A_shared[((((int)threadIdx.x) + 896))] = A[((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168))];
    B_shared[(((int)threadIdx.x))] = B[(((((((((int)blockIdx.x) >> 5) * 131072) + (k_outer * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)))];
    B_shared[((((int)threadIdx.x) + 128))] = B[((((((((((int)blockIdx.x) >> 5) * 131072) + (k_outer * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 2048))];
    B_shared[((((int)threadIdx.x) + 256))] = B[((((((((((int)blockIdx.x) >> 5) * 131072) + (k_outer * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 4096))];
    B_shared[((((int)threadIdx.x) + 384))] = B[((((((((((int)blockIdx.x) >> 5) * 131072) + (k_outer * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 6144))];
    B_shared[((((int)threadIdx.x) + 512))] = B[((((((((((int)blockIdx.x) >> 5) * 131072) + (k_outer * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 65536))];
    B_shared[((((int)threadIdx.x) + 640))] = B[((((((((((int)blockIdx.x) >> 5) * 131072) + (((((int)threadIdx.x) + 640) >> 9) * 65536)) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 4) + 8) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)))];
    B_shared[((((int)threadIdx.x) + 768))] = B[((((((((((int)blockIdx.x) >> 5) * 131072) + (((((int)threadIdx.x) + 768) >> 9) * 65536)) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 4) + 16) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)))];
    B_shared[((((int)threadIdx.x) + 896))] = B[((((((((((int)blockIdx.x) >> 5) * 131072) + (((((int)threadIdx.x) + 896) >> 9) * 65536)) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 4) + 24) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)))];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
      A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer))];
      A_shared_local[(1)] = A_shared[(((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer) + 256))];
      A_shared_local[(2)] = A_shared[(((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer) + 512))];
      A_shared_local[(3)] = A_shared[(((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer) + 768))];
      B_shared_local[(0)] = B_shared[(((k_inner_outer * 16) + (((int)threadIdx.x) & 15)))];
      B_shared_local[(1)] = B_shared[((((k_inner_outer * 16) + (((int)threadIdx.x) & 15)) + 512))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(4)] = (compute_local[(4)] + (A_shared_local[(0)] * B_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(5)] = (compute_local[(5)] + (A_shared_local[(1)] * B_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (A_shared_local[(2)] * B_shared_local[(0)]));
      compute_local[(6)] = (compute_local[(6)] + (A_shared_local[(2)] * B_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (A_shared_local[(3)] * B_shared_local[(0)]));
      compute_local[(7)] = (compute_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
    }
  }
  compute[(((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 16384))] = compute_local[(4)];
  compute[((((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 2048))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 18432))] = compute_local[(5)];
  compute[((((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 4096))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 20480))] = compute_local[(6)];
  compute[((((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 6144))] = compute_local[(3)];
  compute[((((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 22528))] = compute_local[(7)];
}

dim3 grid(128, 1, 1);
dim3 block(128, 1, 1);
best_idx 15