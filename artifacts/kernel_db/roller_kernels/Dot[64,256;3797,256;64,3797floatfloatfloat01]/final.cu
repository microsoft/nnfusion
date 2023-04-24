
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[8];
  __shared__ float A_shared[512];
  __shared__ float B_shared[256];
  float A_shared_local[4];
  float B_shared_local[2];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] = A[(((((((int)threadIdx.x) >> 3) * 256) + (k_outer * 8)) + (((int)threadIdx.x) & 7)))];
    A_shared[((((int)threadIdx.x) + 256))] = A[((((((((int)threadIdx.x) >> 3) * 256) + (k_outer * 8)) + (((int)threadIdx.x) & 7)) + 8192))];
    if (((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 3)) < 3797) {
      B_shared[(((int)threadIdx.x))] = B[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 256)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)))];
    }
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 8; ++k_inner_outer) {
      A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 4) * 8) + k_inner_outer))];
      A_shared_local[(1)] = A_shared[(((((((int)threadIdx.x) >> 4) * 8) + k_inner_outer) + 128))];
      A_shared_local[(2)] = A_shared[(((((((int)threadIdx.x) >> 4) * 8) + k_inner_outer) + 256))];
      A_shared_local[(3)] = A_shared[(((((((int)threadIdx.x) >> 4) * 8) + k_inner_outer) + 384))];
      B_shared_local[(0)] = B_shared[((((((int)threadIdx.x) & 15) * 8) + k_inner_outer))];
      if (((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 15)) < 3781) {
        B_shared_local[(1)] = B_shared[(((((((int)threadIdx.x) & 15) * 8) + k_inner_outer) + 128))];
      }
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(4)] = (compute_local[(4)] + (A_shared_local[(2)] * B_shared_local[(0)]));
      compute_local[(6)] = (compute_local[(6)] + (A_shared_local[(3)] * B_shared_local[(0)]));
      if (((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 15)) < 3781) {
        compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
        compute_local[(3)] = (compute_local[(3)] + (A_shared_local[(1)] * B_shared_local[(1)]));
        compute_local[(5)] = (compute_local[(5)] + (A_shared_local[(2)] * B_shared_local[(1)]));
        compute_local[(7)] = (compute_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
      }
    }
  }
  compute[(((((((int)threadIdx.x) >> 4) * 3797) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)))] = compute_local[(0)];
  compute[((((((((int)threadIdx.x) >> 4) * 3797) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 60752))] = compute_local[(2)];
  compute[((((((((int)threadIdx.x) >> 4) * 3797) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 121504))] = compute_local[(4)];
  compute[((((((((int)threadIdx.x) >> 4) * 3797) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 182256))] = compute_local[(6)];
  if (((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 15)) < 3781) {
    compute[((((((((int)threadIdx.x) >> 4) * 3797) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 16))] = compute_local[(1)];
    compute[((((((((int)threadIdx.x) >> 4) * 3797) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 60768))] = compute_local[(3)];
    compute[((((((((int)threadIdx.x) >> 4) * 3797) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 121520))] = compute_local[(5)];
    compute[((((((((int)threadIdx.x) >> 4) * 3797) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 182272))] = compute_local[(7)];
  }
}

dim3 grid(119, 1, 1);
dim3 block(256, 1, 1);
best_idx 11