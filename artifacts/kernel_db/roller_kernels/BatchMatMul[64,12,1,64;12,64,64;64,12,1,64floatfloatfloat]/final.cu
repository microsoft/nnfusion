
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
  float compute_local[2];
  __shared__ float A_shared[1024];
  __shared__ float B_shared[1024];
  float A_shared_local[1];
  float B_shared_local[2];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  A_shared[(((int)threadIdx.x))] = A[((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)))];
  A_shared[((((int)threadIdx.x) + 128))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 1536))];
  A_shared[((((int)threadIdx.x) + 256))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 3072))];
  A_shared[((((int)threadIdx.x) + 384))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 4608))];
  A_shared[((((int)threadIdx.x) + 512))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 6144))];
  A_shared[((((int)threadIdx.x) + 640))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 7680))];
  A_shared[((((int)threadIdx.x) + 768))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 9216))];
  A_shared[((((int)threadIdx.x) + 896))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 10752))];
  B_shared[(((int)threadIdx.x))] = B[(((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)))];
  B_shared[((((int)threadIdx.x) + 128))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 512))];
  B_shared[((((int)threadIdx.x) + 256))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 1024))];
  B_shared[((((int)threadIdx.x) + 384))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 1536))];
  B_shared[((((int)threadIdx.x) + 512))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 2048))];
  B_shared[((((int)threadIdx.x) + 640))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 2560))];
  B_shared[((((int)threadIdx.x) + 768))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 3072))];
  B_shared[((((int)threadIdx.x) + 896))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 3584))];
  __syncthreads();
  for (int k_inner_outer = 0; k_inner_outer < 64; ++k_inner_outer) {
    A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 3) * 64) + k_inner_outer))];
    B_shared_local[(0)] = B_shared[(((k_inner_outer * 16) + (((int)threadIdx.x) & 7)))];
    B_shared_local[(1)] = B_shared[((((k_inner_outer * 16) + (((int)threadIdx.x) & 7)) + 8))];
    compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
  }
  compute[((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)blockIdx.x) % 48) * 16)) + (((int)threadIdx.x) & 7)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)blockIdx.x) % 48) * 16)) + (((int)threadIdx.x) & 7)) + 8))] = compute_local[(1)];
}

dim3 grid(192, 1, 1);
dim3 block(128, 1, 1);
best_idx 7