
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
extern "C" __global__ void __launch_bounds__(32) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float A_shared[512];
  __shared__ float B_shared[512];
  float A_shared_local[1];
  float B_shared_local[2];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  A_shared[(((int)threadIdx.x))] = A[((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)))];
  A_shared[((((int)threadIdx.x) + 32))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 32))];
  A_shared[((((int)threadIdx.x) + 64))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 64))];
  A_shared[((((int)threadIdx.x) + 96))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 96))];
  A_shared[((((int)threadIdx.x) + 128))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 128))];
  A_shared[((((int)threadIdx.x) + 160))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 160))];
  A_shared[((((int)threadIdx.x) + 192))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 192))];
  A_shared[((((int)threadIdx.x) + 224))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 224))];
  A_shared[((((int)threadIdx.x) + 256))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 256))];
  A_shared[((((int)threadIdx.x) + 288))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 288))];
  A_shared[((((int)threadIdx.x) + 320))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 320))];
  A_shared[((((int)threadIdx.x) + 352))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 352))];
  A_shared[((((int)threadIdx.x) + 384))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 384))];
  A_shared[((((int)threadIdx.x) + 416))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 416))];
  A_shared[((((int)threadIdx.x) + 448))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 448))];
  A_shared[((((int)threadIdx.x) + 480))] = A[(((((((int)blockIdx.x) >> 1) * 512) + ((int)threadIdx.x)) + 480))];
  B_shared[(((int)threadIdx.x))] = B[((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)))];
  B_shared[((((int)threadIdx.x) + 32))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 32))];
  B_shared[((((int)threadIdx.x) + 64))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 64))];
  B_shared[((((int)threadIdx.x) + 96))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 96))];
  if ((((int)blockIdx.x) & 1) < 1) {
    B_shared[((((int)threadIdx.x) + 128))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 128))];
  }
  if ((((((int)blockIdx.x) & 1) * 8) + ((((int)threadIdx.x) + 160) >> 6)) < 10) {
    B_shared[((((int)threadIdx.x) + 160))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 160))];
  }
  if ((((int)blockIdx.x) & 1) < 1) {
    B_shared[((((int)threadIdx.x) + 192))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 192))];
  }
  if ((((((int)blockIdx.x) & 1) * 8) + ((((int)threadIdx.x) + 224) >> 6)) < 10) {
    B_shared[((((int)threadIdx.x) + 224))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 224))];
  }
  if ((((int)blockIdx.x) & 1) < 1) {
    B_shared[((((int)threadIdx.x) + 256))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 256))];
  }
  if ((((((int)blockIdx.x) & 1) * 8) + ((((int)threadIdx.x) + 288) >> 6)) < 10) {
    B_shared[((((int)threadIdx.x) + 288))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 288))];
  }
  if ((((int)blockIdx.x) & 1) < 1) {
    B_shared[((((int)threadIdx.x) + 320))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 320))];
  }
  if ((((((int)blockIdx.x) & 1) * 8) + ((((int)threadIdx.x) + 352) >> 6)) < 10) {
    B_shared[((((int)threadIdx.x) + 352))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 352))];
  }
  if ((((int)blockIdx.x) & 1) < 1) {
    B_shared[((((int)threadIdx.x) + 384))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 384))];
  }
  if ((((((int)blockIdx.x) & 1) * 8) + ((((int)threadIdx.x) + 416) >> 6)) < 10) {
    B_shared[((((int)threadIdx.x) + 416))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 416))];
  }
  if ((((int)blockIdx.x) & 1) < 1) {
    B_shared[((((int)threadIdx.x) + 448))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 448))];
  }
  if ((((((int)blockIdx.x) & 1) * 8) + ((((int)threadIdx.x) + 480) >> 6)) < 10) {
    B_shared[((((int)threadIdx.x) + 480))] = B[(((((((int)blockIdx.x) & 1) * 512) + ((int)threadIdx.x)) + 480))];
  }
  __syncthreads();
  for (int k_inner_outer = 0; k_inner_outer < 64; ++k_inner_outer) {
    A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 2) * 64) + k_inner_outer))];
    if ((((((int)blockIdx.x) & 1) * 8) + (((int)threadIdx.x) & 3)) < 10) {
      B_shared_local[(0)] = B_shared[((((((int)threadIdx.x) & 3) * 64) + k_inner_outer))];
    }
    if ((((((int)blockIdx.x) & 1) * 8) + (((int)threadIdx.x) & 3)) < 6) {
      B_shared_local[(1)] = B_shared[(((((((int)threadIdx.x) & 3) * 64) + k_inner_outer) + 256))];
    }
    if ((((((int)blockIdx.x) & 1) * 8) + (((int)threadIdx.x) & 3)) < 10) {
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
    }
    if ((((((int)blockIdx.x) & 1) * 8) + (((int)threadIdx.x) & 3)) < 6) {
      compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    }
  }
  if ((((((int)blockIdx.x) & 1) * 8) + (((int)threadIdx.x) & 3)) < 10) {
    compute[((((((((int)blockIdx.x) >> 1) * 80) + ((((int)threadIdx.x) >> 2) * 10)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 3)))] = compute_local[(0)];
  }
  if ((((((int)blockIdx.x) & 1) * 8) + (((int)threadIdx.x) & 3)) < 6) {
    compute[(((((((((int)blockIdx.x) >> 1) * 80) + ((((int)threadIdx.x) >> 2) * 10)) + ((((int)blockIdx.x) & 1) * 8)) + (((int)threadIdx.x) & 3)) + 4))] = compute_local[(1)];
  }
}

dim3 grid(16, 1, 1);
dim3 block(32, 1, 1);
best_idx 13