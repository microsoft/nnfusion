
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
  float compute_local[1];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[32];
  compute_local[(0)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(A_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(A + (((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2048))))[0];
    ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 4096))))[0];
    ((float4*)(A_shared + (((((((((int)threadIdx.x) * 4) + 1536) >> 10) * 1024) + (((((int)threadIdx.x) >> 2) + 32) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) * 8192) + ((((((int)threadIdx.x) * 4) + 1536) >> 10) * 4096)) + (((((int)threadIdx.x) >> 2) + 32) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    if (((int)threadIdx.x) < 8) {
      B_shared[((((int)threadIdx.x) * 4))] = B[(((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)))];
    }
    if (((int)threadIdx.x) < 8) {
      B_shared[(((((int)threadIdx.x) * 4) + 1))] = B[(((((((int)blockIdx.x) * 128) + ((((((int)threadIdx.x) * 4) + 1) >> 4) * 64)) + (k_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 1) & 15)))];
    }
    if (((int)threadIdx.x) < 8) {
      B_shared[(((((int)threadIdx.x) * 4) + 2))] = B[(((((((int)blockIdx.x) * 128) + ((((((int)threadIdx.x) * 4) + 2) >> 4) * 64)) + (k_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 2) & 15)))];
    }
    if (((int)threadIdx.x) < 8) {
      B_shared[(((((int)threadIdx.x) * 4) + 3))] = B[(((((((int)blockIdx.x) * 128) + ((((((int)threadIdx.x) * 4) + 3) >> 4) * 64)) + (k_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 3) & 15)))];
    }
    __syncthreads();
    compute_local[(0)] = (compute_local[(0)] + (A_shared[((((int)threadIdx.x) * 16))] * B_shared[(((((int)threadIdx.x) >> 6) * 16))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 1))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 1))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 2))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 2))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 3))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 3))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 4))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 4))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 5))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 5))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 6))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 6))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 7))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 7))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 8))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 8))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 9))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 9))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 10))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 10))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 11))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 11))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 12))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 12))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 13))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 13))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 14))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 14))]));
    compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 15))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 15))]));
  }
  compute[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))] = compute_local[(0)];
}

dim3 grid(384, 1, 1);
dim3 block(128, 1, 1);
