
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
  float compute_local[4];
  __shared__ float A_shared[4096];
  __shared__ float B_shared[4096];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(A_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(A + (((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 31) * 4)))))[0];
    ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(A + ((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2048))))[0];
    ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 2048))))[0] = ((float4*)(A + ((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 4096))))[0];
    ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 3072))))[0] = ((float4*)(A + ((((((((((int)blockIdx.x) & 31) >> 4) * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 6144))))[0];
    ((float4*)(B_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(B + (((((((((int)blockIdx.x) >> 5) * 131072) + (k_outer_outer * 32768)) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    ((float4*)(B_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(B + ((((((((((int)blockIdx.x) >> 5) * 131072) + (k_outer_outer * 32768)) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384))))[0];
    ((float4*)(B_shared + (((((int)threadIdx.x) * 4) + 2048))))[0] = ((float4*)(B + ((((((((((int)blockIdx.x) >> 5) * 131072) + (k_outer_outer * 32768)) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 65536))))[0];
    ((float4*)(B_shared + (((((((((int)threadIdx.x) * 4) + 3072) >> 11) * 2048) + (((((int)threadIdx.x) >> 2) + 64) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0] = ((float4*)(B + ((((((((((int)blockIdx.x) >> 5) * 131072) + ((((((int)threadIdx.x) * 4) + 3072) >> 11) * 65536)) + (k_outer_outer * 32768)) + (((((int)threadIdx.x) >> 2) + 64) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 64; ++k_inner) {
        compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (k_outer_inner * 64)) + k_inner))] * B_shared[((((k_outer_inner * 1024) + (k_inner * 16)) + (((int)threadIdx.x) & 15)))]));
        compute_local[(2)] = (compute_local[(2)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (k_outer_inner * 64)) + k_inner))] * B_shared[(((((k_outer_inner * 1024) + (k_inner * 16)) + (((int)threadIdx.x) & 15)) + 2048))]));
        compute_local[(1)] = (compute_local[(1)] + (A_shared[((((((((int)threadIdx.x) >> 4) * 256) + (k_outer_inner * 64)) + k_inner) + 128))] * B_shared[((((k_outer_inner * 1024) + (k_inner * 16)) + (((int)threadIdx.x) & 15)))]));
        compute_local[(3)] = (compute_local[(3)] + (A_shared[((((((((int)threadIdx.x) >> 4) * 256) + (k_outer_inner * 64)) + k_inner) + 128))] * B_shared[(((((k_outer_inner * 1024) + (k_inner * 16)) + (((int)threadIdx.x) & 15)) + 2048))]));
      }
    }
  }
  for (int x_inner = 0; x_inner < 2; ++x_inner) {
    compute[((((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (x_inner * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)))] = compute_local[(x_inner)];
    compute[(((((((((((int)blockIdx.x) >> 5) * 32768) + (((((int)blockIdx.x) & 31) >> 4) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (x_inner * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 16384))] = compute_local[((x_inner + 2))];
  }
}

dim3 grid(64, 1, 1);
dim3 block(256, 1, 1);
