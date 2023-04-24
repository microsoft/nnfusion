
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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ data, float* __restrict__ avgpool2d) {
  float avgpool2d_local[2];
  __shared__ float padded_data_shared[1024];
  float padded_data_shared_local[2];
  avgpool2d_local[0] = 0.000000e+00f;
  avgpool2d_local[1] = 0.000000e+00f;
  padded_data_shared[((int)threadIdx.x)] = data[((((((int)threadIdx.x) >> 6) * 1024) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63))];
  padded_data_shared[(((int)threadIdx.x) + 128)] = data[(((((((int)threadIdx.x) >> 6) * 1024) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63)) + 2048)];
  padded_data_shared[(((int)threadIdx.x) + 256)] = data[(((((((int)threadIdx.x) >> 6) * 1024) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63)) + 4096)];
  padded_data_shared[(((int)threadIdx.x) + 384)] = data[(((((((int)threadIdx.x) >> 6) * 1024) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63)) + 6144)];
  padded_data_shared[(((int)threadIdx.x) + 512)] = data[(((((((int)threadIdx.x) >> 6) * 1024) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63)) + 8192)];
  padded_data_shared[(((int)threadIdx.x) + 640)] = data[(((((((int)threadIdx.x) >> 6) * 1024) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63)) + 10240)];
  padded_data_shared[(((int)threadIdx.x) + 768)] = data[(((((((int)threadIdx.x) >> 6) * 1024) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63)) + 12288)];
  padded_data_shared[(((int)threadIdx.x) + 896)] = data[(((((((int)threadIdx.x) >> 6) * 1024) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63)) + 14336)];
  __syncthreads();
  for (int kh_inner_outer = 0; kh_inner_outer < 2; ++kh_inner_outer) {
    for (int kw_inner_outer = 0; kw_inner_outer < 2; ++kw_inner_outer) {
      padded_data_shared_local[0] = padded_data_shared[(((((((int)threadIdx.x) >> 3) * 64) + (kh_inner_outer * 32)) + ((((int)threadIdx.x) & 7) * 2)) + kw_inner_outer)];
      padded_data_shared_local[1] = padded_data_shared[((((((((int)threadIdx.x) >> 3) * 64) + (kh_inner_outer * 32)) + ((((int)threadIdx.x) & 7) * 2)) + kw_inner_outer) + 16)];
      avgpool2d_local[0] = (avgpool2d_local[0] + (padded_data_shared_local[0] * 2.500000e-01f));
      avgpool2d_local[1] = (avgpool2d_local[1] + (padded_data_shared_local[1] * 2.500000e-01f));
    }
  }
  avgpool2d[((((((int)threadIdx.x) >> 3) * 256) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 7))] = avgpool2d_local[0];
  avgpool2d[(((((((int)threadIdx.x) >> 3) * 256) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = avgpool2d_local[1];
}

dim3 grid(16, 1, 1);
dim3 block(128, 1, 1);
best_idx 15