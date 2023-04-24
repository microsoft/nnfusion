
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ data, float* __restrict__ avgpool2d) {
  float avgpool2d_local[2];
  __shared__ float padded_data_shared[512];
  float padded_data_shared_local[2];
  avgpool2d_local[0] = 0.000000e+00f;
  avgpool2d_local[1] = 0.000000e+00f;
  padded_data_shared[((int)threadIdx.x)] = data[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))];
  padded_data_shared[(((int)threadIdx.x) + 256)] = data[(((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) + 256)];
  __syncthreads();
  for (int kh_inner_outer = 0; kh_inner_outer < 2; ++kh_inner_outer) {
    for (int kw_inner_outer = 0; kw_inner_outer < 2; ++kw_inner_outer) {
      if ((((int)threadIdx.x) & 127) < 64) {
        padded_data_shared_local[0] = padded_data_shared[((((((((int)threadIdx.x) >> 7) * 256) + (((((int)threadIdx.x) & 127) >> 3) * 32)) + (kh_inner_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + kw_inner_outer)];
      }
      if ((((int)threadIdx.x) & 127) < 64) {
        avgpool2d_local[0] = (avgpool2d_local[0] + (padded_data_shared_local[0] * 2.500000e-01f));
      }
    }
  }
  if ((((int)threadIdx.x) & 127) < 64) {
    avgpool2d[(((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 7) * 64)) + (((int)threadIdx.x) & 127))] = avgpool2d_local[0];
  }
}

dim3 grid(16, 1, 1);
dim3 block(256, 1, 1);
best_idx 2