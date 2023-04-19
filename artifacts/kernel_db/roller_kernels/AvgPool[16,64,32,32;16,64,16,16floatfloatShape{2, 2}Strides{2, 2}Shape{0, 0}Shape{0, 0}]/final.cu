
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
  __shared__ float padded_data_shared[2048];
  float padded_data_shared_local[2];
  avgpool2d_local[(0)] = 0.000000e+00f;
  avgpool2d_local[(1)] = 0.000000e+00f;
  padded_data_shared[(((int)threadIdx.x))] = data[(((((int)blockIdx.x) * 2048) + ((int)threadIdx.x)))];
  padded_data_shared[((((int)threadIdx.x) + 256))] = data[((((((int)blockIdx.x) * 2048) + ((int)threadIdx.x)) + 256))];
  padded_data_shared[((((int)threadIdx.x) + 512))] = data[((((((int)blockIdx.x) * 2048) + ((int)threadIdx.x)) + 512))];
  padded_data_shared[((((int)threadIdx.x) + 768))] = data[((((((int)blockIdx.x) * 2048) + ((int)threadIdx.x)) + 768))];
  padded_data_shared[((((int)threadIdx.x) + 1024))] = data[((((((int)blockIdx.x) * 2048) + ((int)threadIdx.x)) + 1024))];
  padded_data_shared[((((int)threadIdx.x) + 1280))] = data[((((((int)blockIdx.x) * 2048) + ((int)threadIdx.x)) + 1280))];
  padded_data_shared[((((int)threadIdx.x) + 1536))] = data[((((((int)blockIdx.x) * 2048) + ((int)threadIdx.x)) + 1536))];
  padded_data_shared[((((int)threadIdx.x) + 1792))] = data[((((((int)blockIdx.x) * 2048) + ((int)threadIdx.x)) + 1792))];
  __syncthreads();
  for (int kh_inner_outer = 0; kh_inner_outer < 2; ++kh_inner_outer) {
    for (int kw_inner_outer = 0; kw_inner_outer < 2; ++kw_inner_outer) {
      padded_data_shared_local[(0)] = padded_data_shared[((((((((int)threadIdx.x) >> 3) * 64) + (kh_inner_outer * 32)) + ((((int)threadIdx.x) & 7) * 2)) + kw_inner_outer))];
      padded_data_shared_local[(1)] = padded_data_shared[(((((((((int)threadIdx.x) >> 3) * 64) + (kh_inner_outer * 32)) + ((((int)threadIdx.x) & 7) * 2)) + kw_inner_outer) + 16))];
      avgpool2d_local[(0)] = (avgpool2d_local[(0)] + (padded_data_shared_local[(0)] * 2.500000e-01f));
      avgpool2d_local[(1)] = (avgpool2d_local[(1)] + (padded_data_shared_local[(1)] * 2.500000e-01f));
    }
  }
  avgpool2d[((((((int)blockIdx.x) * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)))] = avgpool2d_local[(0)];
  avgpool2d[(((((((int)blockIdx.x) * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8))] = avgpool2d_local[(1)];
}

dim3 grid(512, 1, 1);
dim3 block(256, 1, 1);
best_idx 3