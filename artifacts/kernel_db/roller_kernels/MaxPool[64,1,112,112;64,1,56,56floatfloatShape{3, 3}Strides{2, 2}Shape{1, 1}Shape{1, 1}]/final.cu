
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
extern "C" __global__ void __launch_bounds__(224) default_function_kernel0(float* __restrict__ data, float* __restrict__ maxpool2d) {
  float maxpool2d_local[2];
  __shared__ float padded_data_shared[1921];
  float padded_data_shared_local[2];
  maxpool2d_local[(0)] = -3.402823e+38f;
  maxpool2d_local[(1)] = -3.402823e+38f;
  padded_data_shared[(((int)threadIdx.x))] = (((1 <= (((((int)blockIdx.x) % 7) * 16) + (((int)threadIdx.x) / 113))) && (1 <= (((int)threadIdx.x) % 113))) ? data[(((((((int)blockIdx.x) * 1792) + ((((int)threadIdx.x) / 113) * 112)) + (((int)threadIdx.x) % 113)) - 113))] : -3.402823e+37f);
  padded_data_shared[((((int)threadIdx.x) + 224))] = ((1 <= ((((int)threadIdx.x) + 111) % 113)) ? data[(((((((int)blockIdx.x) * 1792) + (((((int)threadIdx.x) + 224) / 113) * 112)) + ((((int)threadIdx.x) + 111) % 113)) - 113))] : -3.402823e+37f);
  padded_data_shared[((((int)threadIdx.x) + 448))] = ((1 <= ((((int)threadIdx.x) + 109) % 113)) ? data[(((((((int)blockIdx.x) * 1792) + (((((int)threadIdx.x) + 448) / 113) * 112)) + ((((int)threadIdx.x) + 109) % 113)) - 113))] : -3.402823e+37f);
  padded_data_shared[((((int)threadIdx.x) + 672))] = ((1 <= ((((int)threadIdx.x) + 107) % 113)) ? data[(((((((int)blockIdx.x) * 1792) + (((((int)threadIdx.x) + 672) / 113) * 112)) + ((((int)threadIdx.x) + 107) % 113)) - 113))] : -3.402823e+37f);
  padded_data_shared[((((int)threadIdx.x) + 896))] = ((1 <= ((((int)threadIdx.x) + 105) % 113)) ? data[(((((((int)blockIdx.x) * 1792) + (((((int)threadIdx.x) + 896) / 113) * 112)) + ((((int)threadIdx.x) + 105) % 113)) - 113))] : -3.402823e+37f);
  padded_data_shared[((((int)threadIdx.x) + 1120))] = ((1 <= ((((int)threadIdx.x) + 103) % 113)) ? data[(((((((int)blockIdx.x) * 1792) + (((((int)threadIdx.x) + 1120) / 113) * 112)) + ((((int)threadIdx.x) + 103) % 113)) - 113))] : -3.402823e+37f);
  padded_data_shared[((((int)threadIdx.x) + 1344))] = ((1 <= ((((int)threadIdx.x) + 101) % 113)) ? data[(((((((int)blockIdx.x) * 1792) + (((((int)threadIdx.x) + 1344) / 113) * 112)) + ((((int)threadIdx.x) + 101) % 113)) - 113))] : -3.402823e+37f);
  padded_data_shared[((((int)threadIdx.x) + 1568))] = ((1 <= ((((int)threadIdx.x) + 99) % 113)) ? data[(((((((int)blockIdx.x) * 1792) + (((((int)threadIdx.x) + 1568) / 113) * 112)) + ((((int)threadIdx.x) + 99) % 113)) - 113))] : -3.402823e+37f);
  if (((int)threadIdx.x) < 129) {
    padded_data_shared[((((int)threadIdx.x) + 1792))] = ((1 <= ((((int)threadIdx.x) + 97) % 113)) ? data[(((((((int)blockIdx.x) * 1792) + (((((int)threadIdx.x) + 1792) / 113) * 112)) + ((((int)threadIdx.x) + 97) % 113)) - 113))] : -3.402823e+37f);
  }
  __syncthreads();
  for (int kh_inner_outer = 0; kh_inner_outer < 3; ++kh_inner_outer) {
    for (int kw_inner_outer = 0; kw_inner_outer < 3; ++kw_inner_outer) {
      padded_data_shared_local[(0)] = padded_data_shared[((((((((int)threadIdx.x) / 28) * 226) + (kh_inner_outer * 113)) + ((((int)threadIdx.x) % 28) * 2)) + kw_inner_outer))];
      padded_data_shared_local[(1)] = padded_data_shared[(((((((((int)threadIdx.x) / 28) * 226) + (kh_inner_outer * 113)) + ((((int)threadIdx.x) % 28) * 2)) + kw_inner_outer) + 56))];
      maxpool2d_local[(0)] = max(maxpool2d_local[(0)], padded_data_shared_local[(0)]);
      maxpool2d_local[(1)] = max(maxpool2d_local[(1)], padded_data_shared_local[(1)]);
    }
  }
  maxpool2d[((((((int)blockIdx.x) * 448) + ((((int)threadIdx.x) / 28) * 56)) + (((int)threadIdx.x) % 28)))] = maxpool2d_local[(0)];
  maxpool2d[(((((((int)blockIdx.x) * 448) + ((((int)threadIdx.x) / 28) * 56)) + (((int)threadIdx.x) % 28)) + 28))] = maxpool2d_local[(1)];
}

dim3 grid(448, 1, 1);
dim3 block(224, 1, 1);
best_idx 0