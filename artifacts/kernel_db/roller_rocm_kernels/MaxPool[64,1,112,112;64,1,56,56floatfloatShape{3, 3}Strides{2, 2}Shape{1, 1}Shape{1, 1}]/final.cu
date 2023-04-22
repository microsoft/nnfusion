
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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ data, float* __restrict__ maxpool2d) {
  float maxpool2d_local[2];
  __shared__ float padded_data_shared[1026];
  float padded_data_shared_local[2];
  maxpool2d_local[0] = -3.402823e+38f;
  maxpool2d_local[1] = -3.402823e+38f;
  padded_data_shared[((int)threadIdx.x)] = ((((1 <= (((((int)blockIdx.x) % 14) * 8) + (((int)threadIdx.x) / 114))) && (1 <= (((int)threadIdx.x) % 114))) && ((((int)threadIdx.x) % 114) < 113)) ? data[((((((int)blockIdx.x) * 896) + ((((int)threadIdx.x) / 114) * 112)) + (((int)threadIdx.x) % 114)) - 113)] : -3.402823e+37f);
  padded_data_shared[(((int)threadIdx.x) + 128)] = (((1 <= ((((int)threadIdx.x) + 14) % 114)) && (((((int)threadIdx.x) + 14) % 114) < 113)) ? data[((((((int)blockIdx.x) * 896) + (((((int)threadIdx.x) + 128) / 114) * 112)) + ((((int)threadIdx.x) + 14) % 114)) - 113)] : -3.402823e+37f);
  padded_data_shared[(((int)threadIdx.x) + 256)] = (((1 <= ((((int)threadIdx.x) + 28) % 114)) && (((((int)threadIdx.x) + 28) % 114) < 113)) ? data[((((((int)blockIdx.x) * 896) + (((((int)threadIdx.x) + 256) / 114) * 112)) + ((((int)threadIdx.x) + 28) % 114)) - 113)] : -3.402823e+37f);
  padded_data_shared[(((int)threadIdx.x) + 384)] = (((1 <= ((((int)threadIdx.x) + 42) % 114)) && (((((int)threadIdx.x) + 42) % 114) < 113)) ? data[((((((int)blockIdx.x) * 896) + (((((int)threadIdx.x) + 384) / 114) * 112)) + ((((int)threadIdx.x) + 42) % 114)) - 113)] : -3.402823e+37f);
  padded_data_shared[(((int)threadIdx.x) + 512)] = (((1 <= ((((int)threadIdx.x) + 56) % 114)) && (((((int)threadIdx.x) + 56) % 114) < 113)) ? data[((((((int)blockIdx.x) * 896) + (((((int)threadIdx.x) + 512) / 114) * 112)) + ((((int)threadIdx.x) + 56) % 114)) - 113)] : -3.402823e+37f);
  padded_data_shared[(((int)threadIdx.x) + 640)] = (((1 <= ((((int)threadIdx.x) + 70) % 114)) && (((((int)threadIdx.x) + 70) % 114) < 113)) ? data[((((((int)blockIdx.x) * 896) + (((((int)threadIdx.x) + 640) / 114) * 112)) + ((((int)threadIdx.x) + 70) % 114)) - 113)] : -3.402823e+37f);
  padded_data_shared[(((int)threadIdx.x) + 768)] = (((1 <= ((((int)threadIdx.x) + 84) % 114)) && (((((int)threadIdx.x) + 84) % 114) < 113)) ? data[((((((int)blockIdx.x) * 896) + (((((int)threadIdx.x) + 768) / 114) * 112)) + ((((int)threadIdx.x) + 84) % 114)) - 113)] : -3.402823e+37f);
  padded_data_shared[(((int)threadIdx.x) + 896)] = (((1 <= ((((int)threadIdx.x) + 98) % 114)) && (((((int)threadIdx.x) + 98) % 114) < 113)) ? data[((((((int)blockIdx.x) * 896) + (((((int)threadIdx.x) + 896) / 114) * 112)) + ((((int)threadIdx.x) + 98) % 114)) - 113)] : -3.402823e+37f);
  if (((int)threadIdx.x) < 2) {
    padded_data_shared[(((int)threadIdx.x) + 1024)] = ((((int)threadIdx.x) < 1) ? data[(((((int)blockIdx.x) * 896) + ((int)threadIdx.x)) + 895)] : -3.402823e+37f);
  }
  __syncthreads();
  for (int kh_inner_outer = 0; kh_inner_outer < 3; ++kh_inner_outer) {
    for (int kw_inner_outer = 0; kw_inner_outer < 3; ++kw_inner_outer) {
      padded_data_shared_local[0] = padded_data_shared[(((((((int)threadIdx.x) >> 5) * 228) + (kh_inner_outer * 114)) + ((((int)threadIdx.x) & 31) * 2)) + kw_inner_outer)];
      if (((kw_inner_outer >> 1) + (((int)threadIdx.x) & 31)) < 25) {
        padded_data_shared_local[1] = padded_data_shared[((((((((int)threadIdx.x) >> 5) * 228) + (kh_inner_outer * 114)) + ((((int)threadIdx.x) & 31) * 2)) + kw_inner_outer) + 64)];
      }
      maxpool2d_local[0] = max(maxpool2d_local[0], padded_data_shared_local[0]);
      if ((((int)threadIdx.x) & 31) < 24) {
        maxpool2d_local[1] = max(maxpool2d_local[1], padded_data_shared_local[1]);
      }
    }
  }
  maxpool2d[(((((int)blockIdx.x) * 224) + ((((int)threadIdx.x) >> 5) * 56)) + (((int)threadIdx.x) & 31))] = maxpool2d_local[0];
  if ((((int)threadIdx.x) & 31) < 24) {
    maxpool2d[((((((int)blockIdx.x) * 224) + ((((int)threadIdx.x) >> 5) * 56)) + (((int)threadIdx.x) & 31)) + 32)] = maxpool2d_local[1];
  }
}

dim3 grid(896, 1, 1);
dim3 block(128, 1, 1);
best_idx 1