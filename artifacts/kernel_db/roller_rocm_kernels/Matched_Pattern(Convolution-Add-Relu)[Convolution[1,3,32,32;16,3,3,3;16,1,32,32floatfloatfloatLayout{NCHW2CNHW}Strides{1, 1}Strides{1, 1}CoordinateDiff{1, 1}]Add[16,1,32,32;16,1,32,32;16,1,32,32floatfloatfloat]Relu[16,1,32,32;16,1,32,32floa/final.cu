
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
extern "C" __global__ void __launch_bounds__(64) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ bias, float* __restrict__ conv_unpad) {
  float conv_local[2];
  __shared__ float data_pad_shared[864];
  __shared__ float kernel_pad_shared[108];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[1];
  conv_local[0] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  data_pad_shared[((int)threadIdx.x)] = (((0 < (((int)blockIdx.x) & 31)) && (0 < ((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)))) ? data[(((((((int)blockIdx.x) & 31) * 32) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) - 33)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 64)] = ((((0 < (((((int)threadIdx.x) + 64) / 96) + (((int)blockIdx.x) & 31))) && (0 < ((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && (((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 33)) ? data[(((((((((int)threadIdx.x) + 64) / 96) * 32) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) & 31)) + (((((int)threadIdx.x) >> 5) + 2) % 3)) - 33)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 128)] = ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)) < 32) ? data[(((((((((int)threadIdx.x) + 128) / 96) * 32) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) - 32)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 192)] = ((((((int)blockIdx.x) & 31) < 31) && (0 < ((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)))) ? data[(((((((int)blockIdx.x) & 31) * 32) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) + 31)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 256)] = (((((0 < (((((((int)threadIdx.x) >> 5) + 8) % 9) / 3) + (((int)blockIdx.x) & 31))) && ((((((((int)threadIdx.x) >> 5) + 8) % 9) / 3) + (((int)blockIdx.x) & 31)) < 33)) && (0 < ((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && (((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 33)) ? data[((((((((((int)threadIdx.x) + 256) / 288) * 1024) + (((((((int)threadIdx.x) >> 5) + 8) % 9) / 3) * 32)) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) & 31)) + (((((int)threadIdx.x) >> 5) + 2) % 3)) - 33)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 320)] = (((0 < (((int)blockIdx.x) & 31)) && (((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)) < 32)) ? data[(((((((((int)threadIdx.x) + 320) / 288) * 1024) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) - 32)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 384)] = ((0 < ((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31))) ? data[(((((((((int)threadIdx.x) + 384) / 288) * 1024) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) - 1)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 448)] = ((((((((((int)threadIdx.x) >> 5) + 5) / 3) + (((int)blockIdx.x) & 31)) < 33) && (0 < ((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && (((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 33)) ? data[((((((((((int)threadIdx.x) + 448) / 288) * 1024) + ((((((int)threadIdx.x) >> 5) + 5) / 3) * 32)) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) & 31)) + (((((int)threadIdx.x) >> 5) + 2) % 3)) - 33)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 512)] = (((((((((int)threadIdx.x) >> 5) + 7) / 3) + (((int)blockIdx.x) & 31)) < 33) && (((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)) < 32)) ? data[((((((((((int)threadIdx.x) + 512) / 288) * 1024) + ((((((int)threadIdx.x) >> 5) + 7) / 3) * 32)) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) - 32)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 576)] = (((0 < (((int)blockIdx.x) & 31)) && (0 < ((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)))) ? data[(((((((int)blockIdx.x) & 31) * 32) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) + 2015)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 640)] = ((((0 < ((((((int)threadIdx.x) >> 5) + 2) / 3) + (((int)blockIdx.x) & 31))) && (0 < ((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && (((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 33)) ? data[((((((((((int)threadIdx.x) + 640) / 288) * 1024) + ((((((int)threadIdx.x) >> 5) + 2) / 3) * 32)) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) & 31)) + (((((int)threadIdx.x) >> 5) + 2) % 3)) - 33)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 704)] = ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)) < 32) ? data[((((((((((int)threadIdx.x) + 704) / 288) * 1024) + ((((((int)threadIdx.x) >> 5) + 4) / 3) * 32)) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) - 32)] : 0.000000e+00f);
  data_pad_shared[(((int)threadIdx.x) + 768)] = ((((((int)blockIdx.x) & 31) < 31) && (0 < ((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)))) ? data[(((((((((int)threadIdx.x) + 768) / 288) * 1024) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) + 31)] : 0.000000e+00f);
  if (((int)threadIdx.x) < 32) {
    data_pad_shared[(((int)threadIdx.x) + 832)] = ((((((int)blockIdx.x) & 31) < 31) && (((int)threadIdx.x) < 31)) ? data[((((((int)blockIdx.x) & 31) * 32) + ((int)threadIdx.x)) + 2081)] : 0.000000e+00f);
  }
  kernel_pad_shared[((int)threadIdx.x)] = kernel[(((((int)blockIdx.x) >> 5) * 108) + ((int)threadIdx.x))];
  if (((int)threadIdx.x) < 44) {
    kernel_pad_shared[(((int)threadIdx.x) + 64)] = kernel[(((((((int)blockIdx.x) >> 5) * 108) + (((((int)threadIdx.x) + 64) / 27) * 27)) + ((((((int)threadIdx.x) + 10) % 27) / 9) * 9)) + ((((int)threadIdx.x) + 1) % 9))];
  }
  __syncthreads();
  for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 27; ++ra_fused0_inner_outer) {
    data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15))];
    data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 16)];
    kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 4) * 27) + ra_fused0_inner_outer)];
    conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
    conv_local[1] = (conv_local[1] + (data_pad_shared_local[1] * kernel_pad_shared_local[0]));
  }
  conv_unpad[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) & 15))] = max((conv_local[0] + bias[(((((int)blockIdx.x) >> 5) * 4) + (((int)threadIdx.x) >> 4))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = max((conv_local[1] + bias[(((((int)blockIdx.x) >> 5) * 4) + (((int)threadIdx.x) >> 4))]), 0.000000e+00f);
}

dim3 grid(128, 1, 1);
dim3 block(64, 1, 1);
best_idx 13