
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv_unpad) {
  float conv_local[8];
  __shared__ float data_pad_shared[1024];
  __shared__ float kernel_pad_shared[512];
  float data_pad_shared_local[4];
  float kernel_pad_shared_local[2];
  conv_local[0] = 0.000000e+00f;
  conv_local[4] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  conv_local[5] = 0.000000e+00f;
  conv_local[2] = 0.000000e+00f;
  conv_local[6] = 0.000000e+00f;
  conv_local[3] = 0.000000e+00f;
  conv_local[7] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 16; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = (((((((int)blockIdx.x) % 13) * 4) + ((((int)threadIdx.x) & 63) >> 4)) < 49) ? data[((((ra_fused0_outer * 50176) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((((int)blockIdx.x) % 13) * 16) + ((((int)threadIdx.x) & 63) >> 2)) / 7) * 112)) + (((((((int)blockIdx.x) % 13) * 8) + (((int)threadIdx.x) & 63)) % 28) * 2))] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 256)] = (((((((int)blockIdx.x) % 13) * 4) + ((((int)threadIdx.x) & 63) >> 4)) < 49) ? data[(((((ra_fused0_outer * 50176) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((((int)blockIdx.x) % 13) * 16) + ((((int)threadIdx.x) & 63) >> 2)) / 7) * 112)) + (((((((int)blockIdx.x) % 13) * 8) + (((int)threadIdx.x) & 63)) % 28) * 2)) + 12544)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 512)] = (((((((int)blockIdx.x) % 13) * 4) + ((((int)threadIdx.x) & 63) >> 4)) < 49) ? data[(((((ra_fused0_outer * 50176) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((((int)blockIdx.x) % 13) * 16) + ((((int)threadIdx.x) & 63) >> 2)) / 7) * 112)) + (((((((int)blockIdx.x) % 13) * 8) + (((int)threadIdx.x) & 63)) % 28) * 2)) + 25088)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 768)] = (((((((int)blockIdx.x) % 13) * 4) + ((((int)threadIdx.x) & 63) >> 4)) < 49) ? data[(((((ra_fused0_outer * 50176) + ((((int)threadIdx.x) >> 6) * 3136)) + (((((((int)blockIdx.x) % 13) * 16) + ((((int)threadIdx.x) & 63) >> 2)) / 7) * 112)) + (((((((int)blockIdx.x) % 13) * 8) + (((int)threadIdx.x) & 63)) % 28) * 2)) + 37632)] : 0.000000e+00f);
    kernel_pad_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 13) * 8192) + ((((int)threadIdx.x) >> 4) * 256)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) / 13) * 8192) + ((((int)threadIdx.x) >> 4) * 256)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 4096)];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 16; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15))];
      data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)) + 16)];
      data_pad_shared_local[2] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)) + 32)];
      data_pad_shared_local[3] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)) + 48)];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 4) * 16) + ra_fused0_inner_outer)];
      kernel_pad_shared_local[1] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 16) + ra_fused0_inner_outer) + 256)];
      conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
      conv_local[4] = (conv_local[4] + (data_pad_shared_local[0] * kernel_pad_shared_local[1]));
      conv_local[1] = (conv_local[1] + (data_pad_shared_local[1] * kernel_pad_shared_local[0]));
      conv_local[5] = (conv_local[5] + (data_pad_shared_local[1] * kernel_pad_shared_local[1]));
      conv_local[2] = (conv_local[2] + (data_pad_shared_local[2] * kernel_pad_shared_local[0]));
      conv_local[6] = (conv_local[6] + (data_pad_shared_local[2] * kernel_pad_shared_local[1]));
      conv_local[3] = (conv_local[3] + (data_pad_shared_local[3] * kernel_pad_shared_local[0]));
      conv_local[7] = (conv_local[7] + (data_pad_shared_local[3] * kernel_pad_shared_local[1]));
    }
  }
  conv_unpad[(((((((int)blockIdx.x) / 13) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 13) * 64)) + (((int)threadIdx.x) & 15))] = conv_local[0];
  conv_unpad[((((((((int)blockIdx.x) / 13) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 13) * 64)) + (((int)threadIdx.x) & 15)) + 12544)] = conv_local[4];
  if ((((int)blockIdx.x) % 13) < 12) {
    conv_unpad[((((((((int)blockIdx.x) / 13) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 13) * 64)) + (((int)threadIdx.x) & 15)) + 16)] = conv_local[1];
    conv_unpad[((((((((int)blockIdx.x) / 13) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 13) * 64)) + (((int)threadIdx.x) & 15)) + 12560)] = conv_local[5];
  }
  if ((((int)blockIdx.x) % 13) < 12) {
    conv_unpad[((((((((int)blockIdx.x) / 13) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 13) * 64)) + (((int)threadIdx.x) & 15)) + 32)] = conv_local[2];
    conv_unpad[((((((((int)blockIdx.x) / 13) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 13) * 64)) + (((int)threadIdx.x) & 15)) + 12576)] = conv_local[6];
  }
  if ((((int)blockIdx.x) % 13) < 12) {
    conv_unpad[((((((((int)blockIdx.x) / 13) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 13) * 64)) + (((int)threadIdx.x) & 15)) + 48)] = conv_local[3];
    conv_unpad[((((((((int)blockIdx.x) / 13) * 25088) + ((((int)threadIdx.x) >> 4) * 784)) + ((((int)blockIdx.x) % 13) * 64)) + (((int)threadIdx.x) & 15)) + 12592)] = conv_local[7];
  }
}

dim3 grid(208, 1, 1);
dim3 block(256, 1, 1);
best_idx 8