
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
  __shared__ float data_pad_shared[128];
  __shared__ float kernel_pad_shared[4096];
  float data_pad_shared_local[1];
  float kernel_pad_shared_local[8];
  conv_local[0] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  conv_local[2] = 0.000000e+00f;
  conv_local[3] = 0.000000e+00f;
  conv_local[4] = 0.000000e+00f;
  conv_local[5] = 0.000000e+00f;
  conv_local[6] = 0.000000e+00f;
  conv_local[7] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 8; ++ra_fused0_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 128) {
      data_pad_shared[((int)threadIdx.x)] = data[((((ra_fused0_outer * 12544) + ((((int)threadIdx.x) >> 3) * 784)) + ((((int)blockIdx.x) % 98) * 8)) + (((int)threadIdx.x) & 7))];
    }
    kernel_pad_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 2048)];
    kernel_pad_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 4096)];
    kernel_pad_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 6144)];
    kernel_pad_shared[(((int)threadIdx.x) + 1024)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 8192)];
    kernel_pad_shared[(((int)threadIdx.x) + 1280)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 10240)];
    kernel_pad_shared[(((int)threadIdx.x) + 1536)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 12288)];
    kernel_pad_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336)];
    kernel_pad_shared[(((int)threadIdx.x) + 2048)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 16384)];
    kernel_pad_shared[(((int)threadIdx.x) + 2304)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 18432)];
    kernel_pad_shared[(((int)threadIdx.x) + 2560)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 20480)];
    kernel_pad_shared[(((int)threadIdx.x) + 2816)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 22528)];
    kernel_pad_shared[(((int)threadIdx.x) + 3072)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 24576)];
    kernel_pad_shared[(((int)threadIdx.x) + 3328)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 26624)];
    kernel_pad_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 28672)];
    kernel_pad_shared[(((int)threadIdx.x) + 3840)] = kernel[((((((((int)blockIdx.x) / 98) * 32768) + ((((int)threadIdx.x) >> 4) * 128)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 30720)];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 16; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 8) + (((int)threadIdx.x) & 7))];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 3) * 16) + ra_fused0_inner_outer)];
      kernel_pad_shared_local[1] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 16) + ra_fused0_inner_outer) + 512)];
      kernel_pad_shared_local[2] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 16) + ra_fused0_inner_outer) + 1024)];
      kernel_pad_shared_local[3] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 16) + ra_fused0_inner_outer) + 1536)];
      kernel_pad_shared_local[4] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 16) + ra_fused0_inner_outer) + 2048)];
      kernel_pad_shared_local[5] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 16) + ra_fused0_inner_outer) + 2560)];
      kernel_pad_shared_local[6] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 16) + ra_fused0_inner_outer) + 3072)];
      kernel_pad_shared_local[7] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 16) + ra_fused0_inner_outer) + 3584)];
      conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
      conv_local[1] = (conv_local[1] + (data_pad_shared_local[0] * kernel_pad_shared_local[1]));
      conv_local[2] = (conv_local[2] + (data_pad_shared_local[0] * kernel_pad_shared_local[2]));
      conv_local[3] = (conv_local[3] + (data_pad_shared_local[0] * kernel_pad_shared_local[3]));
      conv_local[4] = (conv_local[4] + (data_pad_shared_local[0] * kernel_pad_shared_local[4]));
      conv_local[5] = (conv_local[5] + (data_pad_shared_local[0] * kernel_pad_shared_local[5]));
      conv_local[6] = (conv_local[6] + (data_pad_shared_local[0] * kernel_pad_shared_local[6]));
      conv_local[7] = (conv_local[7] + (data_pad_shared_local[0] * kernel_pad_shared_local[7]));
    }
  }
  conv_unpad[(((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 3) * 784)) + ((((int)blockIdx.x) % 98) * 8)) + (((int)threadIdx.x) & 7))] = conv_local[0];
  conv_unpad[((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 3) * 784)) + ((((int)blockIdx.x) % 98) * 8)) + (((int)threadIdx.x) & 7)) + 25088)] = conv_local[1];
  conv_unpad[((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 3) * 784)) + ((((int)blockIdx.x) % 98) * 8)) + (((int)threadIdx.x) & 7)) + 50176)] = conv_local[2];
  conv_unpad[((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 3) * 784)) + ((((int)blockIdx.x) % 98) * 8)) + (((int)threadIdx.x) & 7)) + 75264)] = conv_local[3];
  conv_unpad[((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 3) * 784)) + ((((int)blockIdx.x) % 98) * 8)) + (((int)threadIdx.x) & 7)) + 100352)] = conv_local[4];
  conv_unpad[((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 3) * 784)) + ((((int)blockIdx.x) % 98) * 8)) + (((int)threadIdx.x) & 7)) + 125440)] = conv_local[5];
  conv_unpad[((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 3) * 784)) + ((((int)blockIdx.x) % 98) * 8)) + (((int)threadIdx.x) & 7)) + 150528)] = conv_local[6];
  conv_unpad[((((((((int)blockIdx.x) / 98) * 200704) + ((((int)threadIdx.x) >> 3) * 784)) + ((((int)blockIdx.x) % 98) * 8)) + (((int)threadIdx.x) & 7)) + 175616)] = conv_local[7];
}

dim3 grid(196, 1, 1);
dim3 block(256, 1, 1);
best_idx 0