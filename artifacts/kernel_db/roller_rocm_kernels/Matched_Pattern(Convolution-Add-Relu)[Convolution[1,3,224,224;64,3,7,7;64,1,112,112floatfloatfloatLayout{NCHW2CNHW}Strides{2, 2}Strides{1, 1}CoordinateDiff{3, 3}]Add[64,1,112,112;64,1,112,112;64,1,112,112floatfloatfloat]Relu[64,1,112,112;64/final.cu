
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ bias, float* __restrict__ conv_unpad) {
  float conv_local[8];
  __shared__ float data_pad_shared[1024];
  __shared__ float kernel_pad_shared[512];
  float data_pad_shared_local[8];
  float kernel_pad_shared_local[1];
  conv_local[0] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  conv_local[2] = 0.000000e+00f;
  conv_local[3] = 0.000000e+00f;
  conv_local[4] = 0.000000e+00f;
  conv_local[5] = 0.000000e+00f;
  conv_local[6] = 0.000000e+00f;
  conv_local[7] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 10; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = ((((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) < 147) && (3 <= ((((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 2) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) % 49) / 7)))) && (((((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 2) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) % 49) / 7)) < 227)) && (3 <= ((((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 6)) % 7)))) && (((((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 6)) % 7)) < 227)) ? data[(((((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) / 49) * 50176) + (((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 448)) + (((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) % 49) / 7) * 224)) + (((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 6)) % 7)) - 675)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 256)] = ((((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) < 143) && (3 <= ((((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 2) + (((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 4) % 49) / 7)))) && (((((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 2) + (((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 4) % 49) / 7)) < 227)) && (3 <= ((((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 4) % 7)))) && (((((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 4) % 7)) < 227)) ? data[((((((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 4) / 49) * 50176) + (((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 448)) + ((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 4) % 49) / 7) * 224)) + (((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2)) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 4) % 7)) - 675)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 512)] = ((((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) < 139) && (3 <= ((((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 2) + (((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 8) % 49) / 7)))) && (((((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 2) + (((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 8) % 49) / 7)) < 227)) && (3 <= ((((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 1) % 7)))) && (((((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 1) % 7)) < 227)) ? data[((((((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 8) / 49) * 50176) + (((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 448)) + ((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 8) % 49) / 7) * 224)) + (((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2)) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 1) % 7)) - 675)] : 0.000000e+00f);
    data_pad_shared[(((int)threadIdx.x) + 768)] = ((((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) < 135) && (3 <= ((((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 2) + (((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 12) % 49) / 7)))) && (((((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 2) + (((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 12) % 49) / 7)) < 227)) && (3 <= ((((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 5) % 7)))) && (((((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 5) % 7)) < 227)) ? data[((((((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 12) / 49) * 50176) + (((((((int)blockIdx.x) % 196) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 7) * 448)) + ((((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 12) % 49) / 7) * 224)) + (((((((int)blockIdx.x) % 196) * 64) + (((int)threadIdx.x) & 63)) % 112) * 2)) + ((((ra_fused0_outer * 16) + (((int)threadIdx.x) >> 6)) + 5) % 7)) - 675)] : 0.000000e+00f);
    kernel_pad_shared[((int)threadIdx.x)] = ((((ra_fused0_outer * 16) + (((int)threadIdx.x) & 15)) < 147) ? kernel[(((((((int)blockIdx.x) / 196) * 4704) + ((((int)threadIdx.x) >> 4) * 147)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = ((((ra_fused0_outer * 16) + (((int)threadIdx.x) & 15)) < 147) ? kernel[((((((((int)blockIdx.x) / 196) * 4704) + ((((int)threadIdx.x) >> 4) * 147)) + (ra_fused0_outer * 16)) + (((int)threadIdx.x) & 15)) + 2352)] : 0.000000e+00f);
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 16; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 7))];
      data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 7)) + 8)];
      data_pad_shared_local[2] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 7)) + 16)];
      data_pad_shared_local[3] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 7)) + 24)];
      data_pad_shared_local[4] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 7)) + 32)];
      data_pad_shared_local[5] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 7)) + 40)];
      data_pad_shared_local[6] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 7)) + 48)];
      data_pad_shared_local[7] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 7)) + 56)];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 3) * 16) + ra_fused0_inner_outer)];
      if (((ra_fused0_outer * 16) + ra_fused0_inner_outer) < 147) {
        conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
        conv_local[1] = (conv_local[1] + (data_pad_shared_local[1] * kernel_pad_shared_local[0]));
        conv_local[2] = (conv_local[2] + (data_pad_shared_local[2] * kernel_pad_shared_local[0]));
        conv_local[3] = (conv_local[3] + (data_pad_shared_local[3] * kernel_pad_shared_local[0]));
        conv_local[4] = (conv_local[4] + (data_pad_shared_local[4] * kernel_pad_shared_local[0]));
        conv_local[5] = (conv_local[5] + (data_pad_shared_local[5] * kernel_pad_shared_local[0]));
        conv_local[6] = (conv_local[6] + (data_pad_shared_local[6] * kernel_pad_shared_local[0]));
        conv_local[7] = (conv_local[7] + (data_pad_shared_local[7] * kernel_pad_shared_local[0]));
      }
    }
  }
  conv_unpad[(((((((int)blockIdx.x) / 196) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + ((((int)blockIdx.x) % 196) * 64)) + (((int)threadIdx.x) & 7))] = max((conv_local[0] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 196) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + ((((int)blockIdx.x) % 196) * 64)) + (((int)threadIdx.x) & 7)) + 8)] = max((conv_local[1] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 196) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + ((((int)blockIdx.x) % 196) * 64)) + (((int)threadIdx.x) & 7)) + 16)] = max((conv_local[2] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 196) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + ((((int)blockIdx.x) % 196) * 64)) + (((int)threadIdx.x) & 7)) + 24)] = max((conv_local[3] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 196) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + ((((int)blockIdx.x) % 196) * 64)) + (((int)threadIdx.x) & 7)) + 32)] = max((conv_local[4] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 196) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + ((((int)blockIdx.x) % 196) * 64)) + (((int)threadIdx.x) & 7)) + 40)] = max((conv_local[5] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 196) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + ((((int)blockIdx.x) % 196) * 64)) + (((int)threadIdx.x) & 7)) + 48)] = max((conv_local[6] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
  conv_unpad[((((((((int)blockIdx.x) / 196) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + ((((int)blockIdx.x) % 196) * 64)) + (((int)threadIdx.x) & 7)) + 56)] = max((conv_local[7] + bias[(((((int)blockIdx.x) / 196) * 32) + (((int)threadIdx.x) >> 3))]), 0.000000e+00f);
}

dim3 grid(392, 1, 1);
dim3 block(256, 1, 1);
best_idx 4