
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
  float conv_local[2];
  __shared__ float data_pad_shared[864];
  __shared__ float kernel_pad_shared[432];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[1];
  conv_local[(0)] = 0.000000e+00f;
  conv_local[(1)] = 0.000000e+00f;
  data_pad_shared[(((int)threadIdx.x))] = (((((0 < ((((int)threadIdx.x) / 96) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5))) && (((((int)threadIdx.x) / 96) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5)) < 33)) && (0 < (((((int)threadIdx.x) % 96) >> 5) + (((int)threadIdx.x) & 31)))) && ((((((int)threadIdx.x) % 96) >> 5) + (((int)threadIdx.x) & 31)) < 33)) ? data[((((((((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) >> 10) * 3072) + ((((int)threadIdx.x) / 96) * 32)) + (((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5) * 32)) + ((((int)threadIdx.x) % 96) >> 5)) + (((int)threadIdx.x) & 31)) - 33))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 256))] = (((((0 < (((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5) + ((((((int)threadIdx.x) >> 5) + 8) % 9) / 3))) && ((((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5) + ((((((int)threadIdx.x) >> 5) + 8) % 9) / 3)) < 33)) && (0 < ((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)))) && (((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 2) % 3)) < 33)) ? data[(((((((((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) >> 10) * 3072) + (((((int)threadIdx.x) + 256) / 288) * 1024)) + (((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5) * 32)) + (((((((int)threadIdx.x) >> 5) + 8) % 9) / 3) * 32)) + (((int)threadIdx.x) & 31)) + (((((int)threadIdx.x) >> 5) + 2) % 3)) - 33))] : 0.000000e+00f);
  data_pad_shared[((((int)threadIdx.x) + 512))] = (((((0 < (((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5) + ((((((int)threadIdx.x) >> 5) + 7) % 9) / 3))) && ((((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5) + ((((((int)threadIdx.x) >> 5) + 7) % 9) / 3)) < 33)) && (0 < ((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 1) % 3)))) && (((((int)threadIdx.x) & 31) + (((((int)threadIdx.x) >> 5) + 1) % 3)) < 33)) ? data[(((((((((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) >> 10) * 3072) + (((((int)threadIdx.x) + 512) / 288) * 1024)) + (((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5) * 32)) + (((((((int)threadIdx.x) >> 5) + 7) % 9) / 3) * 32)) + (((int)threadIdx.x) & 31)) + (((((int)threadIdx.x) >> 5) + 1) % 3)) - 33))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 96) {
    data_pad_shared[((((int)threadIdx.x) + 768))] = (((((((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5) + (((((int)threadIdx.x) >> 5) + 6) / 3)) < 33) && (0 < ((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)))) && (((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 31)) < 33)) ? data[(((((((((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) >> 10) * 3072) + (((((int)threadIdx.x) + 768) / 288) * 1024)) + (((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) & 31)) & 1023) >> 5) * 32)) + ((((((int)threadIdx.x) >> 5) + 6) / 3) * 32)) + (((int)threadIdx.x) >> 5)) + (((int)threadIdx.x) & 31)) - 33))] : 0.000000e+00f);
  }
  kernel_pad_shared[(((int)threadIdx.x))] = kernel[(((int)threadIdx.x))];
  if (((int)threadIdx.x) < 176) {
    kernel_pad_shared[((((int)threadIdx.x) + 256))] = kernel[((((int)threadIdx.x) + 256))];
  }
  __syncthreads();
  for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 27; ++ra_fused0_inner_outer) {
    data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15)))];
    data_pad_shared_local[(1)] = data_pad_shared[((((ra_fused0_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 16))];
    kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 27) + ra_fused0_inner_outer))];
    conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
    conv_local[(1)] = (conv_local[(1)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(0)]));
  }
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 65536) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)))] = max((conv_local[(0)] + bias[((((int)threadIdx.x) >> 4))]), 0.000000e+00f);
  conv_unpad[((((((((int)threadIdx.x) >> 4) * 65536) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 16))] = max((conv_local[(1)] + bias[((((int)threadIdx.x) >> 4))]), 0.000000e+00f);
}

dim3 grid(2048, 1, 1);
dim3 block(256, 1, 1);
