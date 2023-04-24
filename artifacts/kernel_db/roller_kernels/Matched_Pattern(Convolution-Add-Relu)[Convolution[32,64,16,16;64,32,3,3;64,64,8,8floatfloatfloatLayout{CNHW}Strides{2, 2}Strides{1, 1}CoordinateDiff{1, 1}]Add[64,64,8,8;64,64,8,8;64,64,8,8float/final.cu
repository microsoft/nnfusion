
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
  __shared__ float data_pad_shared[2048];
  __shared__ float kernel_pad_shared[1024];
  float data_pad_shared_local[4];
  float kernel_pad_shared_local[2];
  conv_local[(0)] = 0.000000e+00f;
  conv_local[(4)] = 0.000000e+00f;
  conv_local[(1)] = 0.000000e+00f;
  conv_local[(5)] = 0.000000e+00f;
  conv_local[(2)] = 0.000000e+00f;
  conv_local[(6)] = 0.000000e+00f;
  conv_local[(3)] = 0.000000e+00f;
  conv_local[(7)] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 9; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = (((0 < ((((((int)threadIdx.x) & 63) >> 3) * 2) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) % 9) / 3))) && (0 < (((((int)threadIdx.x) & 7) * 2) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) % 3)))) ? data[(((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) / 9) * 16384) + ((((int)blockIdx.x) & 63) * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) % 3)) - 17))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 256))] = (((0 < ((((((int)threadIdx.x) & 63) >> 3) * 2) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 4) % 9) / 3))) && (0 < (((((int)threadIdx.x) & 7) * 2) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 1) % 3)))) ? data[((((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 4) / 9) * 16384) + ((((int)blockIdx.x) & 63) * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 4) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 1) % 3)) - 17))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 512))] = (((0 < ((((((int)threadIdx.x) & 63) >> 3) * 2) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 8) % 9) / 3))) && (0 < (((((int)threadIdx.x) & 7) * 2) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 2) % 3)))) ? data[((((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 8) / 9) * 16384) + ((((int)blockIdx.x) & 63) * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 8) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 2) % 3)) - 17))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 768))] = (((0 < ((((((int)threadIdx.x) & 63) >> 3) * 2) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 3) % 9) / 3))) && (0 < (((((int)threadIdx.x) & 7) * 2) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) % 3)))) ? data[((((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 12) / 9) * 16384) + ((((int)blockIdx.x) & 63) * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 3) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) % 3)) - 17))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1024))] = (((0 < ((((((int)threadIdx.x) & 63) >> 3) * 2) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 7) % 9) / 3))) && (0 < (((((int)threadIdx.x) & 7) * 2) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 1) % 3)))) ? data[((((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 16) / 9) * 16384) + ((((int)blockIdx.x) & 63) * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 7) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 1) % 3)) - 17))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1280))] = (((0 < ((((((int)threadIdx.x) & 63) >> 3) * 2) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 2) % 9) / 3))) && (0 < (((((int)threadIdx.x) & 7) * 2) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 2) % 3)))) ? data[((((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 20) / 9) * 16384) + ((((int)blockIdx.x) & 63) * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 2) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 2) % 3)) - 17))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1536))] = (((0 < ((((((int)threadIdx.x) & 63) >> 3) * 2) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 6) % 9) / 3))) && (0 < (((((int)threadIdx.x) & 7) * 2) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) % 3)))) ? data[((((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 24) / 9) * 16384) + ((((int)blockIdx.x) & 63) * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 6) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + (((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) % 3)) - 17))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1792))] = (((0 < ((((((int)threadIdx.x) & 63) >> 3) * 2) + (((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 1) % 9) / 3))) && (0 < (((((int)threadIdx.x) & 7) * 2) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 1) % 3)))) ? data[((((((((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 28) / 9) * 16384) + ((((int)blockIdx.x) & 63) * 256)) + (((((int)threadIdx.x) & 63) >> 3) * 32)) + ((((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 1) % 9) / 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ((((ra_fused0_outer * 32) + (((int)threadIdx.x) >> 6)) + 1) % 3)) - 17))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) >> 6) * 9216) + ((((int)threadIdx.x) >> 5) * 288)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)))];
    kernel_pad_shared[((((int)threadIdx.x) + 256))] = kernel[(((((((((int)blockIdx.x) >> 6) * 9216) + ((((int)threadIdx.x) >> 5) * 288)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 2304))];
    kernel_pad_shared[((((int)threadIdx.x) + 512))] = kernel[(((((((((int)blockIdx.x) >> 6) * 9216) + ((((int)threadIdx.x) >> 5) * 288)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 4608))];
    kernel_pad_shared[((((int)threadIdx.x) + 768))] = kernel[(((((((((int)blockIdx.x) >> 6) * 9216) + ((((int)threadIdx.x) >> 5) * 288)) + (ra_fused0_outer * 32)) + (((int)threadIdx.x) & 31)) + 6912))];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 32; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)))];
      data_pad_shared_local[(1)] = data_pad_shared[((((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)) + 16))];
      data_pad_shared_local[(2)] = data_pad_shared[((((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)) + 32))];
      data_pad_shared_local[(3)] = data_pad_shared[((((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)) + 48))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer))];
      kernel_pad_shared_local[(1)] = kernel_pad_shared[(((((((int)threadIdx.x) >> 4) * 32) + ra_fused0_inner_outer) + 512))];
      conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
      conv_local[(4)] = (conv_local[(4)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(1)]));
      conv_local[(1)] = (conv_local[(1)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(0)]));
      conv_local[(5)] = (conv_local[(5)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(1)]));
      conv_local[(2)] = (conv_local[(2)] + (data_pad_shared_local[(2)] * kernel_pad_shared_local[(0)]));
      conv_local[(6)] = (conv_local[(6)] + (data_pad_shared_local[(2)] * kernel_pad_shared_local[(1)]));
      conv_local[(3)] = (conv_local[(3)] + (data_pad_shared_local[(3)] * kernel_pad_shared_local[(0)]));
      conv_local[(7)] = (conv_local[(7)] + (data_pad_shared_local[(3)] * kernel_pad_shared_local[(1)]));
    }
  }
  conv_unpad[((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + (((int)threadIdx.x) & 15)))] = max((conv_local[(0)] + bias[((((((int)blockIdx.x) >> 6) * 32) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + (((int)threadIdx.x) & 15)) + 65536))] = max((conv_local[(4)] + bias[(((((((int)blockIdx.x) >> 6) * 32) + (((int)threadIdx.x) >> 4)) + 16))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + (((int)threadIdx.x) & 15)) + 16))] = max((conv_local[(1)] + bias[((((((int)blockIdx.x) >> 6) * 32) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + (((int)threadIdx.x) & 15)) + 65552))] = max((conv_local[(5)] + bias[(((((((int)blockIdx.x) >> 6) * 32) + (((int)threadIdx.x) >> 4)) + 16))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + (((int)threadIdx.x) & 15)) + 32))] = max((conv_local[(2)] + bias[((((((int)blockIdx.x) >> 6) * 32) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + (((int)threadIdx.x) & 15)) + 65568))] = max((conv_local[(6)] + bias[(((((((int)blockIdx.x) >> 6) * 32) + (((int)threadIdx.x) >> 4)) + 16))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + (((int)threadIdx.x) & 15)) + 48))] = max((conv_local[(3)] + bias[((((((int)blockIdx.x) >> 6) * 32) + (((int)threadIdx.x) >> 4)))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) >> 6) * 131072) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + (((int)threadIdx.x) & 15)) + 65584))] = max((conv_local[(7)] + bias[(((((((int)blockIdx.x) >> 6) * 32) + (((int)threadIdx.x) >> 4)) + 16))]), 0.000000e+00f);
}

dim3 grid(128, 1, 1);
dim3 block(256, 1, 1);
