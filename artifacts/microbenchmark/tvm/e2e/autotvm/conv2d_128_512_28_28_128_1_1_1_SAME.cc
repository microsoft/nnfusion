//1_7_256_28_1_8
//128_512_28_28_128_1_1_SAME
//dim3 grid(1, 7, 256);
//dim3 block(28, 1, 8);

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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[32];
  __shared__ float pad_temp_shared[224];
  __shared__ float placeholder_shared[128];
  for (int ff_c_init = 0; ff_c_init < 8; ++ff_c_init) {
    compute_local[(ff_c_init)] = 0.000000e+00f;
    compute_local[((ff_c_init + 8))] = 0.000000e+00f;
    compute_local[((ff_c_init + 16))] = 0.000000e+00f;
    compute_local[((ff_c_init + 24))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 256; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 28) + ((int)threadIdx.x)))] = placeholder[((((((((((int)blockIdx.z) >> 1) * 401408) + (rc_outer * 1568)) + ((((int)threadIdx.z) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + ((((int)threadIdx.z) & 3) * 28)) + ((int)threadIdx.x)))];
    if (((((int)threadIdx.z) * 8) + (((int)threadIdx.x) >> 1)) < 64) {
      if (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) < 128) {
        if (((int)threadIdx.x) < 16) {
          placeholder_shared[(((((int)threadIdx.z) * 16) + ((int)threadIdx.x)))] = placeholder1[(((((((((int)blockIdx.z) & 1) * 32768) + (((int)threadIdx.z) * 4096)) + ((((int)threadIdx.x) >> 1) * 512)) + (rc_outer * 2)) + (((int)threadIdx.x) & 1)))];
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
      for (int ff_c = 0; ff_c < 8; ++ff_c) {
        compute_local[(ff_c)] = (compute_local[(ff_c)] + (pad_temp_shared[(((rc_inner * 112) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff_c * 2)) + rc_inner))]));
        compute_local[((ff_c + 8))] = (compute_local[((ff_c + 8))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff_c * 2)) + rc_inner))]));
        compute_local[((ff_c + 16))] = (compute_local[((ff_c + 16))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff_c * 2)) + rc_inner))]));
        compute_local[((ff_c + 24))] = (compute_local[((ff_c + 24))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff_c * 2)) + rc_inner))]));
      }
    }
  }
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 8; ++ff_inner_inner_inner) {
    compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)))] = compute_local[(ff_inner_inner_inner)];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 28))] = compute_local[((ff_inner_inner_inner + 8))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 56))] = compute_local[((ff_inner_inner_inner + 16))];
    compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 6272)) + (ff_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 84))] = compute_local[((ff_inner_inner_inner + 24))];
  }
}

