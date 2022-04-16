//32768_1_1_128_1_1
//65536_1024_4096
//dim3 grid(32768, 1, 1);
//dim3 block(128, 1, 1);

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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[64];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[1024];
  for (int x_c_outer_inner_init = 0; x_c_outer_inner_init < 2; ++x_c_outer_inner_init) {
    for (int y_c_outer_inner_init = 0; y_c_outer_inner_init < 4; ++y_c_outer_inner_init) {
      compute_local[(((x_c_outer_inner_init * 4) + y_c_outer_inner_init))] = 0.000000e+00f;
      compute_local[((((x_c_outer_inner_init * 4) + y_c_outer_inner_init) + 8))] = 0.000000e+00f;
      compute_local[((((x_c_outer_inner_init * 4) + y_c_outer_inner_init) + 16))] = 0.000000e+00f;
      compute_local[((((x_c_outer_inner_init * 4) + y_c_outer_inner_init) + 24))] = 0.000000e+00f;
      compute_local[((((x_c_outer_inner_init * 4) + y_c_outer_inner_init) + 32))] = 0.000000e+00f;
      compute_local[((((x_c_outer_inner_init * 4) + y_c_outer_inner_init) + 40))] = 0.000000e+00f;
      compute_local[((((x_c_outer_inner_init * 4) + y_c_outer_inner_init) + 48))] = 0.000000e+00f;
      compute_local[((((x_c_outer_inner_init * 4) + y_c_outer_inner_init) + 56))] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      ((float2*)(A_shared + (((ax0_ax1_fused_outer_outer * 256) + (((int)threadIdx.x) * 2)))))[0] = ((float2*)(A + (((((((((int)blockIdx.x) >> 6) * 131072) + (ax0_ax1_fused_outer_outer * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer1 = 0; ax0_ax1_fused_outer_outer1 < 4; ++ax0_ax1_fused_outer_outer1) {
      ((float2*)(B_shared + (((ax0_ax1_fused_outer_outer1 * 256) + (((int)threadIdx.x) * 2)))))[0] = ((float2*)(B + ((((((k_outer_outer * 65536) + (ax0_ax1_fused_outer_outer1 * 16384)) + ((((int)threadIdx.x) >> 5) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 31) * 2)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 16; ++k_outer_inner) {
      for (int x_c_outer_inner = 0; x_c_outer_inner < 2; ++x_c_outer_inner) {
        for (int y_c_outer_inner = 0; y_c_outer_inner < 4; ++y_c_outer_inner) {
          compute_local[(((x_c_outer_inner * 4) + y_c_outer_inner))] = (compute_local[(((x_c_outer_inner * 4) + y_c_outer_inner))] + (A_shared[(((((((int)threadIdx.x) >> 3) * 32) + (x_c_outer_inner * 16)) + k_outer_inner))] * B_shared[((((k_outer_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_outer_inner))]));
          compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 8))] = (compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 8))] + (A_shared[(((((((int)threadIdx.x) >> 3) * 32) + (x_c_outer_inner * 16)) + k_outer_inner))] * B_shared[(((((k_outer_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_outer_inner) + 32))]));
          compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 16))] = (compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 16))] + (A_shared[((((((((int)threadIdx.x) >> 3) * 32) + (x_c_outer_inner * 16)) + k_outer_inner) + 512))] * B_shared[((((k_outer_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_outer_inner))]));
          compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 24))] = (compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 24))] + (A_shared[((((((((int)threadIdx.x) >> 3) * 32) + (x_c_outer_inner * 16)) + k_outer_inner) + 512))] * B_shared[(((((k_outer_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_outer_inner) + 32))]));
          compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 32))] = (compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 32))] + (A_shared[((((((((int)threadIdx.x) >> 3) * 32) + (x_c_outer_inner * 16)) + k_outer_inner) + 1024))] * B_shared[((((k_outer_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_outer_inner))]));
          compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 40))] = (compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 40))] + (A_shared[((((((((int)threadIdx.x) >> 3) * 32) + (x_c_outer_inner * 16)) + k_outer_inner) + 1024))] * B_shared[(((((k_outer_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_outer_inner) + 32))]));
          compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 48))] = (compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 48))] + (A_shared[((((((((int)threadIdx.x) >> 3) * 32) + (x_c_outer_inner * 16)) + k_outer_inner) + 1536))] * B_shared[((((k_outer_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_outer_inner))]));
          compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 56))] = (compute_local[((((x_c_outer_inner * 4) + y_c_outer_inner) + 56))] + (A_shared[((((((((int)threadIdx.x) >> 3) * 32) + (x_c_outer_inner * 16)) + k_outer_inner) + 1536))] * B_shared[(((((k_outer_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_outer_inner) + 32))]));
        }
      }
    }
  }
  for (int x_inner = 0; x_inner < 2; ++x_inner) {
    for (int y_inner = 0; y_inner < 4; ++y_inner) {
      compute[((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner))] = compute_local[(((x_inner * 4) + y_inner))];
      compute[(((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 32))] = compute_local[((((x_inner * 4) + y_inner) + 8))];
      compute[(((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 131072))] = compute_local[((((x_inner * 4) + y_inner) + 16))];
      compute[(((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 131104))] = compute_local[((((x_inner * 4) + y_inner) + 24))];
      compute[(((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 262144))] = compute_local[((((x_inner * 4) + y_inner) + 32))];
      compute[(((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 262176))] = compute_local[((((x_inner * 4) + y_inner) + 40))];
      compute[(((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 393216))] = compute_local[((((x_inner * 4) + y_inner) + 48))];
      compute[(((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 393248))] = compute_local[((((x_inner * 4) + y_inner) + 56))];
    }
  }
}

