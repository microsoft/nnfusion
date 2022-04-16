//4096_1_1_256_1_1
//8192_1024_4096
//dim3 grid(4096, 1, 1);
//dim3 block(256, 1, 1);

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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[32];
  __shared__ float A_shared[4096];
  __shared__ float B_shared[2048];
  for (int y_c_inner_init = 0; y_c_inner_init < 4; ++y_c_inner_init) {
    compute_local[(y_c_inner_init)] = 0.000000e+00f;
    compute_local[((y_c_inner_init + 4))] = 0.000000e+00f;
    compute_local[((y_c_inner_init + 8))] = 0.000000e+00f;
    compute_local[((y_c_inner_init + 12))] = 0.000000e+00f;
    compute_local[((y_c_inner_init + 16))] = 0.000000e+00f;
    compute_local[((y_c_inner_init + 20))] = 0.000000e+00f;
    compute_local[((y_c_inner_init + 24))] = 0.000000e+00f;
    compute_local[((y_c_inner_init + 28))] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 16; ++ax0_ax1_fused_outer_outer) {
      A_shared[(((ax0_ax1_fused_outer_outer * 256) + ((int)threadIdx.x)))] = A[(((((((((int)blockIdx.x) >> 6) * 131072) + (ax0_ax1_fused_outer_outer * 8192)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)))];
    }
    for (int ax0_ax1_fused_outer_outer1 = 0; ax0_ax1_fused_outer_outer1 < 4; ++ax0_ax1_fused_outer_outer1) {
      ((float2*)(B_shared + (((ax0_ax1_fused_outer_outer1 * 512) + (((int)threadIdx.x) * 2)))))[0] = ((float2*)(B + ((((((k_outer_outer * 131072) + (ax0_ax1_fused_outer_outer1 * 32768)) + ((((int)threadIdx.x) >> 5) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 31) * 2)))))[0];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 32; ++k_inner) {
      for (int y_c_inner = 0; y_c_inner < 4; ++y_c_inner) {
        compute_local[(y_c_inner)] = (compute_local[(y_c_inner)] + (A_shared[((((((int)threadIdx.x) >> 3) * 32) + k_inner))] * B_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_inner))]));
        compute_local[((y_c_inner + 4))] = (compute_local[((y_c_inner + 4))] + (A_shared[((((((int)threadIdx.x) >> 3) * 32) + k_inner))] * B_shared[(((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_inner) + 32))]));
        compute_local[((y_c_inner + 8))] = (compute_local[((y_c_inner + 8))] + (A_shared[(((((((int)threadIdx.x) >> 3) * 32) + k_inner) + 1024))] * B_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_inner))]));
        compute_local[((y_c_inner + 12))] = (compute_local[((y_c_inner + 12))] + (A_shared[(((((((int)threadIdx.x) >> 3) * 32) + k_inner) + 1024))] * B_shared[(((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_inner) + 32))]));
        compute_local[((y_c_inner + 16))] = (compute_local[((y_c_inner + 16))] + (A_shared[(((((((int)threadIdx.x) >> 3) * 32) + k_inner) + 2048))] * B_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_inner))]));
        compute_local[((y_c_inner + 20))] = (compute_local[((y_c_inner + 20))] + (A_shared[(((((((int)threadIdx.x) >> 3) * 32) + k_inner) + 2048))] * B_shared[(((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_inner) + 32))]));
        compute_local[((y_c_inner + 24))] = (compute_local[((y_c_inner + 24))] + (A_shared[(((((((int)threadIdx.x) >> 3) * 32) + k_inner) + 3072))] * B_shared[((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_inner))]));
        compute_local[((y_c_inner + 28))] = (compute_local[((y_c_inner + 28))] + (A_shared[(((((((int)threadIdx.x) >> 3) * 32) + k_inner) + 3072))] * B_shared[(((((k_inner * 64) + ((((int)threadIdx.x) & 7) * 4)) + y_c_inner) + 32))]));
      }
    }
  }
  for (int y_inner = 0; y_inner < 4; ++y_inner) {
    compute[(((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner))] = compute_local[(y_inner)];
    compute[((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 32))] = compute_local[((y_inner + 4))];
    compute[((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 131072))] = compute_local[((y_inner + 8))];
    compute[((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 131104))] = compute_local[((y_inner + 12))];
    compute[((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 262144))] = compute_local[((y_inner + 16))];
    compute[((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 262176))] = compute_local[((y_inner + 20))];
    compute[((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 393216))] = compute_local[((y_inner + 24))];
    compute[((((((((((int)blockIdx.x) >> 6) * 524288) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.x) & 63) * 64)) + ((((int)threadIdx.x) & 7) * 4)) + y_inner) + 393248))] = compute_local[((y_inner + 28))];
  }
}

