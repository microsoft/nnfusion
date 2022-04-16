//1_14_256_56_2_2
//128_256_56_56_64_1_1_SAME
//dim3 grid(1, 14, 256);
//dim3 block(56, 2, 2);

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
extern "C" __global__ void __launch_bounds__(224) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[32];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[128];
  #pragma unroll
  for (int ff_init = 0; ff_init < 16; ++ff_init) {
    compute1[(ff_init)] = 0.000000e+00f;
    compute1[((ff_init + 16))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 448) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((((int)blockIdx.z) >> 1) * 802816) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) < 32) {
      if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) < 128) {
        if (((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) < 64) {
          if (((int)threadIdx.x) < 32) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = placeholder1[((((((((((int)blockIdx.z) & 1) * 8192) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 2) * 256)) + (rc_outer * 4)) + (((int)threadIdx.x) & 3)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
      #pragma unroll
      for (int ff = 0; ff < 16; ++ff) {
        compute1[(ff)] = (compute1[(ff)] + (pad_temp_shared[((((rc_inner * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 64) + (ff * 4)) + rc_inner))]));
        compute1[((ff + 16))] = (compute1[((ff + 16))] + (pad_temp_shared[(((((rc_inner * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[((((((int)threadIdx.z) * 64) + (ff * 4)) + rc_inner))]));
      }
    }
  }
  #pragma unroll
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 16; ++i1_inner_inner_inner) {
    compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)))] = max((compute1[(i1_inner_inner_inner)] + input2[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)))]), 0.000000e+00f);
    compute[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) + 112))] = max((compute1[((i1_inner_inner_inner + 16))] + input2[((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (i1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) + 112))]), 0.000000e+00f);
  }
}

