//1_1_1024_14_2_16
//128_1024_14_14_512_1_1_SAME
//dim3 grid(1, 1, 1024);
//dim3 block(14, 2, 16);

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
extern "C" __global__ void __launch_bounds__(448) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[28];
  __shared__ float pad_temp_shared[3136];
  __shared__ float placeholder_shared[1024];
  #pragma unroll
  for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
    compute_local[(ff_c_init)] = 0.000000e+00f;
    compute_local[((ff_c_init + 4))] = 0.000000e+00f;
    compute_local[((ff_c_init + 8))] = 0.000000e+00f;
    compute_local[((ff_c_init + 12))] = 0.000000e+00f;
    compute_local[((ff_c_init + 16))] = 0.000000e+00f;
    compute_local[((ff_c_init + 20))] = 0.000000e+00f;
    compute_local[((ff_c_init + 24))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 196) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((((((int)blockIdx.z) >> 3) * 200704) + (rc_outer * 3136)) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 98)) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 64) {
        if (((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if ((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 32) {
              placeholder_shared[(((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((((int)blockIdx.z) & 7) * 65536) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 1024)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
            }
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      #pragma unroll
      for (int ff_c = 0; ff_c < 4; ++ff_c) {
        compute_local[(ff_c)] = (compute_local[(ff_c)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 64) + (ff_c * 16)) + rc_inner))]));
        compute_local[((ff_c + 4))] = (compute_local[((ff_c + 4))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 64) + (ff_c * 16)) + rc_inner))]));
        compute_local[((ff_c + 8))] = (compute_local[((ff_c + 8))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 64) + (ff_c * 16)) + rc_inner))]));
        compute_local[((ff_c + 12))] = (compute_local[((ff_c + 12))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 64) + (ff_c * 16)) + rc_inner))]));
        compute_local[((ff_c + 16))] = (compute_local[((ff_c + 16))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[((((((int)threadIdx.z) * 64) + (ff_c * 16)) + rc_inner))]));
        compute_local[((ff_c + 20))] = (compute_local[((ff_c + 20))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[((((((int)threadIdx.z) * 64) + (ff_c * 16)) + rc_inner))]));
        compute_local[((ff_c + 24))] = (compute_local[((ff_c + 24))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[((((((int)threadIdx.z) * 64) + (ff_c * 16)) + rc_inner))]));
      }
    }
  }
  #pragma unroll
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
    compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (ff_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = compute_local[(ff_inner_inner_inner)];
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (ff_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] = compute_local[((ff_inner_inner_inner + 4))];
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (ff_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] = compute_local[((ff_inner_inner_inner + 8))];
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (ff_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] = compute_local[((ff_inner_inner_inner + 12))];
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (ff_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] = compute_local[((ff_inner_inner_inner + 16))];
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (ff_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] = compute_local[((ff_inner_inner_inner + 20))];
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (ff_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] = compute_local[((ff_inner_inner_inner + 24))];
  }
}

