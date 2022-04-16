//1_1_512_14_2_16
//128_1024_14_14_256_1_1_SAME
//dim3 grid(1, 1, 512);
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
extern "C" __global__ void __launch_bounds__(448) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[28];
  __shared__ float pad_temp_shared[6272];
  __shared__ float placeholder_shared[2048];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    compute1[(ff_init)] = 0.000000e+00f;
    compute1[((ff_init + 4))] = 0.000000e+00f;
    compute1[((ff_init + 8))] = 0.000000e+00f;
    compute1[((ff_init + 12))] = 0.000000e+00f;
    compute1[((ff_init + 16))] = 0.000000e+00f;
    compute1[((ff_init + 20))] = 0.000000e+00f;
    compute1[((ff_init + 24))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 14; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 392) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 14)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((((((int)blockIdx.z) >> 2) * 200704) + (rc_outer * 6272)) + (((int)threadIdx.z) * 392)) + (((int)threadIdx.y) * 196)) + (((int)threadIdx.x) * 14)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 5)) < 64) {
        if (((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 2048) {
          if ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 128) {
            if (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
              placeholder_shared[(((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((((int)blockIdx.z) & 3) * 65536) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + ((((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 5) * 1024)) + (rc_outer * 32)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 31)))];
            }
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 32; ++rc_inner) {
      #pragma unroll
      for (int ff = 0; ff < 4; ++ff) {
        compute1[(ff)] = (compute1[(ff)] + (pad_temp_shared[((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 128) + (ff * 32)) + rc_inner))]));
        compute1[((ff + 4))] = (compute1[((ff + 4))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 128) + (ff * 32)) + rc_inner))]));
        compute1[((ff + 8))] = (compute1[((ff + 8))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 128) + (ff * 32)) + rc_inner))]));
        compute1[((ff + 12))] = (compute1[((ff + 12))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 128) + (ff * 32)) + rc_inner))]));
        compute1[((ff + 16))] = (compute1[((ff + 16))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] * placeholder_shared[((((((int)threadIdx.z) * 128) + (ff * 32)) + rc_inner))]));
        compute1[((ff + 20))] = (compute1[((ff + 20))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] * placeholder_shared[((((((int)threadIdx.z) * 128) + (ff * 32)) + rc_inner))]));
        compute1[((ff + 24))] = (compute1[((ff + 24))] + (pad_temp_shared[(((((rc_inner * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] * placeholder_shared[((((((int)threadIdx.z) * 128) + (ff * 32)) + rc_inner))]));
      }
    }
  }
  #pragma unroll
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 4; ++i1_inner_inner_inner) {
    compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = max((compute1[(i1_inner_inner_inner)] + input2[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] = max((compute1[((i1_inner_inner_inner + 4))] + input2[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] = max((compute1[((i1_inner_inner_inner + 8))] + input2[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] = max((compute1[((i1_inner_inner_inner + 12))] + input2[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] = max((compute1[((i1_inner_inner_inner + 16))] + input2[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] = max((compute1[((i1_inner_inner_inner + 20))] + input2[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] = max((compute1[((i1_inner_inner_inner + 24))] + input2[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (i1_inner_inner_inner * 196)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))]), 0.000000e+00f);
  }
}

