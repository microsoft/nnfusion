#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void __launch_bounds__(128) tvm_conv_default_kernel(float* __restrict__ X, float* __restrict__ K, float* __restrict__ Y) {
  float Y_local[64];
  __shared__ float PaddedX_shared[384];
  __shared__ float K_shared[96];
  float PaddedX_shared_local[16];
  float K_shared_local[24];
  for (int c_c_init = 0; c_c_init < 8; ++c_c_init) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        Y_local[((((c_c_init * 4) + (i_c_init * 2)) + j_c_init))] = 0.000000e+00f;
        Y_local[(((((c_c_init * 4) + (i_c_init * 2)) + j_c_init) + 32))] = 0.000000e+00f;
      }
    }
  }
  for (int ric_outer = 0; ric_outer < 64; ++ric_outer) {
    for (int rkw_outer = 0; rkw_outer < 3; ++rkw_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_inner_inner_inner) {
        PaddedX_shared[(((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_inner_inner_inner))] = (((((((((int)blockIdx.y) * 4) + (((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_inner_inner_inner) >> 6)) < 1) || (65 <= ((((int)blockIdx.y) * 4) + (((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_inner_inner_inner) >> 6)))) || ((rkw_outer + (((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_inner_inner_inner) & 63)) < 1)) || (65 <= (rkw_outer + (((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_inner_inner_inner) & 63)))) ? 0.000000e+00f : X[(((((((((ric_outer * 4096) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.z) * 96)) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 3)) + rkw_outer) + ax0_ax1_fused_ax2_fused_inner_inner_inner) - 65))]);
      }
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) / 3)) < 32) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 12)) + ((int)threadIdx.x)) < 96) {
          if (((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) < 24) {
            if (((int)threadIdx.x) < 12) {
              K_shared[((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 12)) + ((int)threadIdx.x)))] = K[((((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 4608)) + (((int)threadIdx.y) * 2304)) + ((((int)threadIdx.x) / 3) * 576)) + (ric_outer * 9)) + ((((int)threadIdx.x) % 3) * 3)) + rkw_outer))];
            }
          }
        }
      }
      __syncthreads();
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        for (int ax2 = 0; ax2 < 2; ++ax2) {
          PaddedX_shared_local[(((ax1 * 2) + ax2))] = PaddedX_shared[(((((((int)threadIdx.y) * 128) + (ax1 * 64)) + (((int)threadIdx.x) * 2)) + ax2))];
          PaddedX_shared_local[((((ax1 * 2) + ax2) + 8))] = PaddedX_shared[((((((((int)threadIdx.y) * 128) + (ax1 * 64)) + (((int)threadIdx.x) * 2)) + ax2) + 32))];
        }
      }
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        for (int ax21 = 0; ax21 < 3; ++ax21) {
          K_shared_local[(((ax0 * 3) + ax21))] = K_shared[((((((int)threadIdx.z) * 24) + (ax0 * 3)) + ax21))];
        }
      }
      for (int rkh_inner_inner = 0; rkh_inner_inner < 3; ++rkh_inner_inner) {
        for (int c_c = 0; c_c < 8; ++c_c) {
          for (int i_c = 0; i_c < 2; ++i_c) {
            for (int j_c = 0; j_c < 2; ++j_c) {
              Y_local[((((c_c * 4) + (i_c * 2)) + j_c))] = (Y_local[((((c_c * 4) + (i_c * 2)) + j_c))] + (PaddedX_shared_local[((((i_c * 2) + (rkh_inner_inner * 2)) + j_c))] * K_shared_local[(((c_c * 3) + rkh_inner_inner))]));
              Y_local[(((((c_c * 4) + (i_c * 2)) + j_c) + 32))] = (Y_local[(((((c_c * 4) + (i_c * 2)) + j_c) + 32))] + (PaddedX_shared_local[(((((i_c * 2) + (rkh_inner_inner * 2)) + j_c) + 8))] * K_shared_local[(((c_c * 3) + rkh_inner_inner))]));
            }
          }
        }
      }
    }
  }
  for (int c_inner_inner_inner = 0; c_inner_inner_inner < 8; ++c_inner_inner_inner) {
    for (int i_inner_inner_inner = 0; i_inner_inner_inner < 2; ++i_inner_inner_inner) {
      for (int j_inner_inner_inner = 0; j_inner_inner_inner < 2; ++j_inner_inner_inner) {
        Y[(((((((((((int)blockIdx.z) * 131072) + (((int)threadIdx.z) * 32768)) + (c_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 128)) + (i_inner_inner_inner * 64)) + (((int)threadIdx.x) * 2)) + j_inner_inner_inner))] = Y_local[((((c_inner_inner_inner * 4) + (i_inner_inner_inner * 2)) + j_inner_inner_inner))];
        Y[((((((((((((int)blockIdx.z) * 131072) + (((int)threadIdx.z) * 32768)) + (c_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 128)) + (i_inner_inner_inner * 64)) + (((int)threadIdx.x) * 2)) + j_inner_inner_inner) + 32))] = Y_local[(((((c_inner_inner_inner * 4) + (i_inner_inner_inner * 2)) + j_inner_inner_inner) + 32))];
      }
    }
  }
}

__global__ void __launch_bounds__(256)
    conv_kernel(float *__restrict__ x, float *__restrict__ w,
                float *__restrict__ output) {
    int x_start = blockIdx.x * 8, y_start = blockIdx.y * 8;
    float temp[4][4];
    float kernel[3][3];
    __shared__ float shared_x[10][10];
    int ui = (threadIdx.x / 128) * 4;
    int uj = ((threadIdx.x / 64) & 1) * 4;
    int oc = threadIdx.x % 64;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            temp[i][j] = 0;

    for (int c = 0; c < 64; c++) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                kernel[i][j] = w[oc * 576 + c * 9 + i * 3 + j];
        for (int id = threadIdx.x; id < 10 * 10; id += blockDim.x) {
            int i = id / 10, j = id % 10;
            int ty = y_start + i - 1, tx = x_start + j - 1;
            shared_x[i][j] = (ty >= 0 && ty < 64 && tx >= 0 && tx < 64) ?
                                 x[c * 4096 + ty * 64 + tx] :
                                 0.0;
        }
        __syncthreads();

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int ki = 0; ki < 3; ki++)
                    for (int kj = 0; kj < 3; kj++) {
                        temp[i][j] +=
                            kernel[ki][kj] * shared_x[i + ui + ki][j + uj + kj];
                    }
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            output[oc * 4096 + (y_start + i + ui) * 64 + (x_start + j + uj)] =
                temp[i][j];
}

extern "C" __global__ void __launch_bounds__(128) tvm_conv_tiled(float* __restrict__ X, float* __restrict__ K, float* __restrict__ Y) {
  float Y_local[32];
  __shared__ float PaddedX_shared[100];
  __shared__ float K_shared[192];
  float PaddedX_shared_local[8];
  float K_shared_local[24];
  for (int c_c_init = 0; c_c_init < 8; ++c_c_init) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        Y_local[((((c_c_init * 4) + (i_c_init * 2)) + j_c_init))] = 0.000000e+00f;
      }
    }
  }
  for (int ric_outer = 0; ric_outer < 64; ++ric_outer) {
    if ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 100) {
      if (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) < 13) {
        PaddedX_shared[((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)))] = (((((((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 10)) < 1) || (65 <= ((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 10)))) || (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 10)) < 1)) || (65 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 10)))) ? 0.000000e+00f : X[(((((((ric_outer * 4096) + (((int)blockIdx.y) * 512)) + (((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 10) * 64)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 10)) - 65))]);
      }
    }
    for (int rkw_outer = 0; rkw_outer < 3; ++rkw_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
        if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 3)) < 64) {
          if (((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 192) {
            if ((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 24) {
              if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 6) {
                K_shared[(((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = K[(((((((((int)threadIdx.z) * 4608) + (((int)threadIdx.y) * 1152)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 3) * 576)) + (ric_outer * 9)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 3) * 3)) + rkw_outer))];
              }
            }
          }
        }
      }
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        for (int ax2 = 0; ax2 < 2; ++ax2) {
          PaddedX_shared_local[(((ax1 * 2) + ax2))] = PaddedX_shared[((((((((int)threadIdx.y) * 20) + (ax1 * 10)) + (((int)threadIdx.x) * 2)) + ax2) + rkw_outer))];
        }
      }
      __syncthreads();
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        for (int ax21 = 0; ax21 < 3; ++ax21) {
          K_shared_local[(((ax0 * 3) + ax21))] = K_shared[((((((int)threadIdx.z) * 24) + (ax0 * 3)) + ax21))];
        }
      }
      for (int rkh_inner_inner = 0; rkh_inner_inner < 3; ++rkh_inner_inner) {
        for (int c_c = 0; c_c < 8; ++c_c) {
          for (int i_c = 0; i_c < 2; ++i_c) {
            for (int j_c = 0; j_c < 2; ++j_c) {
              Y_local[((((c_c * 4) + (i_c * 2)) + j_c))] = (Y_local[((((c_c * 4) + (i_c * 2)) + j_c))] + (PaddedX_shared_local[((((i_c * 2) + (rkh_inner_inner * 2)) + j_c))] * K_shared_local[(((c_c * 3) + rkh_inner_inner))]));
            }
          }
        }
      }
    }
  }
  for (int c_inner_inner = 0; c_inner_inner < 8; ++c_inner_inner) {
    for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
      for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
        Y[(((((((((((int)threadIdx.z) * 32768) + (c_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 128)) + (i_inner_inner * 64)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + j_inner_inner))] = Y_local[((((c_inner_inner * 4) + (i_inner_inner * 2)) + j_inner_inner))];
      }
    }
  }
}

extern "C" {

void conv(float *x, float *w, float *output) {
    tvm_conv_tiled<<<dim3(8, 8, 1), dim3(4, 4, 8)>>>(x, w, output);
}

void conv_tvm(float *x, float *w, float *output) {
    tvm_conv_default_kernel<<<dim3(1, 16, 2), dim3(16, 2, 4)>>>(x, w, output);
}

}
