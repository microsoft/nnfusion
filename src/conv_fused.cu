#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

static __device__ void Barrier() {
    static volatile uint8_t global_state_in[256] = {0};
    static volatile uint8_t global_state_out[256] = {0};
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    const int thread_idx_in_block = threadIdx.x + threadIdx.y * blockDim.x
                  + blockDim.x * blockDim.y * threadIdx.z;
    const int BLOCK_NUM = gridDim.x * gridDim.y * gridDim.z;
    const int THREAD_NUM = blockDim.x * blockDim.y * blockDim.z;
    if (thread_idx_in_block == 0) {
        global_state_in[block_idx] = 1;
    }
    if (block_idx == 0) {
        for (int i = thread_idx_in_block; i < BLOCK_NUM; i += THREAD_NUM)
            while (global_state_in[i] != 1) {
            }

        for (int i = thread_idx_in_block; i < BLOCK_NUM; i += THREAD_NUM)
            global_state_in[i] = 0;

        __syncthreads();
        for (int i = thread_idx_in_block; i < BLOCK_NUM; i += THREAD_NUM)
            global_state_out[i] = 1;
    }
    if (thread_idx_in_block == 0) {
        while (global_state_out[block_idx] != 1) {
        };
        global_state_out[block_idx] = 0;
    }
    __syncthreads();
    __threadfence();
}

__device__ void partly_write_back(float *__restrict__ x,
                                  float *__restrict__ shared) {
    // shared 64, 8, 8
    // x 64, 64, 64
    int x_start = blockIdx.x * 8, y_start = blockIdx.y * 8;
    const int thread_idx_in_block = threadIdx.x + threadIdx.y * blockDim.x
                  + blockDim.x * blockDim.y * threadIdx.z;
    int i = thread_idx_in_block % 8;
    int oc = thread_idx_in_block / 8;
    for (; oc < 64; oc += 16) {
        x[oc*4096 + (y_start)*64+x_start+i] = shared[oc*65+i];
        x[oc*4096 + (y_start+7)*64+x_start+i] = shared[oc*65+56+i];
        x[oc*4096 + (y_start+i)*64+x_start] = shared[oc*65+i*8];
        x[oc*4096 + (y_start+i)*64+x_start+7] = shared[oc*65+i*8+7];
    }
    // for (; oc < 64; oc += 16) {
    //   for (int j = 0; j < 8; j++)
    //     x[oc*4096+(y_start+j)*64+x_start+i] = shared[oc*65+j*8+i];
    // }
}

__device__ void conv_device_kernel(float* __restrict__ X, float* __restrict__ K, float* __restrict__ Y) {
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


// shared 64 * 8 * 8
__device__ void conv_device_kernel_shared(float* __restrict__ X, float* __restrict__ K, float* __restrict__ Y, float* shared, bool read_from_global, bool write_to_global) {
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
    if (read_from_global) {
    if ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 100) {
      if (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) < 13) {
        PaddedX_shared[((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)))] = (((((((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 10)) < 1) || (65 <= ((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 10)))) || (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 10)) < 1)) || (65 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 10)))) ? 0.000000e+00f : X[(((((((ric_outer * 4096) + (((int)blockIdx.y) * 512)) + (((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 10) * 64)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 10)) - 65))]);
      }
    }
    } else {
    if ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 100) {
      if (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) < 13) {
          int id = threadIdx.z * 13 + threadIdx.y * 4 + threadIdx.x;
          int i = id % 10 - 1;
          int j = id / 10 - 1;
          int x_start = blockIdx.x * 8, y_start = blockIdx.y * 8;
          bool edge = (x_start + i < 0) || (y_start + j < 0) || (x_start + i == 64) || (y_start + j == 64);
          bool from_global = (i < 0) || (j < 0) || (i == 8) || (j == 8);
          // PaddedX_shared[id] = from_global ? (edge ? 0.0 : X[ric_outer * 4096 + (y_start + j) * 64 + x_start + i]) : shared[ric_outer*65+j*8+i];
          if (from_global)
            PaddedX_shared[id] = edge ? 0.0 : X[ric_outer * 4096 + (y_start + j) * 64 + x_start + i];
          else
            PaddedX_shared[id] = shared[ric_outer*65+j*8+i];
          // PaddedX_shared[((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)))] = (((((((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 10)) < 1) || (65 <= ((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 10)))) || (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 10)) < 1)) || (65 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 10)))) ? 0.000000e+00f : X[(((((((ric_outer * 4096) + (((int)blockIdx.y) * 512)) + (((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 10) * 64)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.z) * 13) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 10)) - 65))]);

      }
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
  if (write_to_global) {
  for (int c_inner_inner = 0; c_inner_inner < 8; ++c_inner_inner) {
    for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
      for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
        Y[(((((((((((int)threadIdx.z) * 32768) + (c_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 128)) + (i_inner_inner * 64)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + j_inner_inner))] = Y_local[((((c_inner_inner * 4) + (i_inner_inner * 2)) + j_inner_inner))];
      }
    }
  }
  } else {
  // 8 2 2 -> 64 8 8
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        shared[(threadIdx.z*8+i)*65 + (threadIdx.y*2+j)*8+(threadIdx.x*2+k)] = Y_local[((((i * 4) + (j * 2)) + k))];
      }
    }
  }
  Barrier();
  __syncthreads();
  partly_write_back(X, shared);
  Barrier();
  }
}

// __device__ void conv_kernel_shared(float *__restrict__ x, float *__restrict__ w,
//                                    float *__restrict__ output,
//                                    float *__restrict__ shared,
//                                    bool read_from_global,
//                                    bool write_to_global) {
//     int x_start = blockIdx.x * 8, y_start = blockIdx.y * 8;
//     float temp[4][4];
//     float kernel[3][3];
//     __shared__ float shared_x[10][10];
//     int ui = (threadIdx.x / 128) * 4;
//     int uj = ((threadIdx.x / 64) & 1) * 4;
//     int oc = threadIdx.x % 64;

//     for (int i = 0; i < 4; i++)
//         for (int j = 0; j < 4; j++)
//             temp[i][j] = 0;

//     for (int c = 0; c < 64; c++) {
//         for (int i = 0; i < 3; i++)
//             for (int j = 0; j < 3; j++)
//                 kernel[i][j] = w[oc * 576 + c * 9 + i * 3 + j];
//         if (read_from_global) {
//             for (int id = threadIdx.x; id < 10 * 10; id += blockDim.x) {
//                 int i = id / 10, j = id % 10;
//                 int ty = y_start + i - 1, tx = x_start + j - 1;
//                 shared_x[i][j] = (ty >= 0 && ty < 64 && tx >= 0 && tx < 64) ?
//                                      x[c * 4096 + ty * 64 + tx] :
//                                      0.0;
//             }
//             __syncthreads();
//         } else {
//         }

//         for (int i = 0; i < 4; i++)
//             for (int j = 0; j < 4; j++)
//                 for (int ki = 0; ki < 3; ki++)
//                     for (int kj = 0; kj < 3; kj++) {
//                         temp[i][j] +=
//                             kernel[ki][kj] * shared_x[i + ui + ki][j + uj + kj];
//                     }
//     }
//     if (write_to_global) {
//         for (int i = 0; i < 4; i++)
//             for (int j = 0; j < 4; j++)
//                 output[oc * 4096 + (y_start + i + ui) * 64
//                        + (x_start + j + uj)] = temp[i][j];
//     } else {
//         for (int i = 0; i < 4; i++)
//             for (int j = 0; j < 4; j++)
//                 shared[oc * 101 + (i + ui + 1) * 10 + (j + uj + 1)] =
//                     temp[i][j];
//         __syncthreads();
//         partly_write_back(x, shared);
//     }
// }

__global__ void conv_fused_kernel(float *__restrict__ x, float *__restrict__ w,
                                  float *__restrict__ output) {
    __shared__ float shared[64 * 65];
    conv_device_kernel_shared(x, w, output, shared, true, false);
    for (int i = 0; i < 8; i++) {
        conv_device_kernel_shared(x, w, output, shared, false, false);
    }
    conv_device_kernel_shared(x, w, output, shared, false, true);
}

__global__ void conv_fused_common_kernel(float *__restrict__ x,
                                         float *__restrict__ w,
                                         float *__restrict__ output) {
    for (int i = 0; i < 5; i++) {
        conv_device_kernel(x, w, output);
        Barrier();
        conv_device_kernel(output, w, x);
        Barrier();
    }
}

extern "C" {

void conv_fused(float *x, float *w, float *output) {
    // default_function_kernel1<<<dim3(8, 8, 1), dim3(4, 4, 8)>>>(x, w, output);
    conv_fused_kernel<<<dim3(8, 8, 1), dim3(4, 4, 8)>>>(x, w, output);
}

void conv_fused_common(float *x, float *w, float *output) {
    // default_function_kernel1<<<dim3(8, 8, 1), dim3(4, 4, 8)>>>(x, w, output);
    conv_fused_common_kernel<<<dim3(8, 8, 1), dim3(4, 4, 8)>>>(x, w, output);
}
}
