#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__device__ void pointwise_conv_device_kernel(float* __restrict__ X, float* __restrict__ K, float* __restrict__ Y) {
  float Y_local[32];
  __shared__ float X_shared[64];
  __shared__ float K_shared[64];
  float X_shared_local[4];
  float K_shared_local[8];
  for (int c_c_init = 0; c_c_init < 8; ++c_c_init) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        Y_local[((((c_c_init * 4) + (i_c_init * 2)) + j_c_init))] = 0.000000e+00f;
      }
    }
  }
  for (int ric_outer = 0; ric_outer < 64; ++ric_outer) {
    __syncthreads();
    if (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) >> 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) < 64) {
        if (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) < 8) {
          if (((int)threadIdx.x) < 2) {
            X_shared[((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = X[(((((((ric_outer * 4096) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) < 64) {
      if (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) < 8) {
        if (((int)threadIdx.x) < 2) {
          K_shared[((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = K[(((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 64)) + ric_outer))];
        }
      }
    }
    __syncthreads();
    for (int ax1 = 0; ax1 < 2; ++ax1) {
      for (int ax2 = 0; ax2 < 2; ++ax2) {
        X_shared_local[(((ax1 * 2) + ax2))] = X_shared[(((((((int)threadIdx.y) * 16) + (ax1 * 8)) + (((int)threadIdx.x) * 2)) + ax2))];
      }
    }
    for (int ax0 = 0; ax0 < 8; ++ax0) {
      K_shared_local[(ax0)] = K_shared[(((((int)threadIdx.z) * 8) + ax0))];
    }
    for (int c_c = 0; c_c < 8; ++c_c) {
      for (int i_c = 0; i_c < 2; ++i_c) {
        for (int j_c = 0; j_c < 2; ++j_c) {
          Y_local[((((c_c * 4) + (i_c * 2)) + j_c))] = (Y_local[((((c_c * 4) + (i_c * 2)) + j_c))] + (X_shared_local[(((i_c * 2) + j_c))] * K_shared_local[(c_c)]));
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

__device__ void pointwise_conv_device_kernel_shared(float* __restrict__ X, float* __restrict__ K, float* __restrict__ Y, float* shared, bool read_from_global, bool write_to_global) {
  float Y_local[32];
  __shared__ float X_shared[64];
  __shared__ float K_shared[64];
  float X_shared_local[4];
  float K_shared_local[8];
  for (int c_c_init = 0; c_c_init < 8; ++c_c_init) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        Y_local[((((c_c_init * 4) + (i_c_init * 2)) + j_c_init))] = 0.000000e+00f;
      }
    }
  }
  for (int ric_outer = 0; ric_outer < 64; ++ric_outer) {
    __syncthreads();
    if (read_from_global) {
    if (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) >> 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) < 64) {
        if (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) < 8) {
          if (((int)threadIdx.x) < 2) {
            X_shared[((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = X[(((((((ric_outer * 4096) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.z) * 64)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))];
          }
        }
      }
    }
    } else {
    if (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) >> 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) < 64) {
        if (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) < 8) {
          if (((int)threadIdx.x) < 2) {
            int id = (((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2) + ((int)threadIdx.x);
            X_shared[id] = shared[ric_outer * 64 + id];
          }
        }
      }
    }
    }
    if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) < 64) {
      if (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) < 8) {
        if (((int)threadIdx.x) < 2) {
          K_shared[((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = K[(((((((int)threadIdx.z) * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 64)) + ric_outer))];
        }
      }
    }
    __syncthreads();
    for (int ax1 = 0; ax1 < 2; ++ax1) {
      for (int ax2 = 0; ax2 < 2; ++ax2) {
        X_shared_local[(((ax1 * 2) + ax2))] = X_shared[(((((((int)threadIdx.y) * 16) + (ax1 * 8)) + (((int)threadIdx.x) * 2)) + ax2))];
      }
    }
    for (int ax0 = 0; ax0 < 8; ++ax0) {
      K_shared_local[(ax0)] = K_shared[(((((int)threadIdx.z) * 8) + ax0))];
    }
    for (int c_c = 0; c_c < 8; ++c_c) {
      for (int i_c = 0; i_c < 2; ++i_c) {
        for (int j_c = 0; j_c < 2; ++j_c) {
          Y_local[((((c_c * 4) + (i_c * 2)) + j_c))] = (Y_local[((((c_c * 4) + (i_c * 2)) + j_c))] + (X_shared_local[(((i_c * 2) + j_c))] * K_shared_local[(c_c)]));
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
        shared[(threadIdx.z*8+i)*64 + (threadIdx.y*2+j)*8+(threadIdx.x*2+k)] = Y_local[((((i * 4) + (j * 2)) + k))];
      }
    }
  }
  }
}

__global__ void pointwise_conv_fused_kernel(float *__restrict__ x, float *__restrict__ w,
                                  float *__restrict__ output) {
    __shared__ float shared[64 * 64];
    pointwise_conv_device_kernel_shared(x, w, output, shared, true, false);
    for (int i = 0; i < 8; i++) {
        pointwise_conv_device_kernel_shared(x, w, output, shared, false, false);
    }
    pointwise_conv_device_kernel_shared(x, w, output, shared, false, true);
}

__global__ void pointwise_conv_fused_common_kernel(float *__restrict__ x,
                                         float *__restrict__ w,
                                         float *__restrict__ output) {
    for (int i = 0; i < 5; i++) {
        pointwise_conv_device_kernel(x, w, output);
        __syncthreads();
        pointwise_conv_device_kernel(output, w, x);
        __syncthreads();
    }
}

extern "C" {

void pointwise_conv_fused(float *x, float *w, float *output) {
    pointwise_conv_fused_kernel<<<dim3(8, 8, 1), dim3(4, 4, 8)>>>(x, w, output);
}

void pointwise_conv_fused_common(float *x, float *w, float *output) {
    pointwise_conv_fused_common_kernel<<<dim3(8, 8, 1), dim3(4, 4, 8)>>>(x, w, output);
}
}
