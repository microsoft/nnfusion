#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void __launch_bounds__(128) pointwise_tvm(float* __restrict__ X, float* __restrict__ K, float* __restrict__ Y) {
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

extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ X, float* __restrict__ K, float* __restrict__ Y) {
  float Y_local[8];
  __shared__ float X_shared[16];
  __shared__ float K_shared[64];
  float X_shared_local[1];
  float K_shared_local[8];
  for (int c_c_init = 0; c_c_init < 8; ++c_c_init) {
    Y_local[(c_c_init)] = 0.000000e+00f;
  }
  for (int ric_outer = 0; ric_outer < 64; ++ric_outer) {
    __syncthreads();
    if ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 16) {
      if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 2) {
        if (((int)threadIdx.x) < 1) {
          X_shared[((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = X[((((((ric_outer * 4096) + (((int)blockIdx.y) * 256)) + (((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)) >> 2) * 64)) + (((int)blockIdx.x) * 4)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) + ((int)threadIdx.y)) & 3)))];
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
    X_shared_local[(0)] = X_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)))];
    for (int ax0 = 0; ax0 < 8; ++ax0) {
      K_shared_local[(ax0)] = K_shared[(((((int)threadIdx.z) * 8) + ax0))];
    }
    for (int c_c = 0; c_c < 8; ++c_c) {
      Y_local[(c_c)] = (Y_local[(c_c)] + (X_shared_local[(0)] * K_shared_local[(c_c)]));
    }
  }
  for (int c_inner_inner = 0; c_inner_inner < 8; ++c_inner_inner) {
    Y[(((((((((int)threadIdx.z) * 32768) + (c_inner_inner * 4096)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 64)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = Y_local[(c_inner_inner)];
  }
}

extern "C" {

void pointwise_conv(float *x, float *w, float *output) {
    // pointwise_tvm<<<dim3(8, 8, 1), dim3(4, 4, 8)>>>(x, w, output);
    default_function_kernel0<<<dim3(16, 16, 1), dim3(4, 4, 8)>>>(x, w, output);
}

}
