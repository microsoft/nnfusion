// %%%

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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[8];
  __shared__ float A_shared[1024];
  __shared__ float B_shared[256];
  float A_shared_local[64];
  float B_shared_local[8];
  float A_shared_local1[64];
  float B_shared_local1[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    C_local[(i_c_init)] = 0.000000e+00f;
  }
  ((float4*)(A_shared + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 4)))))[0] = ((float4*)(A + ((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)threadIdx.x) & 1) * 4)))))[0];
  if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 3)) < 3797) {
    B_shared[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))] = B[(((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 256)) + (((int)threadIdx.x) & 7)))];
  }
  for (int k_outer_outer = 0; k_outer_outer < 31; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(A_shared + ((((((k_outer_outer + 1) & 1) * 512) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 4)))))[0] = ((float4*)(A + ((((((((int)threadIdx.y) * 2048) + ((((int)threadIdx.x) >> 1) * 256)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8))))[0];
    if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 3)) < 3797) {
      B_shared[((((((k_outer_outer + 1) & 1) * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)))] = B[(((((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 256)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 8))];
    }
    for (int ax0 = 0; ax0 < 8; ++ax0) {
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        A_shared_local[(((ax0 * 4) + ax1))] = A_shared[((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 64)) + (ax0 * 8)) + ax1))];
      }
    }
    for (int ax11 = 0; ax11 < 4; ++ax11) {
      if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 3797) {
        B_shared_local[(ax11)] = B_shared[(((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 8)) + ax11))];
      }
    }
    for (int ax01 = 0; ax01 < 8; ++ax01) {
      for (int ax12 = 0; ax12 < 4; ++ax12) {
        A_shared_local[((((ax01 * 4) + ax12) + 32))] = A_shared[(((((((k_outer_outer & 1) * 512) + (((int)threadIdx.y) * 64)) + (ax01 * 8)) + ax12) + 4))];
      }
    }
    for (int ax13 = 0; ax13 < 4; ++ax13) {
      if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 3797) {
        B_shared_local[((ax13 + 4))] = B_shared[((((((k_outer_outer & 1) * 128) + (((int)threadIdx.x) * 8)) + ax13) + 4))];
      }
    }
    for (int i_c = 0; i_c < 8; ++i_c) {
      for (int k_inner_inner = 0; k_inner_inner < 4; ++k_inner_inner) {
        if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 3797) {
          C_local[(i_c)] = (C_local[(i_c)] + (A_shared_local[(((i_c * 4) + k_inner_inner))] * B_shared_local[(k_inner_inner)]));
        }
      }
    }
    for (int i_c1 = 0; i_c1 < 8; ++i_c1) {
      for (int k_inner_inner1 = 0; k_inner_inner1 < 4; ++k_inner_inner1) {
        if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 3797) {
          C_local[(i_c1)] = (C_local[(i_c1)] + (A_shared_local[((((i_c1 * 4) + k_inner_inner1) + 32))] * B_shared_local[((k_inner_inner1 + 4))]));
        }
      }
    }
  }
  __syncthreads();
  for (int ax02 = 0; ax02 < 8; ++ax02) {
    for (int ax14 = 0; ax14 < 4; ++ax14) {
      A_shared_local1[(((ax02 * 4) + ax14))] = A_shared[(((((((int)threadIdx.y) * 64) + (ax02 * 8)) + ax14) + 512))];
    }
  }
  for (int ax15 = 0; ax15 < 4; ++ax15) {
    if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 3797) {
      B_shared_local1[(ax15)] = B_shared[((((((int)threadIdx.x) * 8) + ax15) + 128))];
    }
  }
  for (int ax03 = 0; ax03 < 8; ++ax03) {
    for (int ax16 = 0; ax16 < 4; ++ax16) {
      A_shared_local1[((((ax03 * 4) + ax16) + 32))] = A_shared[(((((((int)threadIdx.y) * 64) + (ax03 * 8)) + ax16) + 516))];
    }
  }
  for (int ax17 = 0; ax17 < 4; ++ax17) {
    if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 3797) {
      B_shared_local1[((ax17 + 4))] = B_shared[((((((int)threadIdx.x) * 8) + ax17) + 132))];
    }
  }
  for (int i_c2 = 0; i_c2 < 8; ++i_c2) {
    for (int k_inner_inner2 = 0; k_inner_inner2 < 4; ++k_inner_inner2) {
      if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 3797) {
        C_local[(i_c2)] = (C_local[(i_c2)] + (A_shared_local1[(((i_c2 * 4) + k_inner_inner2))] * B_shared_local1[(k_inner_inner2)]));
      }
    }
  }
  for (int i_c3 = 0; i_c3 < 8; ++i_c3) {
    for (int k_inner_inner3 = 0; k_inner_inner3 < 4; ++k_inner_inner3) {
      if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 3797) {
        C_local[(i_c3)] = (C_local[(i_c3)] + (A_shared_local1[((((i_c3 * 4) + k_inner_inner3) + 32))] * B_shared_local1[((k_inner_inner3 + 4))]));
      }
    }
  }
  for (int i_inner_inner_inner = 0; i_inner_inner_inner < 8; ++i_inner_inner_inner) {
    if (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < 3797) {
      C[(((((((int)threadIdx.y) * 30376) + (i_inner_inner_inner * 3797)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)))] = C_local[(i_inner_inner_inner)];
    }
  }
}

// %%%
// +++
dim3 grid(238, 1, 1);
dim3 block(16, 8, 1);
// +++
