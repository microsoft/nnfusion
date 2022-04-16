//8192_1_1_128_1_1
//16384_1024_4096
//dim3 grid(8192, 1, 1);
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
  __shared__ float A_shared[1024];
  __shared__ float B_shared[2048];
  for (int x_c_outer_inner_init = 0; x_c_outer_inner_init < 8; ++x_c_outer_inner_init) {
    compute_local[((x_c_outer_inner_init * 2))] = 0.000000e+00f;
    compute_local[(((x_c_outer_inner_init * 2) + 16))] = 0.000000e+00f;
    compute_local[(((x_c_outer_inner_init * 2) + 32))] = 0.000000e+00f;
    compute_local[(((x_c_outer_inner_init * 2) + 48))] = 0.000000e+00f;
    compute_local[(((x_c_outer_inner_init * 2) + 1))] = 0.000000e+00f;
    compute_local[(((x_c_outer_inner_init * 2) + 17))] = 0.000000e+00f;
    compute_local[(((x_c_outer_inner_init * 2) + 33))] = 0.000000e+00f;
    compute_local[(((x_c_outer_inner_init * 2) + 49))] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] = A[((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)))];
    A_shared[((((int)threadIdx.x) + 128))] = A[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 8192))];
    A_shared[((((int)threadIdx.x) + 256))] = A[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 16384))];
    A_shared[((((int)threadIdx.x) + 384))] = A[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 24576))];
    A_shared[((((int)threadIdx.x) + 512))] = A[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 32768))];
    A_shared[((((int)threadIdx.x) + 640))] = A[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 40960))];
    A_shared[((((int)threadIdx.x) + 768))] = A[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 49152))];
    A_shared[((((int)threadIdx.x) + 896))] = A[(((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 57344))];
    ((float2*)(B_shared + ((((int)threadIdx.x) * 2))))[0] = ((float2*)(B + (((((k_outer_outer * 65536) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 63) * 2)))))[0];
    ((float2*)(B_shared + (((((int)threadIdx.x) * 2) + 256))))[0] = ((float2*)(B + ((((((k_outer_outer * 65536) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 63) * 2)) + 8192))))[0];
    ((float2*)(B_shared + (((((int)threadIdx.x) * 2) + 512))))[0] = ((float2*)(B + ((((((k_outer_outer * 65536) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 63) * 2)) + 16384))))[0];
    ((float2*)(B_shared + (((((int)threadIdx.x) * 2) + 768))))[0] = ((float2*)(B + ((((((k_outer_outer * 65536) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 63) * 2)) + 24576))))[0];
    ((float2*)(B_shared + (((((int)threadIdx.x) * 2) + 1024))))[0] = ((float2*)(B + ((((((k_outer_outer * 65536) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 63) * 2)) + 32768))))[0];
    ((float2*)(B_shared + (((((int)threadIdx.x) * 2) + 1280))))[0] = ((float2*)(B + ((((((k_outer_outer * 65536) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 63) * 2)) + 40960))))[0];
    ((float2*)(B_shared + (((((int)threadIdx.x) * 2) + 1536))))[0] = ((float2*)(B + ((((((k_outer_outer * 65536) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 63) * 2)) + 49152))))[0];
    ((float2*)(B_shared + (((((int)threadIdx.x) * 2) + 1792))))[0] = ((float2*)(B + ((((((k_outer_outer * 65536) + ((((int)threadIdx.x) >> 6) * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 63) * 2)) + 57344))))[0];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int x_c_outer_inner = 0; x_c_outer_inner < 8; ++x_c_outer_inner) {
        for (int y_c_outer_inner = 0; y_c_outer_inner < 2; ++y_c_outer_inner) {
          compute_local[(((x_c_outer_inner * 2) + y_c_outer_inner))] = (compute_local[(((x_c_outer_inner * 2) + y_c_outer_inner))] + (A_shared[(((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)))] * B_shared[((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 16))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 16))] + (A_shared[(((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 64))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 32))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 32))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 512))] * B_shared[((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 48))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 48))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 512))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 64))]));
          compute_local[(((x_c_outer_inner * 2) + y_c_outer_inner))] = (compute_local[(((x_c_outer_inner * 2) + y_c_outer_inner))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 1))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 128))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 16))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 16))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 1))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 192))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 32))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 32))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 513))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 128))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 48))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 48))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 513))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 192))]));
          compute_local[(((x_c_outer_inner * 2) + y_c_outer_inner))] = (compute_local[(((x_c_outer_inner * 2) + y_c_outer_inner))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 2))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 256))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 16))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 16))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 2))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 320))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 32))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 32))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 514))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 256))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 48))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 48))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 514))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 320))]));
          compute_local[(((x_c_outer_inner * 2) + y_c_outer_inner))] = (compute_local[(((x_c_outer_inner * 2) + y_c_outer_inner))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 3))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 384))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 16))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 16))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 3))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 448))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 32))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 32))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 515))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 384))]));
          compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 48))] = (compute_local[((((x_c_outer_inner * 2) + y_c_outer_inner) + 48))] + (A_shared[((((((((int)threadIdx.x) >> 5) * 128) + (x_c_outer_inner * 16)) + (k_outer_inner * 4)) + 515))] * B_shared[(((((k_outer_inner * 512) + ((((int)threadIdx.x) & 31) * 2)) + y_c_outer_inner) + 448))]));
        }
      }
    }
  }
  for (int x_inner = 0; x_inner < 8; ++x_inner) {
    for (int y_inner = 0; y_inner < 2; ++y_inner) {
      compute[((((((((((int)blockIdx.x) >> 5) * 262144) + ((((int)threadIdx.x) >> 5) * 32768)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + y_inner))] = compute_local[(((x_inner * 2) + y_inner))];
      compute[(((((((((((int)blockIdx.x) >> 5) * 262144) + ((((int)threadIdx.x) >> 5) * 32768)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + y_inner) + 64))] = compute_local[((((x_inner * 2) + y_inner) + 16))];
      compute[(((((((((((int)blockIdx.x) >> 5) * 262144) + ((((int)threadIdx.x) >> 5) * 32768)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + y_inner) + 131072))] = compute_local[((((x_inner * 2) + y_inner) + 32))];
      compute[(((((((((((int)blockIdx.x) >> 5) * 262144) + ((((int)threadIdx.x) >> 5) * 32768)) + (x_inner * 4096)) + ((((int)blockIdx.x) & 31) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + y_inner) + 131136))] = compute_local[((((x_inner * 2) + y_inner) + 48))];
    }
  }
}

