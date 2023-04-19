
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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ bias, float* __restrict__ conv_unpad) {
  float conv_local[2];
  __shared__ float data_pad_shared[2048];
  __shared__ float kernel_pad_shared[2048];
  float data_pad_shared_local[2];
  float kernel_pad_shared_local[1];
  conv_local[(0)] = 0.000000e+00f;
  conv_local[(1)] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 5; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[(((int)threadIdx.x))] = (((((0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 9) / 3))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)))) && (((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) < 9)) ? data[((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 128))] = (((((0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 8) % 9) / 3))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 8) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 8) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 8) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 256))] = (((((0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 7) % 9) / 3))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 7) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 16) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 7) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 384))] = (((((0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 6) % 9) / 3))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 6) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)))) && (((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 24) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 6) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 512))] = (((((0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 5) % 9) / 3))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 5) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 32) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 5) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 640))] = (((((0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 4) % 9) / 3))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 4) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 40) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 4) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 768))] = (((((0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 3) % 9) / 3))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 3) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)))) && (((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 48) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 3) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 896))] = (((((0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 9) / 3))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 56) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1024))] = ((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) < 512) && (0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 9) / 3)))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 64) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1152))] = ((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) < 504) && (0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 9) / 3)))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)))) && (((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) < 9)) ? data[((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) + 503))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1280))] = ((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) < 496) && (0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 8) % 9) / 3)))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 8) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 80) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 8) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1408))] = ((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) < 488) && (0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 7) % 9) / 3)))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 7) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 88) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 7) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1536))] = ((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) < 480) && (0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 6) % 9) / 3)))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 6) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)))) && (((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 96) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 6) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1664))] = ((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) < 472) && (0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 5) % 9) / 3)))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 5) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 104) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 5) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 2) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1792))] = ((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) < 464) && (0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 4) % 9) / 3)))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 4) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)))) && (((((int)threadIdx.x) & 7) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 112) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 4) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + ((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 1) % 3)) - 9))] : 0.000000e+00f);
    data_pad_shared[((((int)threadIdx.x) + 1920))] = ((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) < 456) && (0 < ((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 3) % 9) / 3)))) && (((((((int)blockIdx.x) & 3) * 2) + ((((int)threadIdx.x) & 15) >> 3)) + (((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 3) % 9) / 3)) < 9)) && (0 < ((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)))) && (((((int)threadIdx.x) & 7) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) < 9)) ? data[(((((((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 120) / 9) * 64) + ((((int)blockIdx.x) & 3) * 16)) + ((((((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) + 3) % 9) / 3) * 8)) + (((int)threadIdx.x) & 15)) + (((ra_fused0_outer * 128) + (((int)threadIdx.x) >> 4)) % 3)) - 9))] : 0.000000e+00f);
    kernel_pad_shared[(((int)threadIdx.x))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[(((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 128))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 576))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 256))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 1152))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 384))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 1728))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 512))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 2304))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 640))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 2880))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 768))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 3456))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 896))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 4032))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1024))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 4608))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1152))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 5184))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1280))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 5760))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1408))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 6336))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1536))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 6912))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1664))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 7488))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1792))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 8064))] : 0.000000e+00f);
    kernel_pad_shared[((((int)threadIdx.x) + 1920))] = ((((ra_fused0_outer * 128) + ((int)threadIdx.x)) < 576) ? kernel[((((((((int)blockIdx.x) >> 2) * 9216) + (ra_fused0_outer * 128)) + ((int)threadIdx.x)) + 8640))] : 0.000000e+00f);
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 128; ++ra_fused0_inner_outer) {
      data_pad_shared_local[(0)] = data_pad_shared[(((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 7)))];
      data_pad_shared_local[(1)] = data_pad_shared[((((ra_fused0_inner_outer * 16) + (((int)threadIdx.x) & 7)) + 8))];
      kernel_pad_shared_local[(0)] = kernel_pad_shared[((((((int)threadIdx.x) >> 3) * 128) + ra_fused0_inner_outer))];
      if (((ra_fused0_outer * 128) + ra_fused0_inner_outer) < 576) {
        conv_local[(0)] = (conv_local[(0)] + (data_pad_shared_local[(0)] * kernel_pad_shared_local[(0)]));
        conv_local[(1)] = (conv_local[(1)] + (data_pad_shared_local[(1)] * kernel_pad_shared_local[(0)]));
      }
    }
  }
  conv_unpad[((((((((int)blockIdx.x) >> 2) * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 7)))] = max((conv_local[(0)] + bias[((((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3)))]), 0.000000e+00f);
  conv_unpad[(((((((((int)blockIdx.x) >> 2) * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 7)) + 8))] = max((conv_local[(1)] + bias[((((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3)))]), 0.000000e+00f);
}

dim3 grid(16, 1, 1);
dim3 block(128, 1, 1);
best_idx 12