//592704_1_1_64_1_1
//avg_128_168_83_83_1_2_VALID
//dim3 grid(592704, 1, 1);
//dim3 block(64, 1, 1);

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

extern "C" __global__ void __launch_bounds__(64) default_function_kernel0(float* __restrict__ tensor, float* __restrict__ data) {
  tensor[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))] = 0.000000e+00f;
  tensor[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))] = (tensor[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))] + data[(((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) / 1764) * 6889) + (((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 1764) / 42) * 166)) + ((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 42) * 2)))]);
}

