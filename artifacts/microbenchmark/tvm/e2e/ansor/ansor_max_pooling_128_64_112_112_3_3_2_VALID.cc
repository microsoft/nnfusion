//387200_1_1_64_1_1
//max_128_64_112_112_3_2_VALID
//dim3 grid(387200, 1, 1);
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
  tensor[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))] = max(tensor[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))], data[(((((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) / 3025) * 12544) + (((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 3025) / 55) * 224)) + (rv0 * 112)) + ((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 55) * 2)) + rv1))]);
    }
  }
}

