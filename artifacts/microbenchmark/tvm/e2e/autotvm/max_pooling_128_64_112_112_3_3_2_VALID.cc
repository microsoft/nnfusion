//24200_1_1_1024_1_1
//max_128_64_112_112_3_2_VALID
//dim3 grid(24200, 1, 1);
//dim3 block(1024, 1, 1);

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
extern "C" __global__ void __launch_bounds__(1024) default_function_kernel0(float* __restrict__ data, float* __restrict__ tensor) {
  float tensor_local[1];
  tensor_local[(0)] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor_local[(0)] = max(tensor_local[(0)], data[(((((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 3025) * 12544) + (((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 3025) / 55) * 224)) + (rv0 * 112)) + ((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 55) * 2)) + rv1))]);
    }
  }
  tensor[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = tensor_local[(0)];
}

