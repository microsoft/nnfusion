
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
extern "C" __global__ void __launch_bounds__(64) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[1];
  __shared__ float A_shared[4096];
  __shared__ float B_shared[128];
  compute_local[(0)] = 0.000000e+00f;
  ((float4*)(A_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(A + (((((((int)blockIdx.x) >> 1) * 8192) + ((((int)blockIdx.x) & 1) * 2048)) + (((int)threadIdx.x) * 4)))))[0];
  ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) >> 1) * 8192) + ((((int)blockIdx.x) & 1) * 2048)) + (((int)threadIdx.x) * 4)) + 256))))[0];
  ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) >> 1) * 8192) + ((((int)blockIdx.x) & 1) * 2048)) + (((int)threadIdx.x) * 4)) + 512))))[0];
  ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) >> 1) * 8192) + ((((int)blockIdx.x) & 1) * 2048)) + (((int)threadIdx.x) * 4)) + 768))))[0];
  ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) >> 1) * 8192) + ((((int)blockIdx.x) & 1) * 2048)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
  ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 1280))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) >> 1) * 8192) + ((((int)blockIdx.x) & 1) * 2048)) + (((int)threadIdx.x) * 4)) + 1280))))[0];
  ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 1536))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) >> 1) * 8192) + ((((int)blockIdx.x) & 1) * 2048)) + (((int)threadIdx.x) * 4)) + 1536))))[0];
  ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 1792))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) >> 1) * 8192) + ((((int)blockIdx.x) & 1) * 2048)) + (((int)threadIdx.x) * 4)) + 1792))))[0];
  ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 2048))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) >> 1) * 8192) + ((((int)blockIdx.x) & 1) * 2048)) + (((int)threadIdx.x) * 4)) + 4096))))[0];
  ((float4*)(A_shared + (((((((((int)threadIdx.x) * 4) + 2304) >> 11) * 2048) + (((((int)threadIdx.x) >> 4) + 4) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(A + (((((((((int)blockIdx.x) >> 1) * 8192) + ((((((int)threadIdx.x) * 4) + 2304) >> 11) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + (((((int)threadIdx.x) >> 4) + 4) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
  ((float4*)(A_shared + (((((((((int)threadIdx.x) * 4) + 2560) >> 11) * 2048) + (((((int)threadIdx.x) >> 4) + 8) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(A + (((((((((int)blockIdx.x) >> 1) * 8192) + ((((((int)threadIdx.x) * 4) + 2560) >> 11) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + (((((int)threadIdx.x) >> 4) + 8) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
  ((float4*)(A_shared + (((((((((int)threadIdx.x) * 4) + 2816) >> 11) * 2048) + (((((int)threadIdx.x) >> 4) + 12) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(A + (((((((((int)blockIdx.x) >> 1) * 8192) + ((((((int)threadIdx.x) * 4) + 2816) >> 11) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + (((((int)threadIdx.x) >> 4) + 12) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
  ((float4*)(A_shared + (((((((((int)threadIdx.x) * 4) + 3072) >> 11) * 2048) + (((((int)threadIdx.x) >> 4) + 16) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(A + (((((((((int)blockIdx.x) >> 1) * 8192) + ((((((int)threadIdx.x) * 4) + 3072) >> 11) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + (((((int)threadIdx.x) >> 4) + 16) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
  ((float4*)(A_shared + (((((((((int)threadIdx.x) * 4) + 3328) >> 11) * 2048) + (((((int)threadIdx.x) >> 4) + 20) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(A + (((((((((int)blockIdx.x) >> 1) * 8192) + ((((((int)threadIdx.x) * 4) + 3328) >> 11) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + (((((int)threadIdx.x) >> 4) + 20) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
  ((float4*)(A_shared + (((((((((int)threadIdx.x) * 4) + 3584) >> 11) * 2048) + (((((int)threadIdx.x) >> 4) + 24) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(A + (((((((((int)blockIdx.x) >> 1) * 8192) + ((((((int)threadIdx.x) * 4) + 3584) >> 11) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + (((((int)threadIdx.x) >> 4) + 24) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
  ((float4*)(A_shared + (((((((((int)threadIdx.x) * 4) + 3840) >> 11) * 2048) + (((((int)threadIdx.x) >> 4) + 28) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(A + (((((((((int)blockIdx.x) >> 1) * 8192) + ((((((int)threadIdx.x) * 4) + 3840) >> 11) * 4096)) + ((((int)blockIdx.x) & 1) * 2048)) + (((((int)threadIdx.x) >> 4) + 28) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
  B_shared[(((int)threadIdx.x))] = B[((((((int)blockIdx.x) >> 1) * 128) + ((int)threadIdx.x)))];
  B_shared[((((int)threadIdx.x) + 64))] = B[(((((((int)blockIdx.x) >> 1) * 128) + ((int)threadIdx.x)) + 64))];
  __syncthreads();
  compute_local[(0)] = (compute_local[(0)] + (A_shared[((((int)threadIdx.x) * 64))] * B_shared[(((((int)threadIdx.x) >> 5) * 64))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 1))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 2))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 2))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 3))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 3))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 4))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 5))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 6))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 6))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 7))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 7))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 8))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 9))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 10))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 10))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 11))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 11))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 12))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 13))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 14))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 14))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 15))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 15))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 16))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 17))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 18))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 18))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 19))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 19))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 20))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 21))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 22))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 22))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 23))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 23))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 24))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 25))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 26))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 26))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 27))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 27))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 28))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 29))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 30))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 30))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 31))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 31))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 32))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 33))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 34))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 34))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 35))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 35))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 36))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 37))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 38))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 38))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 39))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 39))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 40))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 41))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 42))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 42))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 43))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 43))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 44))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 45))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 46))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 46))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 47))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 47))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 48))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 49))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 50))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 50))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 51))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 51))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 52))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 53))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 54))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 54))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 55))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 55))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 56))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 57))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 58))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 58))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 59))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 59))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 60))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 61))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 62))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 62))]));
  compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 64) + 63))] * B_shared[((((((int)threadIdx.x) >> 5) * 64) + 63))]));
  compute[((((((((int)blockIdx.x) >> 1) * 128) + ((((int)threadIdx.x) >> 5) * 64)) + ((((int)blockIdx.x) & 1) * 32)) + (((int)threadIdx.x) & 31)))] = compute_local[(0)];
}

dim3 grid(12, 1, 1);
dim3 block(64, 1, 1);
