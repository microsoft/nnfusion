
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[32];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[4096];
  float A_shared_local[4];
  float B_shared_local[8];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] = A[(((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)))];
    A_shared[((((int)threadIdx.x) + 256))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048))];
    A_shared[((((int)threadIdx.x) + 512))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096))];
    A_shared[((((int)threadIdx.x) + 768))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144))];
    A_shared[((((int)threadIdx.x) + 1024))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192))];
    A_shared[((((int)threadIdx.x) + 1280))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 10240))];
    A_shared[((((int)threadIdx.x) + 1536))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288))];
    A_shared[((((int)threadIdx.x) + 1792))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336))];
    B_shared[(((int)threadIdx.x))] = B[(((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[((((int)threadIdx.x) + 256))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 2048))];
    B_shared[((((int)threadIdx.x) + 512))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 4096))];
    B_shared[((((int)threadIdx.x) + 768))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 6144))];
    B_shared[((((int)threadIdx.x) + 1024))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 65536))];
    B_shared[((((int)threadIdx.x) + 1280))] = B[((((((((((int)threadIdx.x) + 1280) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 8) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[((((int)threadIdx.x) + 1536))] = B[((((((((((int)threadIdx.x) + 1536) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 16) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[((((int)threadIdx.x) + 1792))] = B[((((((((((int)threadIdx.x) + 1792) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 24) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[((((int)threadIdx.x) + 2048))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 131072))];
    B_shared[((((int)threadIdx.x) + 2304))] = B[((((((((((int)threadIdx.x) + 2304) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 8) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[((((int)threadIdx.x) + 2560))] = B[((((((((((int)threadIdx.x) + 2560) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 16) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[((((int)threadIdx.x) + 2816))] = B[((((((((((int)threadIdx.x) + 2816) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 24) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[((((int)threadIdx.x) + 3072))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 196608))];
    B_shared[((((int)threadIdx.x) + 3328))] = B[((((((((((int)threadIdx.x) + 3328) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 8) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[((((int)threadIdx.x) + 3584))] = B[((((((((((int)threadIdx.x) + 3584) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 16) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[((((int)threadIdx.x) + 3840))] = B[((((((((((int)threadIdx.x) + 3840) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 24) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
      A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer))];
      A_shared_local[(1)] = A_shared[(((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer) + 512))];
      A_shared_local[(2)] = A_shared[(((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer) + 1024))];
      A_shared_local[(3)] = A_shared[(((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer) + 1536))];
      B_shared_local[(0)] = B_shared[(((k_inner_outer * 32) + (((int)threadIdx.x) & 15)))];
      B_shared_local[(2)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 1024))];
      B_shared_local[(4)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 2048))];
      B_shared_local[(6)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 3072))];
      B_shared_local[(1)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 16))];
      B_shared_local[(3)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 1040))];
      B_shared_local[(5)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 2064))];
      B_shared_local[(7)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 3088))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(8)] = (compute_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
      compute_local[(16)] = (compute_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
      compute_local[(24)] = (compute_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (A_shared_local[(1)] * B_shared_local[(0)]));
      compute_local[(10)] = (compute_local[(10)] + (A_shared_local[(1)] * B_shared_local[(2)]));
      compute_local[(18)] = (compute_local[(18)] + (A_shared_local[(1)] * B_shared_local[(4)]));
      compute_local[(26)] = (compute_local[(26)] + (A_shared_local[(1)] * B_shared_local[(6)]));
      compute_local[(4)] = (compute_local[(4)] + (A_shared_local[(2)] * B_shared_local[(0)]));
      compute_local[(12)] = (compute_local[(12)] + (A_shared_local[(2)] * B_shared_local[(2)]));
      compute_local[(20)] = (compute_local[(20)] + (A_shared_local[(2)] * B_shared_local[(4)]));
      compute_local[(28)] = (compute_local[(28)] + (A_shared_local[(2)] * B_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (A_shared_local[(3)] * B_shared_local[(0)]));
      compute_local[(14)] = (compute_local[(14)] + (A_shared_local[(3)] * B_shared_local[(2)]));
      compute_local[(22)] = (compute_local[(22)] + (A_shared_local[(3)] * B_shared_local[(4)]));
      compute_local[(30)] = (compute_local[(30)] + (A_shared_local[(3)] * B_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
      compute_local[(9)] = (compute_local[(9)] + (A_shared_local[(0)] * B_shared_local[(3)]));
      compute_local[(17)] = (compute_local[(17)] + (A_shared_local[(0)] * B_shared_local[(5)]));
      compute_local[(25)] = (compute_local[(25)] + (A_shared_local[(0)] * B_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (A_shared_local[(1)] * B_shared_local[(1)]));
      compute_local[(11)] = (compute_local[(11)] + (A_shared_local[(1)] * B_shared_local[(3)]));
      compute_local[(19)] = (compute_local[(19)] + (A_shared_local[(1)] * B_shared_local[(5)]));
      compute_local[(27)] = (compute_local[(27)] + (A_shared_local[(1)] * B_shared_local[(7)]));
      compute_local[(5)] = (compute_local[(5)] + (A_shared_local[(2)] * B_shared_local[(1)]));
      compute_local[(13)] = (compute_local[(13)] + (A_shared_local[(2)] * B_shared_local[(3)]));
      compute_local[(21)] = (compute_local[(21)] + (A_shared_local[(2)] * B_shared_local[(5)]));
      compute_local[(29)] = (compute_local[(29)] + (A_shared_local[(2)] * B_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
      compute_local[(15)] = (compute_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
      compute_local[(23)] = (compute_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
      compute_local[(31)] = (compute_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
    }
  }
  compute[(((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)))] = compute_local[(0)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 16384))] = compute_local[(8)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 32768))] = compute_local[(16)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 49152))] = compute_local[(24)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 4096))] = compute_local[(2)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 20480))] = compute_local[(10)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 36864))] = compute_local[(18)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 53248))] = compute_local[(26)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 8192))] = compute_local[(4)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 24576))] = compute_local[(12)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 40960))] = compute_local[(20)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 57344))] = compute_local[(28)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 12288))] = compute_local[(6)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 28672))] = compute_local[(14)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 45056))] = compute_local[(22)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 61440))] = compute_local[(30)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 16))] = compute_local[(1)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 16400))] = compute_local[(9)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 32784))] = compute_local[(17)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 49168))] = compute_local[(25)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 4112))] = compute_local[(3)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 20496))] = compute_local[(11)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 36880))] = compute_local[(19)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 53264))] = compute_local[(27)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 8208))] = compute_local[(5)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 24592))] = compute_local[(13)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 40976))] = compute_local[(21)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 57360))] = compute_local[(29)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 12304))] = compute_local[(7)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 28688))] = compute_local[(15)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 45072))] = compute_local[(23)];
  compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 61456))] = compute_local[(31)];
}

dim3 grid(8, 1, 1);
dim3 block(256, 1, 1);
