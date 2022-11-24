
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
extern "C" __global__ void __launch_bounds__(384) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float compute_local[32];
  __shared__ float compute_shared[4096];
  __shared__ float compute_d_shared[3072];
  float compute_shared_local[4];
  float compute_d_shared_local[8];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 21; ++k_outer) {
    __syncthreads();
    compute_shared[(((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)))];
    compute_shared[((((int)threadIdx.x) + 384))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 363))];
    compute_shared[((((int)threadIdx.x) + 768))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 726))];
    compute_shared[((((int)threadIdx.x) + 1152))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 1089))];
    compute_shared[((((int)threadIdx.x) + 1536))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 1452))];
    compute_shared[((((int)threadIdx.x) + 1920))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 1815))];
    compute_shared[((((int)threadIdx.x) + 2304))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 2178))];
    compute_shared[((((int)threadIdx.x) + 2688))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 2541))];
    compute_shared[((((int)threadIdx.x) + 3072))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 2904))];
    compute_shared[((((int)threadIdx.x) + 3456))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 3267))];
    if (((int)threadIdx.x) < 256) {
      compute_shared[((((int)threadIdx.x) + 3840))] = data[((((((((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) / 121) * 81312) + (k_outer * 3872)) + ((((int)threadIdx.x) >> 7) * 121)) + ((((((int)blockIdx.x) % 121) * 128) + (((int)threadIdx.x) & 127)) % 121)) + 3630))];
    }
    compute_d_shared[(((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 121) * 64512) + ((((int)threadIdx.x) >> 5) * 672)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)))];
    compute_d_shared[((((int)threadIdx.x) + 384))] = kernel[(((((((((int)blockIdx.x) / 121) * 64512) + ((((int)threadIdx.x) >> 5) * 672)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 8064))];
    compute_d_shared[((((int)threadIdx.x) + 768))] = kernel[(((((((((int)blockIdx.x) / 121) * 64512) + ((((int)threadIdx.x) >> 5) * 672)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 16128))];
    compute_d_shared[((((int)threadIdx.x) + 1152))] = kernel[(((((((((int)blockIdx.x) / 121) * 64512) + ((((int)threadIdx.x) >> 5) * 672)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 24192))];
    compute_d_shared[((((int)threadIdx.x) + 1536))] = kernel[(((((((((int)blockIdx.x) / 121) * 64512) + ((((int)threadIdx.x) >> 5) * 672)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 32256))];
    compute_d_shared[((((int)threadIdx.x) + 1920))] = kernel[(((((((((int)blockIdx.x) / 121) * 64512) + ((((int)threadIdx.x) >> 5) * 672)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 40320))];
    compute_d_shared[((((int)threadIdx.x) + 2304))] = kernel[(((((((((int)blockIdx.x) / 121) * 64512) + ((((int)threadIdx.x) >> 5) * 672)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 48384))];
    compute_d_shared[((((int)threadIdx.x) + 2688))] = kernel[(((((((((int)blockIdx.x) / 121) * 64512) + ((((int)threadIdx.x) >> 5) * 672)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 56448))];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
      compute_shared_local[(0)] = compute_shared[(((k_inner_outer * 128) + (((int)threadIdx.x) & 31)))];
      compute_shared_local[(1)] = compute_shared[((((k_inner_outer * 128) + (((int)threadIdx.x) & 31)) + 32))];
      compute_shared_local[(2)] = compute_shared[((((k_inner_outer * 128) + (((int)threadIdx.x) & 31)) + 64))];
      compute_shared_local[(3)] = compute_shared[((((k_inner_outer * 128) + (((int)threadIdx.x) & 31)) + 96))];
      compute_d_shared_local[(0)] = compute_d_shared[((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer))];
      compute_d_shared_local[(1)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 384))];
      compute_d_shared_local[(2)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 768))];
      compute_d_shared_local[(3)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 1152))];
      compute_d_shared_local[(4)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 1536))];
      compute_d_shared_local[(5)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 1920))];
      compute_d_shared_local[(6)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 2304))];
      compute_d_shared_local[(7)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 2688))];
      compute_local[(0)] = (compute_local[(0)] + (compute_shared_local[(0)] * compute_d_shared_local[(0)]));
      compute_local[(4)] = (compute_local[(4)] + (compute_shared_local[(0)] * compute_d_shared_local[(1)]));
      compute_local[(8)] = (compute_local[(8)] + (compute_shared_local[(0)] * compute_d_shared_local[(2)]));
      compute_local[(12)] = (compute_local[(12)] + (compute_shared_local[(0)] * compute_d_shared_local[(3)]));
      compute_local[(16)] = (compute_local[(16)] + (compute_shared_local[(0)] * compute_d_shared_local[(4)]));
      compute_local[(20)] = (compute_local[(20)] + (compute_shared_local[(0)] * compute_d_shared_local[(5)]));
      compute_local[(24)] = (compute_local[(24)] + (compute_shared_local[(0)] * compute_d_shared_local[(6)]));
      compute_local[(28)] = (compute_local[(28)] + (compute_shared_local[(0)] * compute_d_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (compute_shared_local[(1)] * compute_d_shared_local[(0)]));
      compute_local[(5)] = (compute_local[(5)] + (compute_shared_local[(1)] * compute_d_shared_local[(1)]));
      compute_local[(9)] = (compute_local[(9)] + (compute_shared_local[(1)] * compute_d_shared_local[(2)]));
      compute_local[(13)] = (compute_local[(13)] + (compute_shared_local[(1)] * compute_d_shared_local[(3)]));
      compute_local[(17)] = (compute_local[(17)] + (compute_shared_local[(1)] * compute_d_shared_local[(4)]));
      compute_local[(21)] = (compute_local[(21)] + (compute_shared_local[(1)] * compute_d_shared_local[(5)]));
      compute_local[(25)] = (compute_local[(25)] + (compute_shared_local[(1)] * compute_d_shared_local[(6)]));
      compute_local[(29)] = (compute_local[(29)] + (compute_shared_local[(1)] * compute_d_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (compute_shared_local[(2)] * compute_d_shared_local[(0)]));
      compute_local[(6)] = (compute_local[(6)] + (compute_shared_local[(2)] * compute_d_shared_local[(1)]));
      compute_local[(10)] = (compute_local[(10)] + (compute_shared_local[(2)] * compute_d_shared_local[(2)]));
      compute_local[(14)] = (compute_local[(14)] + (compute_shared_local[(2)] * compute_d_shared_local[(3)]));
      compute_local[(18)] = (compute_local[(18)] + (compute_shared_local[(2)] * compute_d_shared_local[(4)]));
      compute_local[(22)] = (compute_local[(22)] + (compute_shared_local[(2)] * compute_d_shared_local[(5)]));
      compute_local[(26)] = (compute_local[(26)] + (compute_shared_local[(2)] * compute_d_shared_local[(6)]));
      compute_local[(30)] = (compute_local[(30)] + (compute_shared_local[(2)] * compute_d_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (compute_shared_local[(3)] * compute_d_shared_local[(0)]));
      compute_local[(7)] = (compute_local[(7)] + (compute_shared_local[(3)] * compute_d_shared_local[(1)]));
      compute_local[(11)] = (compute_local[(11)] + (compute_shared_local[(3)] * compute_d_shared_local[(2)]));
      compute_local[(15)] = (compute_local[(15)] + (compute_shared_local[(3)] * compute_d_shared_local[(3)]));
      compute_local[(19)] = (compute_local[(19)] + (compute_shared_local[(3)] * compute_d_shared_local[(4)]));
      compute_local[(23)] = (compute_local[(23)] + (compute_shared_local[(3)] * compute_d_shared_local[(5)]));
      compute_local[(27)] = (compute_local[(27)] + (compute_shared_local[(3)] * compute_d_shared_local[(6)]));
      compute_local[(31)] = (compute_local[(31)] + (compute_shared_local[(3)] * compute_d_shared_local[(7)]));
    }
  }
  compute[((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)))] = (compute_local[(0)] + bias[((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 185856))] = (compute_local[(4)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 185856))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 371712))] = (compute_local[(8)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 371712))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 557568))] = (compute_local[(12)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 557568))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 743424))] = (compute_local[(16)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 743424))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 929280))] = (compute_local[(20)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 929280))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1115136))] = (compute_local[(24)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1115136))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1300992))] = (compute_local[(28)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1300992))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 32))] = (compute_local[(1)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 32))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 185888))] = (compute_local[(5)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 185888))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 371744))] = (compute_local[(9)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 371744))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 557600))] = (compute_local[(13)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 557600))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 743456))] = (compute_local[(17)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 743456))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 929312))] = (compute_local[(21)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 929312))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1115168))] = (compute_local[(25)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1115168))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1301024))] = (compute_local[(29)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1301024))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 64))] = (compute_local[(2)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 64))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 185920))] = (compute_local[(6)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 185920))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 371776))] = (compute_local[(10)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 371776))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 557632))] = (compute_local[(14)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 557632))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 743488))] = (compute_local[(18)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 743488))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 929344))] = (compute_local[(22)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 929344))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1115200))] = (compute_local[(26)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1115200))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1301056))] = (compute_local[(30)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1301056))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 96))] = (compute_local[(3)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 96))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 185952))] = (compute_local[(7)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 185952))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 371808))] = (compute_local[(11)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 371808))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 557664))] = (compute_local[(15)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 557664))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 743520))] = (compute_local[(19)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 743520))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 929376))] = (compute_local[(23)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 929376))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1115232))] = (compute_local[(27)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1115232))]);
  compute[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1301088))] = (compute_local[(31)] + bias[(((((((((int)blockIdx.x) / 121) * 1486848) + ((((int)threadIdx.x) >> 5) * 15488)) + ((((int)blockIdx.x) % 121) * 128)) + (((int)threadIdx.x) & 31)) + 1301088))]);
}

dim3 grid(847, 1, 1);
dim3 block(384, 1, 1);
