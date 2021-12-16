
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
  for (int k_outer = 0; k_outer < 3; ++k_outer) {
    __syncthreads();
    compute_shared[(((int)threadIdx.x))] = data[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)))];
    compute_shared[((((int)threadIdx.x) + 384))] = data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 5292))];
    compute_shared[((((int)threadIdx.x) + 768))] = data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 10584))];
    compute_shared[((((int)threadIdx.x) + 1152))] = data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 15876))];
    compute_shared[((((int)threadIdx.x) + 1536))] = data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 21168))];
    compute_shared[((((int)threadIdx.x) + 1920))] = data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 26460))];
    compute_shared[((((int)threadIdx.x) + 2304))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 66) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 31752))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 2688))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 63) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 37044))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 3072))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 60) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 42336))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 3456))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 57) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 47628))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 256) {
      compute_shared[((((int)threadIdx.x) + 3840))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 54) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 148176) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 52920))] : 0.000000e+00f);
    }
    compute_d_shared[(((int)threadIdx.x))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 84) ? kernel[(((((((int)threadIdx.x) >> 5) * 84) + (k_outer * 32)) + (((int)threadIdx.x) & 31)))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 384))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 84) ? kernel[((((((((int)threadIdx.x) >> 5) * 84) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 1008))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 768))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 84) ? kernel[((((((((int)threadIdx.x) >> 5) * 84) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 2016))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1152))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 84) ? kernel[((((((((int)threadIdx.x) >> 5) * 84) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 3024))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1536))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 84) ? kernel[((((((((int)threadIdx.x) >> 5) * 84) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 4032))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1920))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 84) ? kernel[((((((((int)threadIdx.x) >> 5) * 84) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 5040))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 2304))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 84) ? kernel[((((((((int)threadIdx.x) >> 5) * 84) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 6048))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 2688))] = 0.000000e+00f;
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
      if (((k_outer * 32) + k_inner_outer) < 84) {
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
  }
  compute[(((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)))] = (compute_local[(0)] + bias[(((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32))] = (compute_local[(1)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 64))] = (compute_local[(2)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 64))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 96))] = (compute_local[(3)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 96))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 2709504))] = (compute_local[(4)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 2709504))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 2709536))] = (compute_local[(5)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 2709536))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 2709568))] = (compute_local[(6)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 2709568))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 2709600))] = (compute_local[(7)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 2709600))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 5419008))] = (compute_local[(8)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 5419008))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 5419040))] = (compute_local[(9)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 5419040))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 5419072))] = (compute_local[(10)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 5419072))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 5419104))] = (compute_local[(11)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 5419104))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 8128512))] = (compute_local[(12)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 8128512))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 8128544))] = (compute_local[(13)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 8128544))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 8128576))] = (compute_local[(14)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 8128576))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 8128608))] = (compute_local[(15)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 8128608))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 10838016))] = (compute_local[(16)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 10838016))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 10838048))] = (compute_local[(17)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 10838048))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 10838080))] = (compute_local[(18)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 10838080))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 10838112))] = (compute_local[(19)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 10838112))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 13547520))] = (compute_local[(20)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 13547520))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 13547552))] = (compute_local[(21)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 13547552))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 13547584))] = (compute_local[(22)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 13547584))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 13547616))] = (compute_local[(23)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 13547616))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 16257024))] = (compute_local[(24)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 16257024))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 16257056))] = (compute_local[(25)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 16257056))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 16257088))] = (compute_local[(26)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 16257088))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 16257120))] = (compute_local[(27)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 16257120))]);
}

dim3 grid(1764, 1, 1);
dim3 block(384, 1, 1);
