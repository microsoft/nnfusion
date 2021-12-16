
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float compute_local[32];
  __shared__ float compute_shared[4096];
  __shared__ float compute_d_shared[2048];
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
  for (int k_outer = 0; k_outer < 11; ++k_outer) {
    __syncthreads();
    compute_shared[(((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)))];
    compute_shared[((((int)threadIdx.x) + 256))] = data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 882))];
    compute_shared[((((int)threadIdx.x) + 512))] = data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 1764))];
    compute_shared[((((int)threadIdx.x) + 768))] = data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 2646))];
    compute_shared[((((int)threadIdx.x) + 1024))] = data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 3528))];
    compute_shared[((((int)threadIdx.x) + 1280))] = data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 4410))];
    compute_shared[((((int)threadIdx.x) + 1536))] = data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 5292))];
    compute_shared[((((int)threadIdx.x) + 1792))] = data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 6174))];
    compute_shared[((((int)threadIdx.x) + 2048))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 320) ? data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 7056))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 2304))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 318) ? data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 7938))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 2560))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 316) ? data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 8820))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 2816))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 314) ? data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 9702))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 3072))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 312) ? data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 10584))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 3328))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 310) ? data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 11466))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 3584))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 308) ? data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 12348))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 3840))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 306) ? data[((((((((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) / 441) * 148176) + (k_outer * 14112)) + ((((int)threadIdx.x) >> 7) * 441)) + ((((((int)blockIdx.x) % 441) * 128) + (((int)threadIdx.x) & 127)) % 441)) + 13230))] : 0.000000e+00f);
    compute_d_shared[(((int)threadIdx.x))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 336) ? kernel[((((((((int)blockIdx.x) / 441) * 21504) + ((((int)threadIdx.x) >> 5) * 336)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 256))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 336) ? kernel[(((((((((int)blockIdx.x) / 441) * 21504) + ((((int)threadIdx.x) >> 5) * 336)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 2688))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 512))] = ((((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 320) && (((k_outer * 32) + (((int)threadIdx.x) & 31)) < 336)) ? kernel[(((((((((int)blockIdx.x) / 441) * 21504) + ((((int)threadIdx.x) >> 5) * 336)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 5376))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 768))] = ((((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 312) && (((k_outer * 32) + (((int)threadIdx.x) & 31)) < 336)) ? kernel[(((((((((int)blockIdx.x) / 441) * 21504) + ((((int)threadIdx.x) >> 5) * 336)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 8064))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1024))] = ((((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 304) && (((k_outer * 32) + (((int)threadIdx.x) & 31)) < 336)) ? kernel[(((((((((int)blockIdx.x) / 441) * 21504) + ((((int)threadIdx.x) >> 5) * 336)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 10752))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1280))] = ((((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 296) && (((k_outer * 32) + (((int)threadIdx.x) & 31)) < 336)) ? kernel[(((((((((int)blockIdx.x) / 441) * 21504) + ((((int)threadIdx.x) >> 5) * 336)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 13440))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1536))] = ((((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 288) && (((k_outer * 32) + (((int)threadIdx.x) & 31)) < 336)) ? kernel[(((((((((int)blockIdx.x) / 441) * 21504) + ((((int)threadIdx.x) >> 5) * 336)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 16128))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1792))] = ((((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 280) && (((k_outer * 32) + (((int)threadIdx.x) & 31)) < 336)) ? kernel[(((((((((int)blockIdx.x) / 441) * 21504) + ((((int)threadIdx.x) >> 5) * 336)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 18816))] : 0.000000e+00f);
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
      compute_shared_local[(0)] = compute_shared[(((k_inner_outer * 128) + (((int)threadIdx.x) & 31)))];
      compute_shared_local[(1)] = compute_shared[((((k_inner_outer * 128) + (((int)threadIdx.x) & 31)) + 32))];
      compute_shared_local[(2)] = compute_shared[((((k_inner_outer * 128) + (((int)threadIdx.x) & 31)) + 64))];
      compute_shared_local[(3)] = compute_shared[((((k_inner_outer * 128) + (((int)threadIdx.x) & 31)) + 96))];
      compute_d_shared_local[(0)] = compute_d_shared[((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer))];
      compute_d_shared_local[(1)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 256))];
      compute_d_shared_local[(2)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 512))];
      compute_d_shared_local[(3)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 768))];
      compute_d_shared_local[(4)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 1024))];
      compute_d_shared_local[(5)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 1280))];
      compute_d_shared_local[(6)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 1536))];
      compute_d_shared_local[(7)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 1792))];
      if (((k_outer * 32) + k_inner_outer) < 336) {
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
  compute[((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)))] = (compute_local[(0)] + bias[((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)))]);
  compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 32))] = (compute_local[(1)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 32))]);
  compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 64))] = (compute_local[(2)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 64))]);
  compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 96))] = (compute_local[(3)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 96))]);
  compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 451584))] = (compute_local[(4)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 451584))]);
  compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 451616))] = (compute_local[(5)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 451616))]);
  compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 451648))] = (compute_local[(6)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 451648))]);
  compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 451680))] = (compute_local[(7)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 451680))]);
  if ((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 320) {
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 903168))] = (compute_local[(8)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 903168))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 903200))] = (compute_local[(9)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 903200))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 903232))] = (compute_local[(10)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 903232))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 903264))] = (compute_local[(11)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 903264))]);
  }
  if ((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 312) {
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1354752))] = (compute_local[(12)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1354752))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1354784))] = (compute_local[(13)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1354784))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1354816))] = (compute_local[(14)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1354816))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1354848))] = (compute_local[(15)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1354848))]);
  }
  if ((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 304) {
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1806336))] = (compute_local[(16)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1806336))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1806368))] = (compute_local[(17)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1806368))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1806400))] = (compute_local[(18)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1806400))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1806432))] = (compute_local[(19)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 1806432))]);
  }
  if ((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 296) {
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2257920))] = (compute_local[(20)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2257920))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2257952))] = (compute_local[(21)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2257952))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2257984))] = (compute_local[(22)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2257984))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2258016))] = (compute_local[(23)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2258016))]);
  }
  if ((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 288) {
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2709504))] = (compute_local[(24)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2709504))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2709536))] = (compute_local[(25)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2709536))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2709568))] = (compute_local[(26)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2709568))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2709600))] = (compute_local[(27)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 2709600))]);
  }
  if ((((((int)blockIdx.x) / 441) * 64) + (((int)threadIdx.x) >> 5)) < 280) {
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 3161088))] = (compute_local[(28)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 3161088))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 3161120))] = (compute_local[(29)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 3161120))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 3161152))] = (compute_local[(30)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 3161152))]);
    compute[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 3161184))] = (compute_local[(31)] + bias[(((((((((int)blockIdx.x) / 441) * 3612672) + ((((int)threadIdx.x) >> 5) * 56448)) + ((((int)blockIdx.x) % 441) * 128)) + (((int)threadIdx.x) & 31)) + 3161184))]);
  }
}

dim3 grid(2646, 1, 1);
dim3 block(256, 1, 1);
