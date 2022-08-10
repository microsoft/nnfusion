
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
  float compute_local[64];
  __shared__ float compute_shared[4096];
  __shared__ float compute_d_shared[6144];
  float compute_shared_local[4];
  float compute_d_shared_local[16];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(32)] = 0.000000e+00f;
  compute_local[(36)] = 0.000000e+00f;
  compute_local[(40)] = 0.000000e+00f;
  compute_local[(44)] = 0.000000e+00f;
  compute_local[(48)] = 0.000000e+00f;
  compute_local[(52)] = 0.000000e+00f;
  compute_local[(56)] = 0.000000e+00f;
  compute_local[(60)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(33)] = 0.000000e+00f;
  compute_local[(37)] = 0.000000e+00f;
  compute_local[(41)] = 0.000000e+00f;
  compute_local[(45)] = 0.000000e+00f;
  compute_local[(49)] = 0.000000e+00f;
  compute_local[(53)] = 0.000000e+00f;
  compute_local[(57)] = 0.000000e+00f;
  compute_local[(61)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(34)] = 0.000000e+00f;
  compute_local[(38)] = 0.000000e+00f;
  compute_local[(42)] = 0.000000e+00f;
  compute_local[(46)] = 0.000000e+00f;
  compute_local[(50)] = 0.000000e+00f;
  compute_local[(54)] = 0.000000e+00f;
  compute_local[(58)] = 0.000000e+00f;
  compute_local[(62)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  compute_local[(35)] = 0.000000e+00f;
  compute_local[(39)] = 0.000000e+00f;
  compute_local[(43)] = 0.000000e+00f;
  compute_local[(47)] = 0.000000e+00f;
  compute_local[(51)] = 0.000000e+00f;
  compute_local[(55)] = 0.000000e+00f;
  compute_local[(59)] = 0.000000e+00f;
  compute_local[(63)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 6; ++k_outer) {
    __syncthreads();
    compute_shared[(((int)threadIdx.x))] = data[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)))];
    compute_shared[((((int)threadIdx.x) + 384))] = data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 5292))];
    compute_shared[((((int)threadIdx.x) + 768))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 162) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 10584))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 1152))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 159) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 15876))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 1536))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 156) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 21168))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 1920))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 153) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 26460))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 2304))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 150) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 31752))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 2688))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 147) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 37044))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 3072))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 144) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 42336))] : 0.000000e+00f);
    compute_shared[((((int)threadIdx.x) + 3456))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 141) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 47628))] : 0.000000e+00f);
    if (((int)threadIdx.x) < 256) {
      compute_shared[((((int)threadIdx.x) + 3840))] = ((((k_outer * 32) + (((int)threadIdx.x) >> 7)) < 138) ? data[(((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) / 1764) * 296352) + (k_outer * 56448)) + ((((int)threadIdx.x) >> 7) * 1764)) + (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) & 127)) % 1764)) + 52920))] : 0.000000e+00f);
    }
    compute_d_shared[(((int)threadIdx.x))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[(((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 384))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 2016))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 768))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 4032))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1152))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 6048))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1536))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 8064))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 1920))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 10080))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 2304))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 12096))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 2688))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 14112))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 3072))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 16128))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 3456))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 18144))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 3840))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 20160))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 4224))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 22176))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 4608))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 24192))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 4992))] = ((((k_outer * 32) + (((int)threadIdx.x) & 31)) < 168) ? kernel[((((((((int)threadIdx.x) >> 5) * 168) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 26208))] : 0.000000e+00f);
    compute_d_shared[((((int)threadIdx.x) + 5376))] = 0.000000e+00f;
    compute_d_shared[((((int)threadIdx.x) + 5760))] = 0.000000e+00f;
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
      compute_d_shared_local[(8)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 3072))];
      compute_d_shared_local[(9)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 3456))];
      compute_d_shared_local[(10)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 3840))];
      compute_d_shared_local[(11)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 4224))];
      compute_d_shared_local[(12)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 4608))];
      compute_d_shared_local[(13)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 4992))];
      compute_d_shared_local[(14)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 5376))];
      compute_d_shared_local[(15)] = compute_d_shared[(((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer) + 5760))];
      if (((k_outer * 32) + k_inner_outer) < 168) {
        compute_local[(0)] = (compute_local[(0)] + (compute_shared_local[(0)] * compute_d_shared_local[(0)]));
        compute_local[(4)] = (compute_local[(4)] + (compute_shared_local[(0)] * compute_d_shared_local[(1)]));
        compute_local[(8)] = (compute_local[(8)] + (compute_shared_local[(0)] * compute_d_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (compute_shared_local[(0)] * compute_d_shared_local[(3)]));
        compute_local[(16)] = (compute_local[(16)] + (compute_shared_local[(0)] * compute_d_shared_local[(4)]));
        compute_local[(20)] = (compute_local[(20)] + (compute_shared_local[(0)] * compute_d_shared_local[(5)]));
        compute_local[(24)] = (compute_local[(24)] + (compute_shared_local[(0)] * compute_d_shared_local[(6)]));
        compute_local[(28)] = (compute_local[(28)] + (compute_shared_local[(0)] * compute_d_shared_local[(7)]));
        compute_local[(32)] = (compute_local[(32)] + (compute_shared_local[(0)] * compute_d_shared_local[(8)]));
        compute_local[(36)] = (compute_local[(36)] + (compute_shared_local[(0)] * compute_d_shared_local[(9)]));
        compute_local[(40)] = (compute_local[(40)] + (compute_shared_local[(0)] * compute_d_shared_local[(10)]));
        compute_local[(44)] = (compute_local[(44)] + (compute_shared_local[(0)] * compute_d_shared_local[(11)]));
        compute_local[(48)] = (compute_local[(48)] + (compute_shared_local[(0)] * compute_d_shared_local[(12)]));
        compute_local[(52)] = (compute_local[(52)] + (compute_shared_local[(0)] * compute_d_shared_local[(13)]));
        compute_local[(56)] = (compute_local[(56)] + (compute_shared_local[(0)] * compute_d_shared_local[(14)]));
        compute_local[(60)] = (compute_local[(60)] + (compute_shared_local[(0)] * compute_d_shared_local[(15)]));
        compute_local[(1)] = (compute_local[(1)] + (compute_shared_local[(1)] * compute_d_shared_local[(0)]));
        compute_local[(5)] = (compute_local[(5)] + (compute_shared_local[(1)] * compute_d_shared_local[(1)]));
        compute_local[(9)] = (compute_local[(9)] + (compute_shared_local[(1)] * compute_d_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (compute_shared_local[(1)] * compute_d_shared_local[(3)]));
        compute_local[(17)] = (compute_local[(17)] + (compute_shared_local[(1)] * compute_d_shared_local[(4)]));
        compute_local[(21)] = (compute_local[(21)] + (compute_shared_local[(1)] * compute_d_shared_local[(5)]));
        compute_local[(25)] = (compute_local[(25)] + (compute_shared_local[(1)] * compute_d_shared_local[(6)]));
        compute_local[(29)] = (compute_local[(29)] + (compute_shared_local[(1)] * compute_d_shared_local[(7)]));
        compute_local[(33)] = (compute_local[(33)] + (compute_shared_local[(1)] * compute_d_shared_local[(8)]));
        compute_local[(37)] = (compute_local[(37)] + (compute_shared_local[(1)] * compute_d_shared_local[(9)]));
        compute_local[(41)] = (compute_local[(41)] + (compute_shared_local[(1)] * compute_d_shared_local[(10)]));
        compute_local[(45)] = (compute_local[(45)] + (compute_shared_local[(1)] * compute_d_shared_local[(11)]));
        compute_local[(49)] = (compute_local[(49)] + (compute_shared_local[(1)] * compute_d_shared_local[(12)]));
        compute_local[(53)] = (compute_local[(53)] + (compute_shared_local[(1)] * compute_d_shared_local[(13)]));
        compute_local[(57)] = (compute_local[(57)] + (compute_shared_local[(1)] * compute_d_shared_local[(14)]));
        compute_local[(61)] = (compute_local[(61)] + (compute_shared_local[(1)] * compute_d_shared_local[(15)]));
        compute_local[(2)] = (compute_local[(2)] + (compute_shared_local[(2)] * compute_d_shared_local[(0)]));
        compute_local[(6)] = (compute_local[(6)] + (compute_shared_local[(2)] * compute_d_shared_local[(1)]));
        compute_local[(10)] = (compute_local[(10)] + (compute_shared_local[(2)] * compute_d_shared_local[(2)]));
        compute_local[(14)] = (compute_local[(14)] + (compute_shared_local[(2)] * compute_d_shared_local[(3)]));
        compute_local[(18)] = (compute_local[(18)] + (compute_shared_local[(2)] * compute_d_shared_local[(4)]));
        compute_local[(22)] = (compute_local[(22)] + (compute_shared_local[(2)] * compute_d_shared_local[(5)]));
        compute_local[(26)] = (compute_local[(26)] + (compute_shared_local[(2)] * compute_d_shared_local[(6)]));
        compute_local[(30)] = (compute_local[(30)] + (compute_shared_local[(2)] * compute_d_shared_local[(7)]));
        compute_local[(34)] = (compute_local[(34)] + (compute_shared_local[(2)] * compute_d_shared_local[(8)]));
        compute_local[(38)] = (compute_local[(38)] + (compute_shared_local[(2)] * compute_d_shared_local[(9)]));
        compute_local[(42)] = (compute_local[(42)] + (compute_shared_local[(2)] * compute_d_shared_local[(10)]));
        compute_local[(46)] = (compute_local[(46)] + (compute_shared_local[(2)] * compute_d_shared_local[(11)]));
        compute_local[(50)] = (compute_local[(50)] + (compute_shared_local[(2)] * compute_d_shared_local[(12)]));
        compute_local[(54)] = (compute_local[(54)] + (compute_shared_local[(2)] * compute_d_shared_local[(13)]));
        compute_local[(58)] = (compute_local[(58)] + (compute_shared_local[(2)] * compute_d_shared_local[(14)]));
        compute_local[(62)] = (compute_local[(62)] + (compute_shared_local[(2)] * compute_d_shared_local[(15)]));
        compute_local[(3)] = (compute_local[(3)] + (compute_shared_local[(3)] * compute_d_shared_local[(0)]));
        compute_local[(7)] = (compute_local[(7)] + (compute_shared_local[(3)] * compute_d_shared_local[(1)]));
        compute_local[(11)] = (compute_local[(11)] + (compute_shared_local[(3)] * compute_d_shared_local[(2)]));
        compute_local[(15)] = (compute_local[(15)] + (compute_shared_local[(3)] * compute_d_shared_local[(3)]));
        compute_local[(19)] = (compute_local[(19)] + (compute_shared_local[(3)] * compute_d_shared_local[(4)]));
        compute_local[(23)] = (compute_local[(23)] + (compute_shared_local[(3)] * compute_d_shared_local[(5)]));
        compute_local[(27)] = (compute_local[(27)] + (compute_shared_local[(3)] * compute_d_shared_local[(6)]));
        compute_local[(31)] = (compute_local[(31)] + (compute_shared_local[(3)] * compute_d_shared_local[(7)]));
        compute_local[(35)] = (compute_local[(35)] + (compute_shared_local[(3)] * compute_d_shared_local[(8)]));
        compute_local[(39)] = (compute_local[(39)] + (compute_shared_local[(3)] * compute_d_shared_local[(9)]));
        compute_local[(43)] = (compute_local[(43)] + (compute_shared_local[(3)] * compute_d_shared_local[(10)]));
        compute_local[(47)] = (compute_local[(47)] + (compute_shared_local[(3)] * compute_d_shared_local[(11)]));
        compute_local[(51)] = (compute_local[(51)] + (compute_shared_local[(3)] * compute_d_shared_local[(12)]));
        compute_local[(55)] = (compute_local[(55)] + (compute_shared_local[(3)] * compute_d_shared_local[(13)]));
        compute_local[(59)] = (compute_local[(59)] + (compute_shared_local[(3)] * compute_d_shared_local[(14)]));
        compute_local[(63)] = (compute_local[(63)] + (compute_shared_local[(3)] * compute_d_shared_local[(15)]));
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
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 18966528))] = (compute_local[(28)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 18966528))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 18966560))] = (compute_local[(29)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 18966560))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 18966592))] = (compute_local[(30)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 18966592))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 18966624))] = (compute_local[(31)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 18966624))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 21676032))] = (compute_local[(32)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 21676032))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 21676064))] = (compute_local[(33)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 21676064))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 21676096))] = (compute_local[(34)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 21676096))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 21676128))] = (compute_local[(35)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 21676128))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 24385536))] = (compute_local[(36)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 24385536))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 24385568))] = (compute_local[(37)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 24385568))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 24385600))] = (compute_local[(38)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 24385600))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 24385632))] = (compute_local[(39)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 24385632))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 27095040))] = (compute_local[(40)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 27095040))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 27095072))] = (compute_local[(41)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 27095072))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 27095104))] = (compute_local[(42)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 27095104))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 27095136))] = (compute_local[(43)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 27095136))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 29804544))] = (compute_local[(44)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 29804544))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 29804576))] = (compute_local[(45)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 29804576))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 29804608))] = (compute_local[(46)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 29804608))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 29804640))] = (compute_local[(47)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 29804640))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32514048))] = (compute_local[(48)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32514048))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32514080))] = (compute_local[(49)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32514080))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32514112))] = (compute_local[(50)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32514112))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32514144))] = (compute_local[(51)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 32514144))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 35223552))] = (compute_local[(52)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 35223552))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 35223584))] = (compute_local[(53)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 35223584))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 35223616))] = (compute_local[(54)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 35223616))]);
  compute[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 35223648))] = (compute_local[(55)] + bias[((((((((int)threadIdx.x) >> 5) * 225792) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) & 31)) + 35223648))]);
}

dim3 grid(1764, 1, 1);
dim3 block(384, 1, 1);