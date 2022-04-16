//1_3_512_21_1_6
//128_1008_21_21_168_1_1_SAME
//dim3 grid(1, 3, 512);
//dim3 block(21, 1, 6);

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
extern "C" __global__ void __launch_bounds__(126) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[49];
  __shared__ float pad_temp_shared[882];
  __shared__ float placeholder_shared[252];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(35)] = 0.000000e+00f;
  compute_local[(42)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(36)] = 0.000000e+00f;
  compute_local[(43)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(37)] = 0.000000e+00f;
  compute_local[(44)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  compute_local[(38)] = 0.000000e+00f;
  compute_local[(45)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(32)] = 0.000000e+00f;
  compute_local[(39)] = 0.000000e+00f;
  compute_local[(46)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(33)] = 0.000000e+00f;
  compute_local[(40)] = 0.000000e+00f;
  compute_local[(47)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(34)] = 0.000000e+00f;
  compute_local[(41)] = 0.000000e+00f;
  compute_local[(48)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 168; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 147) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((((((int)blockIdx.z) >> 2) * 444528) + (rc_outer * 2646)) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + (((int)threadIdx.x) * 7)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      placeholder_shared[((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((((int)blockIdx.z) & 3) * 42336) + (((int)threadIdx.z) * 7056)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 6) * 1008)) + (rc_outer * 6)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 6)))];
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 6; ++rc_inner) {
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((rc_inner * 147) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 6) + rc_inner))]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((rc_inner * 147) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 36))]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((rc_inner * 147) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 72))]));
      compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((rc_inner * 147) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 108))]));
      compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((rc_inner * 147) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 144))]));
      compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[(((rc_inner * 147) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 180))]));
      compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((rc_inner * 147) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 216))]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 6) + rc_inner))]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 21))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 36))]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 21))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 72))]));
      compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 21))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 108))]));
      compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 21))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 144))]));
      compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 21))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 180))]));
      compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 21))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 216))]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 6) + rc_inner))]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 42))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 36))]));
      compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 42))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 72))]));
      compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 42))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 108))]));
      compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 42))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 144))]));
      compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 42))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 180))]));
      compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 42))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 216))]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 6) + rc_inner))]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 63))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 36))]));
      compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 63))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 72))]));
      compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 63))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 108))]));
      compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 63))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 144))]));
      compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 63))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 180))]));
      compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 63))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 216))]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 6) + rc_inner))]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 36))]));
      compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 72))]));
      compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 108))]));
      compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 144))]));
      compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 180))]));
      compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 216))]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 6) + rc_inner))]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 105))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 36))]));
      compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 105))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 72))]));
      compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 105))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 108))]));
      compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 105))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 144))]));
      compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 105))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 180))]));
      compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 105))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 216))]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 6) + rc_inner))]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 126))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 36))]));
      compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 126))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 72))]));
      compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 126))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 108))]));
      compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 126))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 144))]));
      compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 126))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 180))]));
      compute_local[(48)] = (compute_local[(48)] + (pad_temp_shared[((((rc_inner * 147) + ((int)threadIdx.x)) + 126))] * placeholder_shared[((((((int)threadIdx.z) * 6) + rc_inner) + 216))]));
    }
  }
  compute[(((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 2646))] = compute_local[(7)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 5292))] = compute_local[(14)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 7938))] = compute_local[(21)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 10584))] = compute_local[(28)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 13230))] = compute_local[(35)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 15876))] = compute_local[(42)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 21))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 2667))] = compute_local[(8)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 5313))] = compute_local[(15)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 7959))] = compute_local[(22)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 10605))] = compute_local[(29)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 13251))] = compute_local[(36)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 15897))] = compute_local[(43)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 42))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 2688))] = compute_local[(9)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 5334))] = compute_local[(16)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 7980))] = compute_local[(23)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 10626))] = compute_local[(30)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 13272))] = compute_local[(37)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 15918))] = compute_local[(44)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 63))] = compute_local[(3)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 2709))] = compute_local[(10)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 5355))] = compute_local[(17)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 8001))] = compute_local[(24)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 10647))] = compute_local[(31)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 13293))] = compute_local[(38)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 15939))] = compute_local[(45)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 84))] = compute_local[(4)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 2730))] = compute_local[(11)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 5376))] = compute_local[(18)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 8022))] = compute_local[(25)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 10668))] = compute_local[(32)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 13314))] = compute_local[(39)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 15960))] = compute_local[(46)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 105))] = compute_local[(5)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 2751))] = compute_local[(12)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 5397))] = compute_local[(19)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 8043))] = compute_local[(26)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 10689))] = compute_local[(33)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 13335))] = compute_local[(40)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 15981))] = compute_local[(47)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 126))] = compute_local[(6)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 2772))] = compute_local[(13)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 5418))] = compute_local[(20)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 8064))] = compute_local[(27)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 10710))] = compute_local[(34)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 13356))] = compute_local[(41)];
  compute[((((((((int)blockIdx.z) * 18522) + (((int)threadIdx.z) * 441)) + (((int)blockIdx.y) * 147)) + ((int)threadIdx.x)) + 16002))] = compute_local[(48)];
}

