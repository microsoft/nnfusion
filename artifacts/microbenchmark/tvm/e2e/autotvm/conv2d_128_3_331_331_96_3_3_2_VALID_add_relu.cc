//1_165_512_15_1_4
//128_3_331_331_96_3_2_VALID
//dim3 grid(1, 165, 512);
//dim3 block(15, 1, 4);

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
extern "C" __global__ void __launch_bounds__(60) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[66];
  __shared__ float pad_temp_shared[331];
  __shared__ float placeholder_shared[72];
  for (int ff_init = 0; ff_init < 6; ++ff_init) {
    compute1[(ff_init)] = 0.000000e+00f;
    compute1[((ff_init + 6))] = 0.000000e+00f;
    compute1[((ff_init + 12))] = 0.000000e+00f;
    compute1[((ff_init + 18))] = 0.000000e+00f;
    compute1[((ff_init + 24))] = 0.000000e+00f;
    compute1[((ff_init + 30))] = 0.000000e+00f;
    compute1[((ff_init + 36))] = 0.000000e+00f;
    compute1[((ff_init + 42))] = 0.000000e+00f;
    compute1[((ff_init + 48))] = 0.000000e+00f;
    compute1[((ff_init + 54))] = 0.000000e+00f;
    compute1[((ff_init + 60))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
        if ((((((int)threadIdx.z) * 83) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 331) {
          if (((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 83) {
            pad_temp_shared[((((((int)threadIdx.z) * 83) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((((((int)blockIdx.z) >> 2) * 328683) + (rc_outer * 109561)) + (((int)blockIdx.y) * 662)) + (ry_outer * 331)) + (((int)threadIdx.z) * 83)) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
          }
        }
      }
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
        if (((((int)threadIdx.z) * 6) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 3)) < 24) {
          if ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 72) {
            if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 18) {
              placeholder_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((((int)blockIdx.z) & 3) * 648) + (((int)threadIdx.z) * 162)) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 3) * 27)) + (rc_outer * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 3)))];
            }
          }
        }
      }
      __syncthreads();
      for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
        for (int ff = 0; ff < 6; ++ff) {
          compute1[(ff)] = (compute1[(ff)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 6))] = (compute1[((ff + 6))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 30))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 12))] = (compute1[((ff + 12))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 60))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 18))] = (compute1[((ff + 18))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 90))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 24))] = (compute1[((ff + 24))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 120))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 30))] = (compute1[((ff + 30))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 150))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 36))] = (compute1[((ff + 36))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 180))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 42))] = (compute1[((ff + 42))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 210))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 48))] = (compute1[((ff + 48))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 240))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 54))] = (compute1[((ff + 54))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 270))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
          compute1[((ff + 60))] = (compute1[((ff + 60))] + (pad_temp_shared[((((((int)threadIdx.x) * 2) + rx_inner) + 300))] * placeholder_shared[((((((int)threadIdx.z) * 18) + (ff * 3)) + rx_inner))]));
        }
      }
    }
  }
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 6; ++i1_inner_inner_inner) {
    compute[((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)))] = max((compute1[(i1_inner_inner_inner)] + input2[((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 15))] = max((compute1[((i1_inner_inner_inner + 6))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 15))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 30))] = max((compute1[((i1_inner_inner_inner + 12))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 30))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 45))] = max((compute1[((i1_inner_inner_inner + 18))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 45))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 60))] = max((compute1[((i1_inner_inner_inner + 24))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 60))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 75))] = max((compute1[((i1_inner_inner_inner + 30))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 75))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 90))] = max((compute1[((i1_inner_inner_inner + 36))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 90))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 105))] = max((compute1[((i1_inner_inner_inner + 42))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 105))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 120))] = max((compute1[((i1_inner_inner_inner + 48))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 120))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 135))] = max((compute1[((i1_inner_inner_inner + 54))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 135))]), 0.000000e+00f);
    compute[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 150))] = max((compute1[((i1_inner_inner_inner + 60))] + input2[(((((((((int)blockIdx.z) * 653400) + (((int)threadIdx.z) * 163350)) + (i1_inner_inner_inner * 27225)) + (((int)blockIdx.y) * 165)) + ((int)threadIdx.x)) + 150))]), 0.000000e+00f);
  }
}

