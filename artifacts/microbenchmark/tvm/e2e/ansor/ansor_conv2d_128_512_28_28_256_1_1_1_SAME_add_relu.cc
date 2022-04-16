//3584_1_1_112_1_1
//128_512_28_28_256_1_1_SAME
//dim3 grid(3584, 1, 1);
//dim3 block(112, 1, 1);

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
extern "C" __global__ void __launch_bounds__(112) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[64];
  __shared__ float pad_temp_shared[1792];
  __shared__ float input1_shared[1024];
  for (int ff_outer_inner_init = 0; ff_outer_inner_init < 4; ++ff_outer_inner_init) {
    compute1[((ff_outer_inner_init * 16))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 2))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 4))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 6))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 8))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 10))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 12))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 14))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 1))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 3))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 5))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 7))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 9))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 11))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 13))] = 0.000000e+00f;
    compute1[(((ff_outer_inner_init * 16) + 15))] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    ((float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(input0 + (((((((((int)blockIdx.x) / 28) * 401408) + (rc_outer_outer * 12544)) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + ((((int)threadIdx.x) % 28) * 4)))))[0];
    ((float4*)(pad_temp_shared + (((((int)threadIdx.x) * 4) + 448))))[0] = ((float4*)(input0 + ((((((((((int)blockIdx.x) / 28) * 401408) + (rc_outer_outer * 12544)) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + ((((int)threadIdx.x) % 28) * 4)) + 3136))))[0];
    ((float4*)(pad_temp_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(input0 + ((((((((((int)blockIdx.x) / 28) * 401408) + (rc_outer_outer * 12544)) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + ((((int)threadIdx.x) % 28) * 4)) + 6272))))[0];
    ((float4*)(pad_temp_shared + (((((int)threadIdx.x) * 4) + 1344))))[0] = ((float4*)(input0 + ((((((((((int)blockIdx.x) / 28) * 401408) + (rc_outer_outer * 12544)) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) % 7) * 112)) + ((((int)threadIdx.x) % 28) * 4)) + 9408))))[0];
    input1_shared[(((int)threadIdx.x))] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)))];
    input1_shared[((((int)threadIdx.x) + 112))] = input1[((((((((((int)blockIdx.x) % 28) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 3584))];
    input1_shared[((((int)threadIdx.x) + 224))] = input1[((((((((((int)blockIdx.x) % 28) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 7168))];
    input1_shared[((((int)threadIdx.x) + 336))] = input1[((((((((((int)blockIdx.x) % 28) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 10752))];
    input1_shared[((((int)threadIdx.x) + 448))] = input1[((((((((((int)blockIdx.x) % 28) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336))];
    input1_shared[((((int)threadIdx.x) + 560))] = input1[((((((((((int)blockIdx.x) % 28) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 17920))];
    input1_shared[((((int)threadIdx.x) + 672))] = input1[((((((((((int)blockIdx.x) % 28) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 21504))];
    input1_shared[((((int)threadIdx.x) + 784))] = input1[((((((((((int)blockIdx.x) % 28) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 25088))];
    input1_shared[((((int)threadIdx.x) + 896))] = input1[((((((((((int)blockIdx.x) % 28) / 7) * 32768) + ((((int)threadIdx.x) >> 4) * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 28672))];
    if (((int)threadIdx.x) < 16) {
      input1_shared[((((int)threadIdx.x) + 1008))] = input1[(((((((((int)blockIdx.x) % 28) / 7) * 32768) + (rc_outer_outer * 16)) + ((int)threadIdx.x)) + 32256))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 16; ++rc_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 4; ++ff_outer_inner) {
        compute1[((ff_outer_inner * 16))] = (compute1[((ff_outer_inner * 16))] + (pad_temp_shared[(((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)))] * input1_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner))]));
        compute1[(((ff_outer_inner * 16) + 2))] = (compute1[(((ff_outer_inner * 16) + 2))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 28))] * input1_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner))]));
        compute1[(((ff_outer_inner * 16) + 4))] = (compute1[(((ff_outer_inner * 16) + 4))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 56))] * input1_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner))]));
        compute1[(((ff_outer_inner * 16) + 6))] = (compute1[(((ff_outer_inner * 16) + 6))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 84))] * input1_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner))]));
        compute1[(((ff_outer_inner * 16) + 8))] = (compute1[(((ff_outer_inner * 16) + 8))] + (pad_temp_shared[(((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)))] * input1_shared[((((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner) + 16))]));
        compute1[(((ff_outer_inner * 16) + 10))] = (compute1[(((ff_outer_inner * 16) + 10))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 28))] * input1_shared[((((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner) + 16))]));
        compute1[(((ff_outer_inner * 16) + 12))] = (compute1[(((ff_outer_inner * 16) + 12))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 56))] * input1_shared[((((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner) + 16))]));
        compute1[(((ff_outer_inner * 16) + 14))] = (compute1[(((ff_outer_inner * 16) + 14))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 84))] * input1_shared[((((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner) + 16))]));
        compute1[(((ff_outer_inner * 16) + 1))] = (compute1[(((ff_outer_inner * 16) + 1))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 1))] * input1_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner))]));
        compute1[(((ff_outer_inner * 16) + 3))] = (compute1[(((ff_outer_inner * 16) + 3))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 29))] * input1_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner))]));
        compute1[(((ff_outer_inner * 16) + 5))] = (compute1[(((ff_outer_inner * 16) + 5))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 57))] * input1_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner))]));
        compute1[(((ff_outer_inner * 16) + 7))] = (compute1[(((ff_outer_inner * 16) + 7))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 85))] * input1_shared[(((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner))]));
        compute1[(((ff_outer_inner * 16) + 9))] = (compute1[(((ff_outer_inner * 16) + 9))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 1))] * input1_shared[((((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner) + 16))]));
        compute1[(((ff_outer_inner * 16) + 11))] = (compute1[(((ff_outer_inner * 16) + 11))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 29))] * input1_shared[((((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner) + 16))]));
        compute1[(((ff_outer_inner * 16) + 13))] = (compute1[(((ff_outer_inner * 16) + 13))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 57))] * input1_shared[((((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner) + 16))]));
        compute1[(((ff_outer_inner * 16) + 15))] = (compute1[(((ff_outer_inner * 16) + 15))] + (pad_temp_shared[((((rc_outer_inner * 112) + ((((int)threadIdx.x) % 14) * 2)) + 85))] * input1_shared[((((((((int)threadIdx.x) / 14) * 128) + (ff_outer_inner * 32)) + rc_outer_inner) + 16))]));
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 4; ++i2_inner) {
      for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
        compute[(((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 6272)) + (i1_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (i2_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner))] = max((compute1[((((i1_inner * 8) + (i2_inner * 2)) + i3_inner))] + input2[(((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 6272)) + (i1_inner * 784)) + ((((int)blockIdx.x) % 7) * 112)) + (i2_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner))]), 0.000000e+00f);
      }
    }
  }
}

