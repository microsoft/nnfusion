//1792_1_1_448_1_1
//128_512_28_28_128_1_1_SAME
//dim3 grid(1792, 1, 1);
//dim3 block(448, 1, 1);

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
extern "C" __global__ void __launch_bounds__(448) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[16];
  __shared__ float pad_temp_shared[3584];
  __shared__ float input1_shared[2048];
  compute1[(0)] = 0.000000e+00f;
  compute1[(1)] = 0.000000e+00f;
  compute1[(2)] = 0.000000e+00f;
  compute1[(3)] = 0.000000e+00f;
  compute1[(4)] = 0.000000e+00f;
  compute1[(5)] = 0.000000e+00f;
  compute1[(6)] = 0.000000e+00f;
  compute1[(7)] = 0.000000e+00f;
  compute1[(8)] = 0.000000e+00f;
  compute1[(9)] = 0.000000e+00f;
  compute1[(10)] = 0.000000e+00f;
  compute1[(11)] = 0.000000e+00f;
  compute1[(12)] = 0.000000e+00f;
  compute1[(13)] = 0.000000e+00f;
  compute1[(14)] = 0.000000e+00f;
  compute1[(15)] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    ((float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2))))[0] = ((float2*)(input0 + (((((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 25088)) + ((((int)threadIdx.x) / 28) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 28) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)))))[0];
    ((float2*)(pad_temp_shared + (((((int)threadIdx.x) * 2) + 896))))[0] = ((float2*)(input0 + ((((((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 25088)) + ((((int)threadIdx.x) / 28) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 28) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + 12544))))[0];
    ((float2*)(pad_temp_shared + (((((int)threadIdx.x) * 2) + 1792))))[0] = ((float2*)(input0 + ((((((((((((int)blockIdx.x) / 28) * 802816) + (rc_outer_outer * 25088)) + ((((int)threadIdx.x) / 28) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 28) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + 401408))))[0];
    ((float2*)(pad_temp_shared + (((((((((int)threadIdx.x) * 2) + 2688) / 1792) * 1792) + (((((int)threadIdx.x) / 28) + 16) * 56)) + ((((int)threadIdx.x) % 28) * 2)))))[0] = ((float2*)(input0 + ((((((((((((int)blockIdx.x) / 28) * 802816) + ((((((int)threadIdx.x) * 2) + 2688) / 1792) * 401408)) + (rc_outer_outer * 25088)) + (((((int)threadIdx.x) / 28) + 16) * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 28) / 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + ((((int)threadIdx.x) % 7) * 2)))))[0];
    input1_shared[(((int)threadIdx.x))] = input1[(((((((((int)blockIdx.x) % 28) / 14) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)))];
    input1_shared[((((int)threadIdx.x) + 448))] = input1[((((((((((int)blockIdx.x) % 28) / 14) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168))];
    input1_shared[((((int)threadIdx.x) + 896))] = input1[((((((((((int)blockIdx.x) % 28) / 14) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336))];
    input1_shared[((((int)threadIdx.x) + 1344))] = input1[((((((((((int)blockIdx.x) % 28) / 14) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 21504))];
    if (((int)threadIdx.x) < 256) {
      input1_shared[((((int)threadIdx.x) + 1792))] = input1[((((((((((int)blockIdx.x) % 28) / 14) * 32768) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 28672))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 32; ++rc_outer_inner) {
      compute1[(0)] = (compute1[(0)] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)))] * input1_shared[((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner))]));
      compute1[(1)] = (compute1[(1)] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 32))]));
      compute1[(2)] = (compute1[(2)] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 64))]));
      compute1[(3)] = (compute1[(3)] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 96))]));
      compute1[(4)] = (compute1[(4)] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 128))]));
      compute1[(5)] = (compute1[(5)] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 160))]));
      compute1[(6)] = (compute1[(6)] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 192))]));
      compute1[(7)] = (compute1[(7)] + (pad_temp_shared[(((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 224))]));
      compute1[(8)] = (compute1[(8)] + (pad_temp_shared[((((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)) + 1792))] * input1_shared[((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner))]));
      compute1[(9)] = (compute1[(9)] + (pad_temp_shared[((((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)) + 1792))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 32))]));
      compute1[(10)] = (compute1[(10)] + (pad_temp_shared[((((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)) + 1792))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 64))]));
      compute1[(11)] = (compute1[(11)] + (pad_temp_shared[((((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)) + 1792))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 96))]));
      compute1[(12)] = (compute1[(12)] + (pad_temp_shared[((((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)) + 1792))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 128))]));
      compute1[(13)] = (compute1[(13)] + (pad_temp_shared[((((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)) + 1792))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 160))]));
      compute1[(14)] = (compute1[(14)] + (pad_temp_shared[((((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)) + 1792))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 192))]));
      compute1[(15)] = (compute1[(15)] + (pad_temp_shared[((((rc_outer_inner * 56) + (((int)threadIdx.x) % 56)) + 1792))] * input1_shared[(((((((int)threadIdx.x) / 56) * 256) + rc_outer_inner) + 224))]));
    }
  }
  for (int i0_inner = 0; i0_inner < 2; ++i0_inner) {
    for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
      compute[(((((((((((((int)blockIdx.x) / 28) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 28) / 14) * 50176)) + ((((int)threadIdx.x) / 56) * 6272)) + (i1_inner * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)))] = max((compute1[(((i0_inner * 8) + i1_inner))] + input2[(((((((((((((int)blockIdx.x) / 28) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 28) / 14) * 50176)) + ((((int)threadIdx.x) / 56) * 6272)) + (i1_inner * 784)) + (((((int)blockIdx.x) % 14) >> 1) * 112)) + (((((int)threadIdx.x) % 56) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)))]), 0.000000e+00f);
    }
  }
}

