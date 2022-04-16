//3584_1_1_224_1_1
//128_1024_14_14_512_1_1_SAME
//dim3 grid(3584, 1, 1);
//dim3 block(224, 1, 1);

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
extern "C" __global__ void __launch_bounds__(224) default_function_kernel0(float* __restrict__ input0, float* __restrict__ input1, float* __restrict__ compute, float* __restrict__ input2) {
  float compute1[16];
  __shared__ float pad_temp_shared[896];
  __shared__ float input1_shared[1024];
  compute1[(0)] = 0.000000e+00f;
  compute1[(4)] = 0.000000e+00f;
  compute1[(8)] = 0.000000e+00f;
  compute1[(12)] = 0.000000e+00f;
  compute1[(1)] = 0.000000e+00f;
  compute1[(5)] = 0.000000e+00f;
  compute1[(9)] = 0.000000e+00f;
  compute1[(13)] = 0.000000e+00f;
  compute1[(2)] = 0.000000e+00f;
  compute1[(6)] = 0.000000e+00f;
  compute1[(10)] = 0.000000e+00f;
  compute1[(14)] = 0.000000e+00f;
  compute1[(3)] = 0.000000e+00f;
  compute1[(7)] = 0.000000e+00f;
  compute1[(11)] = 0.000000e+00f;
  compute1[(15)] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    ((float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2))))[0] = ((float2*)(input0 + (((((((((int)blockIdx.x) / 56) * 401408) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)))))[0];
    ((float2*)(pad_temp_shared + (((((int)threadIdx.x) * 2) + 448))))[0] = ((float2*)(input0 + ((((((((((int)blockIdx.x) / 56) * 401408) + (rc_outer_outer * 3136)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 200704))))[0];
    input1_shared[(((int)threadIdx.x))] = input1[(((((((((int)blockIdx.x) % 56) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)))];
    input1_shared[((((int)threadIdx.x) + 224))] = input1[((((((((((int)blockIdx.x) % 56) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336))];
    input1_shared[((((int)threadIdx.x) + 448))] = input1[((((((((((int)blockIdx.x) % 56) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 28672))];
    input1_shared[((((int)threadIdx.x) + 672))] = input1[((((((((((int)blockIdx.x) % 56) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 43008))];
    if (((int)threadIdx.x) < 128) {
      input1_shared[((((int)threadIdx.x) + 896))] = input1[((((((((((int)blockIdx.x) % 56) / 7) * 65536) + ((((int)threadIdx.x) >> 4) * 1024)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 57344))];
    }
    __syncthreads();
    for (int nn_outer_inner = 0; nn_outer_inner < 2; ++nn_outer_inner) {
      for (int xx_outer_inner = 0; xx_outer_inner < 2; ++xx_outer_inner) {
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner))] * input1_shared[(((((int)threadIdx.x) / 14) * 16))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 256))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 512))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 768))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 28))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 1))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 28))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 257))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 28))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 513))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 28))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 769))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 56))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 2))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 56))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 258))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 56))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 514))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 56))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 770))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 84))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 3))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 84))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 259))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 84))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 515))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 84))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 771))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 112))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 4))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 112))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 260))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 112))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 516))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 112))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 772))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 140))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 5))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 140))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 261))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 140))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 517))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 140))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 773))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 168))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 6))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 168))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 262))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 168))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 518))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 168))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 774))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 196))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 7))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 196))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 263))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 196))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 519))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 196))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 775))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 224))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 8))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 224))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 264))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 224))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 520))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 224))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 776))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 252))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 9))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 252))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 265))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 252))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 521))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 252))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 777))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 280))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 10))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 280))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 266))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 280))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 522))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 280))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 778))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 308))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 11))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 308))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 267))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 308))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 523))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 308))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 779))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 336))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 12))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 336))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 268))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 336))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 524))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 336))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 780))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 364))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 13))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 364))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 269))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 364))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 525))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 364))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 781))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 392))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 14))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 392))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 270))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 392))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 526))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 392))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 782))]));
        compute1[(((nn_outer_inner * 2) + xx_outer_inner))] = (compute1[(((nn_outer_inner * 2) + xx_outer_inner))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 420))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 15))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 4))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 420))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 271))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 8))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 420))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 527))]));
        compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] = (compute1[((((nn_outer_inner * 2) + xx_outer_inner) + 12))] + (pad_temp_shared[(((((nn_outer_inner * 448) + ((((int)threadIdx.x) % 14) * 2)) + xx_outer_inner) + 420))] * input1_shared[((((((int)threadIdx.x) / 14) * 16) + 783))]));
      }
    }
  }
  for (int i0_inner = 0; i0_inner < 2; ++i0_inner) {
    for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
      compute[(((((((((((int)blockIdx.x) / 56) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 56) / 7) * 12544)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner))] = max((compute1[(((i0_inner * 2) + i3_inner))] + input2[(((((((((((int)blockIdx.x) / 56) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 56) / 7) * 12544)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner))]), 0.000000e+00f);
      compute[((((((((((((int)blockIdx.x) / 56) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 56) / 7) * 12544)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner) + 3136))] = max((compute1[((((i0_inner * 2) + i3_inner) + 4))] + input2[((((((((((((int)blockIdx.x) / 56) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 56) / 7) * 12544)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner) + 3136))]), 0.000000e+00f);
      compute[((((((((((((int)blockIdx.x) / 56) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 56) / 7) * 12544)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner) + 6272))] = max((compute1[((((i0_inner * 2) + i3_inner) + 8))] + input2[((((((((((((int)blockIdx.x) / 56) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 56) / 7) * 12544)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner) + 6272))]), 0.000000e+00f);
      compute[((((((((((((int)blockIdx.x) / 56) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 56) / 7) * 12544)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner) + 9408))] = max((compute1[((((i0_inner * 2) + i3_inner) + 12))] + input2[((((((((((((int)blockIdx.x) / 56) * 200704) + (i0_inner * 100352)) + (((((int)blockIdx.x) % 56) / 7) * 12544)) + ((((int)threadIdx.x) / 14) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 14) * 2)) + i3_inner) + 9408))]), 0.000000e+00f);
    }
  }
}

