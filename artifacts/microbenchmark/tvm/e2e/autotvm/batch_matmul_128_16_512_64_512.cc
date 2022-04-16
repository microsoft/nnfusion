//8_8_2048_32_4_1
//128_16_512_64_512
//dim3 grid(8, 8, 2048);
//dim3 block(32, 4, 1);

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
extern "C" __global__ void __launch_bounds__(128) conv_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NN) {
  float T_batch_matmul_NN_local[32];
  __shared__ float placeholder_shared[4096];
  __shared__ float placeholder_d_shared[4096];
  float placeholder_shared_local[16];
  float placeholder_d_shared_local[2];
  for (int i_c_init = 0; i_c_init < 16; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
      T_batch_matmul_NN_local[(((i_c_init * 2) + j_c_init))] = 0.000000e+00f;
    }
  }
  placeholder_shared[(((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)))] = placeholder[(((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 1))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 1))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 64))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 64))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 65))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 65))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 128))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 128))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 129))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 129))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 192))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 192))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 193))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 193))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 256))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 256))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 257))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 257))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 320))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 320))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 321))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 321))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 384))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 384))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 385))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 385))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 448))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 448))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 449))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 449))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 512))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 512))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 513))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 513))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 576))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 576))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 577))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 577))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 640))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 640))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 641))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 641))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 704))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 704))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 705))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 705))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 768))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 768))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 769))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 769))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 832))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 832))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 833))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 833))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 896))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 896))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 897))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 897))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 960))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 960))];
  placeholder_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 961))] = placeholder[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 2)) + 961))];
  placeholder_d_shared[(((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)))] = placeholder1[(((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 64))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 512))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 65))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 513))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 128))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1024))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 129))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1025))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 192))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1536))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 193))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1537))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 256))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 2048))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 257))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 2049))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 320))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 2560))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 321))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 2561))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 384))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 3072))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 385))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 3073))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 448))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 3584))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 449))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 3585))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 512))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 4096))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 513))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 4097))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 576))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 4608))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 577))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 4609))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 640))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 5120))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 641))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 5121))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 704))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 5632))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 705))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 5633))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 768))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 6144))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 769))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 6145))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 832))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 6656))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 833))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 6657))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 896))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 7168))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 897))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 7169))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 960))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 7680))];
  placeholder_d_shared[((((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 2)) + 961))] = placeholder1[((((((((int)blockIdx.z) * 32768) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 7681))];
  __syncthreads();
  for (int k_inner = 0; k_inner < 64; ++k_inner) {
    placeholder_shared_local[(0)] = placeholder_shared[(((((int)threadIdx.y) * 1024) + k_inner))];
    placeholder_shared_local[(1)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 64))];
    placeholder_shared_local[(2)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 128))];
    placeholder_shared_local[(3)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 192))];
    placeholder_shared_local[(4)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 256))];
    placeholder_shared_local[(5)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 320))];
    placeholder_shared_local[(6)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 384))];
    placeholder_shared_local[(7)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 448))];
    placeholder_shared_local[(8)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 512))];
    placeholder_shared_local[(9)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 576))];
    placeholder_shared_local[(10)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 640))];
    placeholder_shared_local[(11)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 704))];
    placeholder_shared_local[(12)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 768))];
    placeholder_shared_local[(13)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 832))];
    placeholder_shared_local[(14)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 896))];
    placeholder_shared_local[(15)] = placeholder_shared[((((((int)threadIdx.y) * 1024) + k_inner) + 960))];
    placeholder_d_shared_local[(0)] = placeholder_d_shared[(((k_inner * 64) + (((int)threadIdx.x) * 2)))];
    placeholder_d_shared_local[(1)] = placeholder_d_shared[((((k_inner * 64) + (((int)threadIdx.x) * 2)) + 1))];
    T_batch_matmul_NN_local[(0)] = (T_batch_matmul_NN_local[(0)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(1)] = (T_batch_matmul_NN_local[(1)] + (placeholder_shared_local[(0)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(2)] = (T_batch_matmul_NN_local[(2)] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(3)] = (T_batch_matmul_NN_local[(3)] + (placeholder_shared_local[(1)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(4)] = (T_batch_matmul_NN_local[(4)] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(5)] = (T_batch_matmul_NN_local[(5)] + (placeholder_shared_local[(2)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(6)] = (T_batch_matmul_NN_local[(6)] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(7)] = (T_batch_matmul_NN_local[(7)] + (placeholder_shared_local[(3)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(8)] = (T_batch_matmul_NN_local[(8)] + (placeholder_shared_local[(4)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(9)] = (T_batch_matmul_NN_local[(9)] + (placeholder_shared_local[(4)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(10)] = (T_batch_matmul_NN_local[(10)] + (placeholder_shared_local[(5)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(11)] = (T_batch_matmul_NN_local[(11)] + (placeholder_shared_local[(5)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(12)] = (T_batch_matmul_NN_local[(12)] + (placeholder_shared_local[(6)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(13)] = (T_batch_matmul_NN_local[(13)] + (placeholder_shared_local[(6)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(14)] = (T_batch_matmul_NN_local[(14)] + (placeholder_shared_local[(7)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(15)] = (T_batch_matmul_NN_local[(15)] + (placeholder_shared_local[(7)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(16)] = (T_batch_matmul_NN_local[(16)] + (placeholder_shared_local[(8)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(17)] = (T_batch_matmul_NN_local[(17)] + (placeholder_shared_local[(8)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(18)] = (T_batch_matmul_NN_local[(18)] + (placeholder_shared_local[(9)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(19)] = (T_batch_matmul_NN_local[(19)] + (placeholder_shared_local[(9)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(20)] = (T_batch_matmul_NN_local[(20)] + (placeholder_shared_local[(10)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(21)] = (T_batch_matmul_NN_local[(21)] + (placeholder_shared_local[(10)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(22)] = (T_batch_matmul_NN_local[(22)] + (placeholder_shared_local[(11)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(23)] = (T_batch_matmul_NN_local[(23)] + (placeholder_shared_local[(11)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(24)] = (T_batch_matmul_NN_local[(24)] + (placeholder_shared_local[(12)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(25)] = (T_batch_matmul_NN_local[(25)] + (placeholder_shared_local[(12)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(26)] = (T_batch_matmul_NN_local[(26)] + (placeholder_shared_local[(13)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(27)] = (T_batch_matmul_NN_local[(27)] + (placeholder_shared_local[(13)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(28)] = (T_batch_matmul_NN_local[(28)] + (placeholder_shared_local[(14)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(29)] = (T_batch_matmul_NN_local[(29)] + (placeholder_shared_local[(14)] * placeholder_d_shared_local[(1)]));
    T_batch_matmul_NN_local[(30)] = (T_batch_matmul_NN_local[(30)] + (placeholder_shared_local[(15)] * placeholder_d_shared_local[(0)]));
    T_batch_matmul_NN_local[(31)] = (T_batch_matmul_NN_local[(31)] + (placeholder_shared_local[(15)] * placeholder_d_shared_local[(1)]));
  }
  T_batch_matmul_NN[((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)))] = T_batch_matmul_NN_local[(0)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1))] = T_batch_matmul_NN_local[(1)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 512))] = T_batch_matmul_NN_local[(2)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 513))] = T_batch_matmul_NN_local[(3)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1024))] = T_batch_matmul_NN_local[(4)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1025))] = T_batch_matmul_NN_local[(5)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1536))] = T_batch_matmul_NN_local[(6)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 1537))] = T_batch_matmul_NN_local[(7)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 2048))] = T_batch_matmul_NN_local[(8)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 2049))] = T_batch_matmul_NN_local[(9)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 2560))] = T_batch_matmul_NN_local[(10)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 2561))] = T_batch_matmul_NN_local[(11)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 3072))] = T_batch_matmul_NN_local[(12)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 3073))] = T_batch_matmul_NN_local[(13)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 3584))] = T_batch_matmul_NN_local[(14)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 3585))] = T_batch_matmul_NN_local[(15)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 4096))] = T_batch_matmul_NN_local[(16)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 4097))] = T_batch_matmul_NN_local[(17)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 4608))] = T_batch_matmul_NN_local[(18)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 4609))] = T_batch_matmul_NN_local[(19)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 5120))] = T_batch_matmul_NN_local[(20)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 5121))] = T_batch_matmul_NN_local[(21)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 5632))] = T_batch_matmul_NN_local[(22)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 5633))] = T_batch_matmul_NN_local[(23)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 6144))] = T_batch_matmul_NN_local[(24)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 6145))] = T_batch_matmul_NN_local[(25)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 6656))] = T_batch_matmul_NN_local[(26)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 6657))] = T_batch_matmul_NN_local[(27)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 7168))] = T_batch_matmul_NN_local[(28)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 7169))] = T_batch_matmul_NN_local[(29)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 7680))] = T_batch_matmul_NN_local[(30)];
  T_batch_matmul_NN[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 8192)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 2)) + 7681))] = T_batch_matmul_NN_local[(31)];
}

