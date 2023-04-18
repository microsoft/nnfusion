
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
extern "C" __global__ void __launch_bounds__(64) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float A_shared[64];
  __shared__ float B_shared[4096];
  float A_shared_local[1];
  float B_shared_local[2];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 2; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] = A[(((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)))];
    B_shared[(((int)threadIdx.x))] = B[((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)))];
    B_shared[((((int)threadIdx.x) + 64))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 64))];
    B_shared[((((int)threadIdx.x) + 128))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 128))];
    B_shared[((((int)threadIdx.x) + 192))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 192))];
    B_shared[((((int)threadIdx.x) + 256))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 256))];
    B_shared[((((int)threadIdx.x) + 320))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 320))];
    B_shared[((((int)threadIdx.x) + 384))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 384))];
    B_shared[((((int)threadIdx.x) + 448))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 448))];
    B_shared[((((int)threadIdx.x) + 512))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 512))];
    B_shared[((((int)threadIdx.x) + 576))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 576))];
    B_shared[((((int)threadIdx.x) + 640))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 640))];
    B_shared[((((int)threadIdx.x) + 704))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 704))];
    B_shared[((((int)threadIdx.x) + 768))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 768))];
    B_shared[((((int)threadIdx.x) + 832))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 832))];
    B_shared[((((int)threadIdx.x) + 896))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 896))];
    B_shared[((((int)threadIdx.x) + 960))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 960))];
    B_shared[((((int)threadIdx.x) + 1024))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1024))];
    B_shared[((((int)threadIdx.x) + 1088))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1088))];
    B_shared[((((int)threadIdx.x) + 1152))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1152))];
    B_shared[((((int)threadIdx.x) + 1216))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1216))];
    B_shared[((((int)threadIdx.x) + 1280))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1280))];
    B_shared[((((int)threadIdx.x) + 1344))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1344))];
    B_shared[((((int)threadIdx.x) + 1408))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1408))];
    B_shared[((((int)threadIdx.x) + 1472))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1472))];
    B_shared[((((int)threadIdx.x) + 1536))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1536))];
    B_shared[((((int)threadIdx.x) + 1600))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1600))];
    B_shared[((((int)threadIdx.x) + 1664))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1664))];
    B_shared[((((int)threadIdx.x) + 1728))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1728))];
    B_shared[((((int)threadIdx.x) + 1792))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1792))];
    B_shared[((((int)threadIdx.x) + 1856))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1856))];
    B_shared[((((int)threadIdx.x) + 1920))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1920))];
    B_shared[((((int)threadIdx.x) + 1984))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1984))];
    B_shared[((((int)threadIdx.x) + 2048))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4096))];
    B_shared[((((int)threadIdx.x) + 2112))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4160))];
    B_shared[((((int)threadIdx.x) + 2176))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4224))];
    B_shared[((((int)threadIdx.x) + 2240))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4288))];
    B_shared[((((int)threadIdx.x) + 2304))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4352))];
    B_shared[((((int)threadIdx.x) + 2368))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4416))];
    B_shared[((((int)threadIdx.x) + 2432))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4480))];
    B_shared[((((int)threadIdx.x) + 2496))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4544))];
    B_shared[((((int)threadIdx.x) + 2560))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4608))];
    B_shared[((((int)threadIdx.x) + 2624))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4672))];
    B_shared[((((int)threadIdx.x) + 2688))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4736))];
    B_shared[((((int)threadIdx.x) + 2752))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4800))];
    B_shared[((((int)threadIdx.x) + 2816))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4864))];
    B_shared[((((int)threadIdx.x) + 2880))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4928))];
    B_shared[((((int)threadIdx.x) + 2944))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4992))];
    B_shared[((((int)threadIdx.x) + 3008))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5056))];
    B_shared[((((int)threadIdx.x) + 3072))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5120))];
    B_shared[((((int)threadIdx.x) + 3136))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5184))];
    B_shared[((((int)threadIdx.x) + 3200))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5248))];
    B_shared[((((int)threadIdx.x) + 3264))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5312))];
    B_shared[((((int)threadIdx.x) + 3328))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5376))];
    B_shared[((((int)threadIdx.x) + 3392))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5440))];
    B_shared[((((int)threadIdx.x) + 3456))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5504))];
    B_shared[((((int)threadIdx.x) + 3520))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5568))];
    B_shared[((((int)threadIdx.x) + 3584))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5632))];
    B_shared[((((int)threadIdx.x) + 3648))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5696))];
    B_shared[((((int)threadIdx.x) + 3712))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5760))];
    B_shared[((((int)threadIdx.x) + 3776))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5824))];
    B_shared[((((int)threadIdx.x) + 3840))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5888))];
    B_shared[((((int)threadIdx.x) + 3904))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5952))];
    B_shared[((((int)threadIdx.x) + 3968))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 6016))];
    B_shared[((((int)threadIdx.x) + 4032))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 6080))];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
      A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer))];
      B_shared_local[(0)] = B_shared[(((((((int)threadIdx.x) >> 5) * 2048) + (k_inner_outer * 64)) + (((int)threadIdx.x) & 31)))];
      B_shared_local[(1)] = B_shared[((((((((int)threadIdx.x) >> 5) * 2048) + (k_inner_outer * 64)) + (((int)threadIdx.x) & 31)) + 32))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    }
  }
  compute[((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)))] = compute_local[(0)];
  compute[(((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32))] = compute_local[(1)];
}

dim3 grid(384, 1, 1);
dim3 block(64, 1, 1);
best_idx 9