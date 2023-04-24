
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
extern "C" __global__ void __launch_bounds__(32) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float A_shared[32];
  __shared__ float B_shared[2048];
  float A_shared_local[1];
  float B_shared_local[2];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 2; ++k_outer) {
    __syncthreads();
    A_shared[(((int)threadIdx.x))] = A[((((((int)blockIdx.x) * 64) + (k_outer * 32)) + ((int)threadIdx.x)))];
    B_shared[(((int)threadIdx.x))] = B[((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)))];
    B_shared[((((int)threadIdx.x) + 32))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 32))];
    B_shared[((((int)threadIdx.x) + 64))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 64))];
    B_shared[((((int)threadIdx.x) + 96))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 96))];
    B_shared[((((int)threadIdx.x) + 128))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 128))];
    B_shared[((((int)threadIdx.x) + 160))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 160))];
    B_shared[((((int)threadIdx.x) + 192))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 192))];
    B_shared[((((int)threadIdx.x) + 224))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 224))];
    B_shared[((((int)threadIdx.x) + 256))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 256))];
    B_shared[((((int)threadIdx.x) + 288))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 288))];
    B_shared[((((int)threadIdx.x) + 320))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 320))];
    B_shared[((((int)threadIdx.x) + 352))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 352))];
    B_shared[((((int)threadIdx.x) + 384))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 384))];
    B_shared[((((int)threadIdx.x) + 416))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 416))];
    B_shared[((((int)threadIdx.x) + 448))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 448))];
    B_shared[((((int)threadIdx.x) + 480))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 480))];
    B_shared[((((int)threadIdx.x) + 512))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 512))];
    B_shared[((((int)threadIdx.x) + 544))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 544))];
    B_shared[((((int)threadIdx.x) + 576))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 576))];
    B_shared[((((int)threadIdx.x) + 608))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 608))];
    B_shared[((((int)threadIdx.x) + 640))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 640))];
    B_shared[((((int)threadIdx.x) + 672))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 672))];
    B_shared[((((int)threadIdx.x) + 704))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 704))];
    B_shared[((((int)threadIdx.x) + 736))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 736))];
    B_shared[((((int)threadIdx.x) + 768))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 768))];
    B_shared[((((int)threadIdx.x) + 800))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 800))];
    B_shared[((((int)threadIdx.x) + 832))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 832))];
    B_shared[((((int)threadIdx.x) + 864))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 864))];
    B_shared[((((int)threadIdx.x) + 896))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 896))];
    B_shared[((((int)threadIdx.x) + 928))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 928))];
    B_shared[((((int)threadIdx.x) + 960))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 960))];
    B_shared[((((int)threadIdx.x) + 992))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 992))];
    B_shared[((((int)threadIdx.x) + 1024))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1024))];
    B_shared[((((int)threadIdx.x) + 1056))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1056))];
    B_shared[((((int)threadIdx.x) + 1088))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1088))];
    B_shared[((((int)threadIdx.x) + 1120))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1120))];
    B_shared[((((int)threadIdx.x) + 1152))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1152))];
    B_shared[((((int)threadIdx.x) + 1184))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1184))];
    B_shared[((((int)threadIdx.x) + 1216))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1216))];
    B_shared[((((int)threadIdx.x) + 1248))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1248))];
    B_shared[((((int)threadIdx.x) + 1280))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1280))];
    B_shared[((((int)threadIdx.x) + 1312))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1312))];
    B_shared[((((int)threadIdx.x) + 1344))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1344))];
    B_shared[((((int)threadIdx.x) + 1376))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1376))];
    B_shared[((((int)threadIdx.x) + 1408))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1408))];
    B_shared[((((int)threadIdx.x) + 1440))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1440))];
    B_shared[((((int)threadIdx.x) + 1472))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1472))];
    B_shared[((((int)threadIdx.x) + 1504))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1504))];
    B_shared[((((int)threadIdx.x) + 1536))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1536))];
    B_shared[((((int)threadIdx.x) + 1568))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1568))];
    B_shared[((((int)threadIdx.x) + 1600))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1600))];
    B_shared[((((int)threadIdx.x) + 1632))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1632))];
    B_shared[((((int)threadIdx.x) + 1664))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1664))];
    B_shared[((((int)threadIdx.x) + 1696))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1696))];
    B_shared[((((int)threadIdx.x) + 1728))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1728))];
    B_shared[((((int)threadIdx.x) + 1760))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1760))];
    B_shared[((((int)threadIdx.x) + 1792))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1792))];
    B_shared[((((int)threadIdx.x) + 1824))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1824))];
    B_shared[((((int)threadIdx.x) + 1856))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1856))];
    B_shared[((((int)threadIdx.x) + 1888))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1888))];
    B_shared[((((int)threadIdx.x) + 1920))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1920))];
    B_shared[((((int)threadIdx.x) + 1952))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1952))];
    B_shared[((((int)threadIdx.x) + 1984))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1984))];
    B_shared[((((int)threadIdx.x) + 2016))] = B[(((((((int)blockIdx.x) * 4096) + (k_outer * 2048)) + ((int)threadIdx.x)) + 2016))];
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
      A_shared_local[(0)] = A_shared[(k_inner_outer)];
      B_shared_local[(0)] = B_shared[(((k_inner_outer * 64) + ((int)threadIdx.x)))];
      B_shared_local[(1)] = B_shared[((((k_inner_outer * 64) + ((int)threadIdx.x)) + 32))];
      compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
    }
  }
  compute[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) + 32))] = compute_local[(1)];
}

dim3 grid(12, 1, 1);
dim3 block(32, 1, 1);
best_idx 5