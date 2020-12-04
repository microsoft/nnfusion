// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cuda_langunit.hpp"
//#include "cuda_cublas.hpp"
#include "cuda_cudnn.hpp"

using namespace nnfusion::kernels;

// Header
LU_DEFINE(header::cuda, "#include <cuda.h>\n#include <cuda_runtime.h>\n");
LU_DEFINE(header::cublas, "#include <cublas_v2.h>\n");
LU_DEFINE(header::cudnn, "#include <cudnn.h>\n");
LU_DEFINE(header::super_scaler, "#include \"super_scaler.h\"\n");
LU_DEFINE(header::cupti, "#include <cupti.h>\n");
LU_DEFINE(header::cuda_prof_api, "#include <cuda_profiler_api.h>\n");
LU_DEFINE(header::cuda_fp16, "#include <cuda_fp16.h>\n");

// Macro
LU_DEFINE(macro::HALF_MAX,
          R"(#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half max(half x, half y) { return x > y ? x : y; }
inline __device__ half min(half x, half y) { return x < y ? x : y; }
#endif
)");

LU_DEFINE(
    macro::CUDA_SAFE_CALL_NO_THROW,
    R"(#define CUDA_SAFE_CALL_NO_THROW(x)                                                                 \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            std::cout << safe_call_ss.str() << std::endl;                                          \
        }                                                                                          \
    } while (0)
)");

LU_DEFINE(
    macro::CUDA_SAFE_CALL,
    R"(#define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
)");

LU_DEFINE(
    macro::CUDNN_SAFE_CALL_NO_THROW,
    R"(#define CUDNN_SAFE_CALL_NO_THROW(func)                                                             \
    do                                                                                             \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            const char* msg = cudnnGetErrorString(e);                                              \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            std::cout << safe_call_ss.str() << std::endl;                                          \
        }                                                                                          \
    } while (0)
)");

LU_DEFINE(
    macro::CUDNN_SAFE_CALL,
    R"(#define CUDNN_SAFE_CALL(func)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            const char* msg = cudnnGetErrorString(e);                                              \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
)");

LU_DEFINE(
    macro::CUBLAS_SAFE_CALL_NO_THROW,
    R"(#define CUBLAS_SAFE_CALL_NO_THROW(func)                                                            \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t e = (func);                                                                 \
        if (e != CUBLAS_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
            std::cout << safe_call_ss.str() << std::endl;                                          \
        }                                                                                          \
    } while (0)
    )");

LU_DEFINE(
    macro::CUBLAS_SAFE_CALL,
    R"(#define CUBLAS_SAFE_CALL(func)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t e = (func);                                                                 \
        if (e != CUBLAS_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
   )");

LU_DEFINE(
    macro::CUDA_SAFE_LAUNCH,
    R"(#define CUDA_SAFE_LAUNCH(x)                                                                       \
    do                                                                                             \
    {                                                                                              \
        (x);                                                                                       \
        cudaError_t result = cudaGetLastError();                                                   \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
)");

LU_DEFINE(macro::CUPTI_CALL,
          R"(#define CUPTI_CALL(call)                                                \
    do {                                                                  \
      CUptiResult _status = call;                                         \
      if (_status != CUPTI_SUCCESS) {                                     \
        const char *errstr;                                               \
        cuptiGetResultString(_status, &errstr);                           \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                __FILE__, __LINE__, #call, errstr);                       \
        exit(-1);                                                         \
      }                                                                   \
    } while (0)
)");

// Declaration
//<TODO>Need special code for this global_cublas_handle
LU_DEFINE(declaration::num_SMs, "int num_SMs;\n");
LU_DEFINE(declaration::global_cublas_handle, "cublasHandle_t global_cublas_handle;\n");
LU_DEFINE(declaration::global_cudnn_handle, "cudnnHandle_t global_cudnn_handle;\n");
LU_DEFINE(
    declaration::division_by_invariant_multiplication,
    R"(__device__ __forceinline__ int division_by_invariant_multiplication(int value, int magic, int shift)
{
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}
)");

LU_DEFINE(
    declaration::rocm_division_by_invariant_multiplication,
    R"(__device__ __forceinline__ int division_by_invariant_multiplication(int value, int magic, int shift)
{
    long long res64 = ((long long)(unsigned int)value) * ((long long)(unsigned int)magic);
    int hi32 = res64 >> 32;
    if(magic == 1)
        hi32 = value;
    int result = hi32 >> shift;
    return result;
}
)");

LU_DEFINE(declaration::mod16,
          R"(__device__ __forceinline__ int mod16(int numerator, int div, int maxdiv)
{
    int res;
    asm("vmad.s32.u32.u32 %0, -%1.h0, %2.h0, %3;" : "=r"(res) : "r"(div), "r"(maxdiv), "r"(numerator));
    return res;
}
)");

LU_DEFINE(declaration::mad16,
          R"(__device__ __forceinline__ int mad16(int a, int b, int c)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}
)");

LU_DEFINE(
    declaration::load,
    R"(__device__ __forceinline__ char  load(const char*  __restrict__ in, int i=0, bool b=true)
{
    char v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
} 
__device__ __forceinline__ float  load(const float*  __restrict__ in, int i=0, bool b=true)
{
    float v = 0.0f;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ half  load(const half*  __restrict__ in, int i=0, bool b=true)
{
    half v = 0.0f;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int32_t  load(const int32_t*  __restrict__ in, int i=0, bool b=true)
{
    int32_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int64_t  load(const int64_t*  __restrict__ in, int i=0, bool b=true)
{
    int64_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
)");

LU_DEFINE(declaration::cuda_fp16_scale,
          R"(
__global__ void nnfusionHalfScaleKernel(half *x, half *alpha, size_t count)
{
    size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
    x += offset;
    if (offset < count)
    {
        *x *= *alpha;
    }
}

void nnfusionHalfScale(half *x, half *alpha, size_t len)
{
    nnfusionHalfScaleKernel<<<(len+255)/256, 256>>>(x, alpha, len);
}
  )")

LU_DEFINE_EXTEND(declaration::cuda_reduce_primitive,
                 R"(
#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

__forceinline__ __device__ float CudaShuffleDownSync(unsigned mask, float val,
                                                     int delta,
                                                     int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, delta, width);
#endif
}

__device__ static float reduceMax(float val, int tid, int blockSize, float* shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val = max(val, CudaShuffleDownSync(mask, val, 16));
  val = max(val, CudaShuffleDownSync(mask, val, 8));
  val = max(val, CudaShuffleDownSync(mask, val, 4));
  val = max(val, CudaShuffleDownSync(mask, val, 2));
  val = max(val, CudaShuffleDownSync(mask, val, 1));

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val = max(val, CudaShuffleDownSync(mask, val, 16));
    val = max(val, CudaShuffleDownSync(mask, val, 8));
    val = max(val, CudaShuffleDownSync(mask, val, 4));
    val = max(val, CudaShuffleDownSync(mask, val, 2));
    val = max(val, CudaShuffleDownSync(mask, val, 1));
  }

  return val;
}

__device__ static float reduceSum(float val, int tid, int blockSize, float* shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val += CudaShuffleDownSync(mask, val, 16);
  val += CudaShuffleDownSync(mask, val, 8);
  val += CudaShuffleDownSync(mask, val, 4);
  val += CudaShuffleDownSync(mask, val, 2);
  val += CudaShuffleDownSync(mask, val, 1);

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;

  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val += CudaShuffleDownSync(mask, val, 16);
    val += CudaShuffleDownSync(mask, val, 8);
    val += CudaShuffleDownSync(mask, val, 4);
    val += CudaShuffleDownSync(mask, val, 2);
    val += CudaShuffleDownSync(mask, val, 1);
  }

  return val;
}
)",
                 R"(
#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

__forceinline__ __device__ float CudaShuffleDownSync(unsigned mask, float val,
                                                     int delta,
                                                     int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, delta, width);
#endif
}

__device__ static float reduceMax(float val, int tid, int blockSize, float* shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val = max(val, CudaShuffleDownSync(mask, val, 16));
  val = max(val, CudaShuffleDownSync(mask, val, 8));
  val = max(val, CudaShuffleDownSync(mask, val, 4));
  val = max(val, CudaShuffleDownSync(mask, val, 2));
  val = max(val, CudaShuffleDownSync(mask, val, 1));

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val = max(val, CudaShuffleDownSync(mask, val, 16));
    val = max(val, CudaShuffleDownSync(mask, val, 8));
    val = max(val, CudaShuffleDownSync(mask, val, 4));
    val = max(val, CudaShuffleDownSync(mask, val, 2));
    val = max(val, CudaShuffleDownSync(mask, val, 1));
  }

  return val;
}

__device__ static float reduceSum(float val, int tid, int blockSize, float* shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val += CudaShuffleDownSync(mask, val, 16);
  val += CudaShuffleDownSync(mask, val, 8);
  val += CudaShuffleDownSync(mask, val, 4);
  val += CudaShuffleDownSync(mask, val, 2);
  val += CudaShuffleDownSync(mask, val, 1);

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;

  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val += CudaShuffleDownSync(mask, val, 16);
    val += CudaShuffleDownSync(mask, val, 8);
    val += CudaShuffleDownSync(mask, val, 4);
    val += CudaShuffleDownSync(mask, val, 2);
    val += CudaShuffleDownSync(mask, val, 1);
  }

  return val;
}
)",
                 "");

LU_DEFINE_EXTEND(declaration::cuda_layer_norm,
                 R"(
// Check compute capability
const int GPU_WARP_SIZE = 32;
const uint64_t MAX_GRID_Y = 65535;

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, srcLane, width);
#else
  return __shfl(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(mask, value, delta, width);
#else
  return __shfl_up(value, delta, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

__device__ void cuWelfordOnlineSum(
    const float curr,
    float& mu,
    float& sigma2,
    float& count) {
  count = count + float(1);
  float delta = curr - mu;
  float lmean = mu + delta / count;
  mu = lmean;
  float delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

__device__ void cuChanOnlineSum(
    const float muB,
    const float sigma2B,
    const float countB,
    float& mu,
    float& sigma2,
    float& count) {
  float delta = muB - mu;
  float nA = count;
  float nB = countB;
  count = count + countB;
  float nX = count;
  if (nX > float(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = float(0);
    sigma2 = float(0);
  }
}

__device__ void cuWelfordMuSigma2(
    const float* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    float& mu,
    float& sigma2,
    float* buf) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(float)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = float(0);
  mu = float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const float* lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        float curr = static_cast<float>(lvals[l + k]);
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      cuWelfordOnlineSum(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    #pragma unroll
    for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
      float muB = WARP_SHFL_DOWN(mu, stride);
      float countB = WARP_SHFL_DOWN(count, stride);
      float sigma2B = WARP_SHFL_DOWN(sigma2, stride);
      cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float muB = ubuf[2 * threadIdx.y];
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / float(n2), 0);
    }
  }
}

__global__ void cuApplyLayerNorm(
    float* __restrict__ output_vals,
    float* __restrict__ mean,
    float* __restrict__ invvar,
    const float* __restrict__ vals,
    const int n1,
    const int n2,
    const float epsilon,
    const float* __restrict__ gamma,
    const float* __restrict__ beta) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensors are contiguous
  //
  for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    extern __shared__ float s_float[];
    float* buf = (float*)s_float;
    float mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf);
    const float* lvals = vals + i1 * n2;
    float* ovals = output_vals + i1 * n2;
    float c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && beta != NULL) {
      for (int i = thrx; i < n2; i += numx) {
        float curr = static_cast<float>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<float>(c_invvar * (curr - mu)) + beta[i];
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        float curr = static_cast<float>(lvals[i]);
        ovals[i] = static_cast<float>(c_invvar * (curr - mu));
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
  }
}

void HostApplyLayerNorm(
    float* output,
    float* mean,
    float* invvar,
    const float* input,
    int64_t n1,
    int64_t n2,
    double epsilon,
    const float* gamma,
    const float* beta) {
  const dim3 threads(GPU_WARP_SIZE, 4, 1);
  const dim3 blocks(1, std::min((uint64_t)n1, MAX_GRID_Y), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(float) + (threads.y / 2) * sizeof(float) : 0;
  cuApplyLayerNorm<<<blocks, threads, nshared, 0>>>(
      output,
      mean,
      invvar,
      input,
      n1, n2,
      float(epsilon),
      gamma, beta);
}
)",
                 R"(
// Check compute capability
extern const int GPU_WARP_SIZE;
extern const uint64_t MAX_GRID_Y;

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, srcLane, width);
#else
  return __shfl(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(mask, value, delta, width);
#else
  return __shfl_up(value, delta, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

extern __device__ void cuWelfordOnlineSum(const float curr, float& mu, float& sigma2, float& count);

extern __device__ void cuChanOnlineSum(const float muB, const float sigma2B, const float countB, float& mu, float& sigma2, float& count); 

extern __device__ void cuWelfordMuSigma2(
    const float* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    float& mu,
    float& sigma2,
    float* buf);

extern __global__ void cuApplyLayerNorm(
    float* __restrict__ output_vals,
    float* __restrict__ mean,
    float* __restrict__ invvar,
    const float* __restrict__ vals,
    const int n1,
    const int n2,
    const float epsilon,
    const float* __restrict__ gamma,
    const float* __restrict__ beta);

extern void HostApplyLayerNorm(
    float* output,
    float* mean,
    float* invvar,
    const float* input,
    int64_t n1,
    int64_t n2,
    double epsilon,
    const float* gamma,
    const float* beta);

)",
                 R"(
// Check compute capability
const int GPU_WARP_SIZE = 32;
const uint64_t MAX_GRID_Y = 65535;

__device__ void cuWelfordOnlineSum(
    const float curr,
    float& mu,
    float& sigma2,
    float& count) {
  count = count + float(1);
  float delta = curr - mu;
  float lmean = mu + delta / count;
  mu = lmean;
  float delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

__device__ void cuChanOnlineSum(
    const float muB,
    const float sigma2B,
    const float countB,
    float& mu,
    float& sigma2,
    float& count) {
  float delta = muB - mu;
  float nA = count;
  float nB = countB;
  count = count + countB;
  float nX = count;
  if (nX > float(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = float(0);
    sigma2 = float(0);
  }
}

__device__ void cuWelfordMuSigma2(
    const float* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    float& mu,
    float& sigma2,
    float* buf) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(float)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = float(0);
  mu = float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const float* lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        float curr = static_cast<float>(lvals[l + k]);
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      cuWelfordOnlineSum(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    #pragma unroll
    for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
      float muB = WARP_SHFL_DOWN(mu, stride);
      float countB = WARP_SHFL_DOWN(count, stride);
      float sigma2B = WARP_SHFL_DOWN(sigma2, stride);
      cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float muB = ubuf[2 * threadIdx.y];
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / float(n2), 0);
    }
  }
}

__global__ void cuApplyLayerNorm(
    float* __restrict__ output_vals,
    float* __restrict__ mean,
    float* __restrict__ invvar,
    const float* __restrict__ vals,
    const int n1,
    const int n2,
    const float epsilon,
    const float* __restrict__ gamma,
    const float* __restrict__ beta) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensors are contiguous
  //
  for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    extern __shared__ float s_float[];
    float* buf = (float*)s_float;
    float mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf);
    const float* lvals = vals + i1 * n2;
    float* ovals = output_vals + i1 * n2;
    float c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && beta != NULL) {
      for (int i = thrx; i < n2; i += numx) {
        float curr = static_cast<float>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<float>(c_invvar * (curr - mu)) + beta[i];
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        float curr = static_cast<float>(lvals[i]);
        ovals[i] = static_cast<float>(c_invvar * (curr - mu));
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
  }
}

void HostApplyLayerNorm(
    float* output,
    float* mean,
    float* invvar,
    const float* input,
    int64_t n1,
    int64_t n2,
    double epsilon,
    const float* gamma,
    const float* beta) {
  const dim3 threads(GPU_WARP_SIZE, 4, 1);
  const dim3 blocks(1, std::min((uint64_t)n1, MAX_GRID_Y), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(float) + (threads.y / 2) * sizeof(float) : 0;
  cuApplyLayerNorm<<<blocks, threads, nshared, 0>>>(
      output,
      mean,
      invvar,
      input,
      n1, n2,
      float(epsilon),
      gamma, beta);
}
)");