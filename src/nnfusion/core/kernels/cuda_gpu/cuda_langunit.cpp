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
LU_DEFINE(header::superscaler, "#include \"superscaler.h\"\n");
LU_DEFINE(header::cupti, "#include <cupti.h>\n");
LU_DEFINE(header::cuda_prof_api, "#include <cuda_profiler_api.h>\n");
LU_DEFINE(header::cuda_fp16, "#include <cuda_fp16.h>\n");
LU_DEFINE(header::cuda_mma, "#include <mma.h>\n");
LU_DEFINE(header::cub, "#include <cub/cub.cuh>\n");
LU_DEFINE(header::math_constants, "#include <math_constants.h>\n");
LU_DEFINE(header::cutlass, "#include \"cutlass/cutlass.h\"\n");
LU_DEFINE(header::kernel_forward, "#include \"kernel_forward.h\"\n");

// Macro
LU_DEFINE(macro::HALF_MAX,
          R"(#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half hmax(half x, half y) { return x > y ? x : y; }
inline __device__ half hmin(half x, half y) { return x < y ? x : y; }
#endif
)");

LU_DEFINE(macro::CUDA_HALF_OPERATIONS,
          R"(
#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x, half y) {                   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x) {                          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
)");

LU_DEFINE(macro::TVM_PACK_VALUES,
          R"(
inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  int2 i0 = make_int2(x0, x1);
  int2 i1 = make_int2(x2, x3);
  int2 i2 = make_int2(x4, x5);
  int2 i3 = make_int2(x6, x7);
  long long l0 = *(long long*)&i0;
  long long l1 = *(long long*)&i1;
  long long l2 = *(long long*)&i2;
  long long l3 = *(long long*)&i3;
  return make_longlong4(l0, l1, l2, l3);
}

inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}
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
__device__ __forceinline__ long long  load(const long long*  __restrict__ in, int i=0, bool b=true)
{
    long long v = 0;
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
  )");

LU_DEFINE(declaration::cuda_convert_template,
          R"(template<typename InT, typename OutT>
__device__ __forceinline__ OutT convert(InT x0)
{
    return x0;
}

template <>
__device__ __forceinline__ half convert(int64_t a)
{
    return 	__ll2half_rn((long long)a);
}

template <>
__device__ __forceinline__ int64_t convert(half a)
{
    return 	__half2ll_rn(a);
}

)");

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
template <typename T>
__device__ void cuWelfordOnlineSum(
    const T curr,
    T& mu,
    T& sigma2,
    T& count) {
  count = count + T(1);
  T delta = curr - mu;
  T lmean = mu + delta / count;
  mu = lmean;
  T delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template <typename T>
__device__ void cuChanOnlineSum(
    const T muB,
    const T sigma2B,
    const T countB,
    T& mu,
    T& sigma2,
    T& count) {
  T delta = muB - mu;
  T nA = count;
  T nB = countB;
  count = count + countB;
  T nX = count;
  if (nX > T(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = T(0);
    sigma2 = T(0);
  }
}

template <typename T>
__device__ void cuWelfordMuSigma2(
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    T& mu,
    T& sigma2,
    T* buf) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(T)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  T count = T(0);
  mu = T(0);
  sigma2 = T(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        T curr = static_cast<T>(lvals[l + k]);
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      T curr = static_cast<T>(lvals[l]);
      cuWelfordOnlineSum(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    #pragma unroll
    for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
      T muB = WARP_SHFL_DOWN(mu, stride);
      T countB = WARP_SHFL_DOWN(count, stride);
      T sigma2B = WARP_SHFL_DOWN(sigma2, stride);
      cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      T* ubuf = (T*)buf;
      T* ibuf = (T*)(ubuf + blockDim.y);
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
          T muB = ubuf[2 * threadIdx.y];
          T sigma2B = ubuf[2 * threadIdx.y + 1];
          T countB = ibuf[threadIdx.y];
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
      sigma2 = ubuf[1] / T(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / T(n2), 0);
    }
  }
}

template <typename T>
__global__ void cuApplyLayerNorm(
    T* __restrict__ output_vals,
    T* __restrict__ mean,
    T* __restrict__ invvar,
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const T epsilon,
    const T* __restrict__ gamma,
    const T* __restrict__ beta) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensors are contiguous
  //
  for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    extern __shared__ T s_float[];
    T* buf = (T*)s_float;
    T mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf);
    const T* lvals = vals + i1 * n2;
    T* ovals = output_vals + i1 * n2;
    T c_invvar = Rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && beta != NULL) {
      for (int i = thrx; i < n2; i += numx) {
        T curr = static_cast<T>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<T>(c_invvar * (curr - mu)) + beta[i];
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        T curr = static_cast<T>(lvals[i]);
        ovals[i] = static_cast<T>(c_invvar * (curr - mu));
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
  }
}

template <typename T>
void HostApplyLayerNorm(
    T* output,
    T* mean,
    T* invvar,
    const T* input,
    int64_t n1,
    int64_t n2,
    T epsilon,
    const T* gamma,
    const T* beta) {
  const dim3 threads(GPU_WARP_SIZE, 4, 1);
  const dim3 blocks(1, std::min((uint64_t)n1, MAX_GRID_Y), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(T) + (threads.y / 2) * sizeof(T) : 0;
  cuApplyLayerNorm<<<blocks, threads, nshared, 0>>>(
      output,
      mean,
      invvar,
      input,
      n1, n2,
      epsilon,
      gamma, beta);
}
)",
                 R"(
template <typename T>
__device__ void cuWelfordOnlineSum(
    const T curr,
    T& mu,
    T& sigma2,
    T& count) {
  count = count + T(1);
  T delta = curr - mu;
  T lmean = mu + delta / count;
  mu = lmean;
  T delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template <typename T>
__device__ void cuChanOnlineSum(
    const T muB,
    const T sigma2B,
    const T countB,
    T& mu,
    T& sigma2,
    T& count) {
  T delta = muB - mu;
  T nA = count;
  T nB = countB;
  count = count + countB;
  T nX = count;
  if (nX > T(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = T(0);
    sigma2 = T(0);
  }
}

template <typename T>
__device__ void cuWelfordMuSigma2(
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    T& mu,
    T& sigma2,
    T* buf) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(T)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  T count = T(0);
  mu = T(0);
  sigma2 = T(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        T curr = static_cast<T>(lvals[l + k]);
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      T curr = static_cast<T>(lvals[l]);
      cuWelfordOnlineSum(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    #pragma unroll
    for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
      T muB = WARP_SHFL_DOWN(mu, stride);
      T countB = WARP_SHFL_DOWN(count, stride);
      T sigma2B = WARP_SHFL_DOWN(sigma2, stride);
      cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      T* ubuf = (T*)buf;
      T* ibuf = (T*)(ubuf + blockDim.y);
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
          T muB = ubuf[2 * threadIdx.y];
          T sigma2B = ubuf[2 * threadIdx.y + 1];
          T countB = ibuf[threadIdx.y];
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
      sigma2 = ubuf[1] / T(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / T(n2), 0);
    }
  }
}

template <typename T>
__global__ void cuApplyLayerNorm(
    T* __restrict__ output_vals,
    T* __restrict__ mean,
    T* __restrict__ invvar,
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const T epsilon,
    const T* __restrict__ gamma,
    const T* __restrict__ beta) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensors are contiguous
  //
  for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    extern __shared__ T s_float[];
    T* buf = (T*)s_float;
    T mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf);
    const T* lvals = vals + i1 * n2;
    T* ovals = output_vals + i1 * n2;
    T c_invvar = Rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && beta != NULL) {
      for (int i = thrx; i < n2; i += numx) {
        T curr = static_cast<T>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<T>(c_invvar * (curr - mu)) + beta[i];
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        T curr = static_cast<T>(lvals[i]);
        ovals[i] = static_cast<T>(c_invvar * (curr - mu));
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
  }
}

template <typename T>
void HostApplyLayerNorm(
    T* output,
    T* mean,
    T* invvar,
    const T* input,
    int64_t n1,
    int64_t n2,
    T epsilon,
    const T* gamma,
    const T* beta) {
  const dim3 threads(GPU_WARP_SIZE, 4, 1);
  const dim3 blocks(1, std::min((uint64_t)n1, MAX_GRID_Y), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(T) + (threads.y / 2) * sizeof(T) : 0;
  cuApplyLayerNorm<<<blocks, threads, nshared, 0>>>(
      output,
      mean,
      invvar,
      input,
      n1, n2,
      epsilon,
      gamma, beta);
}

)",
                 "");

LU_DEFINE(declaration::ort_layer_norm, R"(
__device__ inline half2 AddHalf2(const half2 a, const half2 b) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return __hadd2(a, b);
#else
  return __halves2half2(__hadd(a.x, b.x), __hadd(a.y, b.y));
#endif
}

struct KeyValuePairSum {
  __device__ inline cub::KeyValuePair<float, float> operator()(const cub::KeyValuePair<float, float>& a, const cub::KeyValuePair<float, float>& b) {
    return cub::KeyValuePair<float, float>(a.key + b.key, a.value + b.value);
  }

  __device__ inline cub::KeyValuePair<half, half> operator()(const cub::KeyValuePair<half, half>& a, const cub::KeyValuePair<half, half>& b) {
    const half2 a2 = __halves2half2(a.key, a.value);
    const half2 b2 = __halves2half2(b.key, b.value);
    const half2 res = AddHalf2(a2, b2);
    return cub::KeyValuePair<half, half>(res.x, res.y);
  }

  __device__ inline cub::KeyValuePair<half2, half2> operator()(const cub::KeyValuePair<half2, half2>& a, const cub::KeyValuePair<half2, half2>& b) {
    return cub::KeyValuePair<half2, half2>(AddHalf2(a.key, b.key), AddHalf2(a.value, b.value));
  }
};

template <typename T, int TPB>
__device__ inline void LayerNorm(
    const cub::KeyValuePair<T, T>& thread_data, const int ld, const int offset, const T* beta,
    const T* gamma, const T epsilon, T* output) {
  // Assuming thread_data is already divided by ld

  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<T, T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  KeyValuePairSum pair_sum;
  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = Rsqrt(sum_kv.value - mu * mu + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = output[idx];
    const T g(gamma[i]);
    const T b(beta[i]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, int TPB>
__device__ inline void LayerNormSmall(const T val, const cub::KeyValuePair<T, T>& thread_data, const int ld, const int idx,
                                      const T* beta, const T* gamma, const T epsilon, T* output) {
  // Assuming thread_data is already divided by ld
  // Small settings: the block covers the leading dimension TPB >= ld. The input
  // value is available in a register

  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<T, T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  KeyValuePairSum pair_sum;
  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = Rsqrt(sum_kv.value - mu * mu + epsilon);
  }
  __syncthreads();

  if (threadIdx.x < ld) {
    const T g(gamma[threadIdx.x]);
    const T b(beta[threadIdx.x]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

)");

LU_DEFINE(declaration::ort_qkv_to_context, R"(
template <class INT, class INT2>
static INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

static size_t AlignTo(size_t a, size_t b) {
  return CeilDiv(a, b) * b;
}

size_t ScratchSize(size_t element_size, int batch_size, int num_heads, int sequence_length, int all_sequence_length) {
  const size_t len = batch_size * num_heads * sequence_length * all_sequence_length;
  const size_t bytes = len * element_size;

  const size_t alignment = 256;
  const size_t bytesAligned = AlignTo(bytes, alignment);
  return bytesAligned;
}

template <typename T, unsigned TPB>
__device__ inline void Softmax(const int all_sequence_length,
                               const int sequence_length,
                               const int valid_end,
                               const int valid_start,
                               const T* input,
                               T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  float thread_data_max(-CUDART_INF_F);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      const int index = offset + i;
      if (thread_data_max < float(input[index])) {
        thread_data_max = float(input[index]);
      }
    }
  }

  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max());

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_sum(0.f);
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      const int index = offset + i;
      const float val = input[index];
      thread_data_sum += expf(val - max_block);
    }
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_sum, cub::Sum());
  if (threadIdx.x == 0) {
    sum_reverse_block = 1.f / sum;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < all_sequence_length; i += TPB) {
    const int index = offset + i;
    const float val = (i >= valid_start && i < valid_end) ? expf(float(input[index]) - max_block) * sum_reverse_block : 0.f;
    output[index] = T(val);
  }
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxSmall(const int all_sequence_length,
                                    const int sequence_length,
                                    const int valid_end,
                                    const int valid_start,
                                    const T* input,
                                    T* output,
                                    bool is_unidirectional) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  const int index = offset + threadIdx.x;

  bool is_valid = false;  // whether it has attention mask == 1.

  // Update end position for unidirectional.
  int end = valid_end;
  if (is_unidirectional) {
    int end_unid = all_sequence_length - sequence_length + (blockIdx.x % sequence_length) + 1;
    if (end_unid <= valid_start) {
      // In this situation, mask of [0, end_unid) and [valid_start, valid_end) has -10000, and [end_unid, valid_start) and [valid_end, all_seq_len) has -20000.
      // So [0, end_unid) will also have value after softmax.
      is_valid = threadIdx.x < end_unid;
    } else {
      end = min(valid_end, end_unid);
    }
  }

  is_valid = is_valid || (threadIdx.x >= valid_start && threadIdx.x < end);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  float thread_data_max = is_valid ? float(input[index]) : float(-CUDART_INF_F);
  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max(), end);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp(0.f);
  if (is_valid) {
    thread_data_exp = expf(float(input[index]) - max_block);
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), end);

  // Store value of 1.0/sum.
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  // threadIdx.x might be larger than all_sequence_length due to alignment to 32x.
  if (threadIdx.x < all_sequence_length) {
    output[index] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxWithMask2DSmall(const int all_sequence_length,
                                              const int sequence_length,
                                              const int* attention_mask,  // 2D attention mask
                                              const T* input,
                                              T* output,
                                              const bool is_unidirectional,
                                              const float scalar) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  int index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length + threadIdx.x;

  float thread_data = -CUDART_INF_F;
  if (threadIdx.x < all_sequence_length) {
    const int& mask = attention_mask[blockIdx.y * all_sequence_length + threadIdx.x];
    float mask_value = mask > 0 ? 0.0f : -10000.0f;

    if (is_unidirectional) {
      int from_index = all_sequence_length - sequence_length + (blockIdx.x % sequence_length);  // offset of from token in all sequence length.
      if (threadIdx.x > from_index) {
        mask_value += -10000.0f;
      }
    }

    thread_data = float(input[index]) * scalar + mask_value;
  }

  const float max = BlockReduce(tmp_storage).Reduce(thread_data, cub::Max(), all_sequence_length);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp = threadIdx.x < all_sequence_length ? expf(thread_data - max_block) : 0.0f;
  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), all_sequence_length);

  // Store value of 1.0/sum
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  if (threadIdx.x < all_sequence_length) {
    output[index] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernelSmall(const int all_sequence_length, const int sequence_length, const T* input, T* output, bool is_unidirectional) {
  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, all_sequence_length, 0, input, output, is_unidirectional);
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernel(const int all_sequence_length, const int sequence_length, const T* input, T* output) {
  Softmax<T, TPB>(all_sequence_length, sequence_length, all_sequence_length, 0, input, output);
}

template <typename T>
void ComputeSoftmax(
    cudaStream_t stream, const int all_sequence_length, const int sequence_length, const int batch_size, const int num_heads,
    const T* input, T* output, bool is_unidirectional) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);
  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output, is_unidirectional);
  } else if (!is_unidirectional) {
    const int blockSize = 1024;
    SoftmaxKernel<T, blockSize><<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, input, output);
  }
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernelSmall(const int all_sequence_length, const int sequence_length, const int* mask_end, const int* mask_start, const T* input, T* output, bool is_unidirectional) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    start_position = mask_start != nullptr ? max(0, mask_start[batch]) : 0;
    end_position = min(all_sequence_length, mask_end[batch]);

    // Attend to no word has same effect as attend to all words. This is added to get parity with CPU result.
    if (start_position >= end_position) {
      start_position = 0;
      end_position = all_sequence_length;
    }
  }
  __syncthreads();

  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, end_position, start_position, input, output, is_unidirectional);
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernel(const int all_sequence_length, const int sequence_length, const int* mask_end, const int* mask_start, const T* input, T* output) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    start_position = mask_start != nullptr ? max(0, mask_start[batch]) : 0;
    end_position = min(all_sequence_length, mask_end[batch]);

    // Attend to no word has same effect as attend to all words. This is added to get parity with CPU result.
    if (start_position >= end_position) {
      start_position = 0;
      end_position = all_sequence_length;
    }
  }
  __syncthreads();

  Softmax<T, TPB>(all_sequence_length, sequence_length, end_position, start_position, input, output);
}

template <typename T, unsigned TPB>
__global__ void SoftmaxWithMask2DSmallKernel(const int all_sequence_length, const int sequence_length, const int* attention_mask, const T* input, T* output, const bool is_unidirectional, const float scalar) {
  SoftmaxWithMask2DSmall<T, TPB>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
}

template <typename T>
void ComputeSoftmaxWithMask1D(cudaStream_t stream, const int all_sequence_length, const int sequence_length, const int batch_size, const int num_heads,
                              const int* mask_index, const int* mask_start, const T* input, T* output, const bool is_unidirectional) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    MaskedSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output, is_unidirectional);
  } else if (!is_unidirectional) {
    const int blockSize = 1024;
    MaskedSoftmaxKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start, input, output);
  }
}

template <typename T>
void ComputeSoftmaxWithMask2D(cudaStream_t stream, const int all_sequence_length, const int sequence_length, const int batch_size, const int num_heads,
                              const int* attention_mask, const T* input, T* output, const bool is_unidirectional, const float scalar) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    SoftmaxWithMask2DSmallKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, attention_mask, input, output, is_unidirectional, scalar);
  }
}

template <typename T>
__global__ void TransposeCtx(const int H, const T* input, T* output) {
  // Input:  BxNxSxH
  // Output: BxSxNxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  const int in_offset = s * H + n * sequence_length * H + b * NHS;
  const int out_offset = n * H + s * NH + b * NHS;

  const int i = threadIdx.x;
  if (i < H) {
    output[out_offset + i] = input[in_offset + i];
  }
}

void LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const float* input, float* output) {
  const dim3 grid(sequence_length, batch_size, 1);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    const dim3 block(H, num_heads, 1);
    TransposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {
    const dim3 block(head_size, num_heads, 1);
    TransposeCtx<float><<<grid, block, 0, stream>>>(head_size, input, output);
  }
}

void LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const half* input, half* output) {
  const dim3 grid(sequence_length, batch_size, 1);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const dim3 block(H, num_heads, 1);
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    TransposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const dim3 block(H, num_heads, 1);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    TransposeCtx<half2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    const dim3 block(head_size, num_heads, 1);
    TransposeCtx<half><<<grid, block, 0, stream>>>(head_size, input, output);
  }
}

template <typename T>
__global__ void TransposeQKV(const int H, const T* input, T* output) {
  // Input:  BxSx3xNxH
  // Output: 3xBxNxSxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
  const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  const int i = threadIdx.x;
  if (i < H) {
    output[out_offset + i] = input[in_offset + i];
  }
}

void LaunchTransQkv(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const float* input, float* output) {
  const dim3 grid(sequence_length, batch_size, 3);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    const dim3 block(H, num_heads, 1);
    TransposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {
    const dim3 block(head_size, num_heads, 1);
    TransposeQKV<float><<<grid, block, 0, stream>>>(head_size, input, output);
  }
}

void LaunchTransQkv(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const half* input, half* output) {
  const dim3 grid(sequence_length, batch_size, 3);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const dim3 block(H, num_heads, 1);
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    TransposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const dim3 block(H, num_heads, 1);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    TransposeQKV<half2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel..
    const dim3 block(head_size, num_heads, 1);
    TransposeQKV<half><<<grid, block, 0, stream>>>(head_size, input, output);
  }
}

template <typename T>
__global__ void ConcatPastToPresent(const int sequence_length,
                                    const T* past,
                                    const T* k_v,
                                    T* present) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int is_v = blockIdx.z;  // 0 for k, 1 for v

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  // past:    2 x BxNxS'xH   (past_k and past_v)
  // k_v:     2 x BxNxSxH    (k and v)
  // present: 2 x BxNxS*xH   (present_k and present_v)
  const int past_sequence_length = all_sequence_length - sequence_length;

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  int out_offset = b * present_NSH + n * present_SH + s * H + h + is_v * (present_NSH * batch_size);
  if (s < past_sequence_length) {
    const int past_SH = past_sequence_length * H;
    const int past_NSH = num_heads * past_SH;
    const int in_offset = b * past_NSH + n * past_SH + s * H + h + is_v * (past_NSH * batch_size);
    present[out_offset] = past[in_offset];
  } else if (s < all_sequence_length) {
    const int SH = sequence_length * H;
    const int NSH = num_heads * SH;
    const int in_offset = b * NSH + n * SH + (s - past_sequence_length) * H + h + is_v * (NSH * batch_size);
    present[out_offset] = k_v[in_offset];
  }
}

void LaunchConcatPastToPresent(cudaStream_t stream,
                               const int all_sequence_length,
                               const int sequence_length,
                               const int batch_size,
                               const int head_size,
                               const int num_heads,
                               const float* past,
                               const float* k_v,
                               float* present) {
  const dim3 grid(all_sequence_length, batch_size, 2);
  if (0 == (head_size & 1)) {
    const dim3 block(head_size / 2, num_heads, 1);
    ConcatPastToPresent<float2><<<grid, block, 0, stream>>>(sequence_length, reinterpret_cast<const float2*>(past), reinterpret_cast<const float2*>(k_v), reinterpret_cast<float2*>(present));
  } else {
    const dim3 block(head_size, num_heads, 1);
    ConcatPastToPresent<float><<<grid, block, 0, stream>>>(sequence_length, past, k_v, present);
  }
}

void LaunchConcatPastToPresent(cudaStream_t stream,
                               const int all_sequence_length,
                               const int sequence_length,
                               const int batch_size,
                               const int head_size,
                               const int num_heads,
                               const half* past,
                               const half* k_v,
                               half* present) {
  const dim3 grid(all_sequence_length, batch_size, 2);
  if (0 == (head_size % 4)) {
    const dim3 block(head_size / 4, num_heads, 1);
    ConcatPastToPresent<float2><<<grid, block, 0, stream>>>(sequence_length, reinterpret_cast<const float2*>(past), reinterpret_cast<const float2*>(k_v), reinterpret_cast<float2*>(present));
  } else if (0 == (head_size & 1)) {
    const dim3 block(head_size / 2, num_heads, 1);
    ConcatPastToPresent<half2><<<grid, block, 0, stream>>>(sequence_length, reinterpret_cast<const half2*>(past), reinterpret_cast<const half2*>(k_v), reinterpret_cast<half2*>(present));
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    const dim3 block(head_size, num_heads, 1);
    ConcatPastToPresent<half><<<grid, block, 0, stream>>>(sequence_length, past, k_v, present);
  }
}

void inline CublasGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float alpha,
    const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB,
    const float beta, float* C, int ldc, long long int strideC, int batchCount) {
  cublasSgemmStridedBatched(
      handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

void inline CublasGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const half alpha,
    const half* A, int lda, long long int strideA, const half* B, int ldb, long long int strideB,
    const half beta, half* C, int ldc, long long int strideC, int batchCount) {
  cublasHgemmStridedBatched(
      handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

template <typename T>
void QkvToContext(
    cublasHandle_t cublas, cudaStream_t stream,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size, const size_t element_size,
    const T* input, T* output, T* workspace,
    const int* mask_index,
    bool is_unidirectional, int past_sequence_length, const T* past, T* present, bool use_2d_attention_mask, const int* mask_start) {
  const int all_sequence_length = past_sequence_length + sequence_length;
  const size_t bytes = ScratchSize(element_size, batch_size, num_heads, sequence_length, all_sequence_length);
  T* scratch1 = workspace;
  T* scratch2 = scratch1 + (bytes / element_size);
  T* scratch3 = scratch2 + (bytes / element_size);

  // input should be BxSx3xNxH => scratch3: 3xBxNxSxH
  LaunchTransQkv(stream, sequence_length, batch_size, head_size, num_heads, input, scratch3);

  // now scratch3 has Q, K, V: each has size BxNxSxH
  const int batches = batch_size * num_heads;
  const int size_per_batch = sequence_length * head_size;
  const int total_size = batches * size_per_batch;

  const T* q = scratch3;
  const T* k = q + total_size;
  const T* v = k + total_size;

  cublasSetStream(cublas, stream);
  cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);

  // Concat past (2xBxNxS'xH) to present (2xBxNxS*xH):
  // past_k (BxNxS'xH) + k (BxNxSxH) => present_k (BxNxS*xH)
  // past_v (BxNxS'xH) + v (BxNxSxH) => present_v (BxNxS*xH)
  const int present_size_per_batch = all_sequence_length * head_size;
  if (nullptr != present) {
    LaunchConcatPastToPresent(stream, all_sequence_length, sequence_length, batch_size, head_size, num_heads, past, k, present);
    // update pointers to present_k and present_v.
    k = present;
    v = present + batches * present_size_per_batch;
  }

  // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxS*
  // Q: BxNxSxH, K (present_k): BxNxS*xH, Q*K': BxNxSxS*
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));
  const int temp_matrix_size = sequence_length * all_sequence_length;
  T alpha = (T)(use_2d_attention_mask ? 1.0f : rsqrt_head_size);
  CublasGemmStridedBatched(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, all_sequence_length, sequence_length, head_size, alpha, k, head_size, present_size_per_batch,
          q, head_size, size_per_batch, 0.f, scratch1, all_sequence_length, temp_matrix_size, batches);

  // apply softmax and store result P to scratch2: BxNxSxS*
  if (use_2d_attention_mask) {  // 2d attention mask
    ComputeSoftmaxWithMask2D<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads, mask_index, scratch1, scratch2, is_unidirectional, rsqrt_head_size);
  } else if (nullptr != mask_index) {  // 1d mask index
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    ComputeSoftmaxWithMask1D<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads, mask_index, mask_start, scratch1, scratch2, is_unidirectional);
  } else {  // no mask
    ComputeSoftmax<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads, scratch1, scratch2, is_unidirectional);
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  CublasGemmStridedBatched(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, head_size, sequence_length, all_sequence_length, 1.f, v, head_size, present_size_per_batch,
          scratch2, all_sequence_length, temp_matrix_size, 0.f, scratch3, head_size, size_per_batch, batches);

  // scratch3 is BxNxSxH, transpose to output BxSxNxH
  LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads, scratch3, output);
}
)");

LU_DEFINE(declaration::math_Rsqrt, R"(
template <typename T>
__device__ inline T Rsqrt(const T& x);

template <>
__device__ inline float Rsqrt(const float& x) {
  return rsqrtf(x);
}

template <>
__device__ inline half Rsqrt(const half& x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return hrsqrt(x);
#else
  return half(rsqrtf(float(x)));
#endif
}
)");

LU_DEFINE(declaration::math_Gelu, R"(
template <typename T>
__device__ __inline__ T _Normcdf(T a);

template <>
__device__ __inline__ float _Normcdf(float a) { return normcdff(a); }

template <>
__device__ __inline__ double _Normcdf(double a) { return normcdf(a); }

template <>
__device__ __inline__ half _Normcdf(half a) { return half(normcdff((float)a)); }

template <typename T>
__device__ __inline__ T _Gelu(T a) {
  return a * _Normcdf(a);
}
)");

LU_DEFINE_EXTEND(declaration::ort_softmax,
                 R"(
inline int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
  ReduceOp<acc_t> r;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
      sum[i] = r(sum[i], b);
    }
  }
}

/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/* Modifications Copyright (c) Microsoft. */

// The code below(from the definition of softmax_warp_forward to the definition of dispatch_softmax_forward) 
// is mostly copied from Pytorch PersistentSoftmax.cuh

// The softmax_warp_* methods perform softmax forward and backward propagation on samples spanning the fast dimension.
// Each sample contains element_count scalar elements. element_count can be any integer value <= 1024.
// The template arguments have the following meaning:
// One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples.
// WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small.
// A "WARP" contains "GPU_WARP_SIZE" threads, these treads are guaranteed to belong to the same warp.
// This is important because it means only __shfl_ instructions are required for reductions.
// Note that this means WARP_SIZE must be a power of two and <= architecture warp size.
// CUDA warp size is 32 for all existing GPU architecures, but there is no guarantee this will not change for future arch.
// is_log_softmax is a flag indicating whether SoftMax or LogSoftMax should be computed.
// The template can be instantiated with any floating point type for the type arguments input_t, output_t and acc_t.
// This allows SoftMax to be fused with a cast immediately following the SoftMax.
// For instance:
// input_t=half,  acc_t=float, output_t=half  => read half tensor, float accumulators, write half tensor.
// input_t=half,  acc_t=float, output_t=float => read half tensor, float accumulators, write float tensor.
// input_t_float, acc_t=float, output_t=half  => read float tensor, float accumulators, write half tensor.

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_forward(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the batch
  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
  // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
  // the nested loops.
  // This should have no impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] = src[i * element_count + it * WARP_SIZE];
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  // compute max_value
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      if (is_log_softmax) {
        sum[i] += std::exp((float)(elements[i][it] - max_value[i]));
      } else {
        elements[i][it] = std::exp((float)(elements[i][it] - max_value[i]));
        sum[i] += elements[i][it];
      }
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
    if (is_log_softmax) sum[i] = max_value[i] + std::log((float)(sum[i]));
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        if (is_log_softmax) {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] - sum[i];
        } else {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        }
      } else {
        break;
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_forward(cudaStream_t stream, output_t* dst, const input_t* src, int softmax_elements, int softmax_elements_stride, int batch_count) {
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        softmax_warp_forward<input_t, output_t, acc_t, 0, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 1:  // 2
        softmax_warp_forward<input_t, output_t, acc_t, 1, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 2:  // 4
        softmax_warp_forward<input_t, output_t, acc_t, 2, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 3:  // 8
        softmax_warp_forward<input_t, output_t, acc_t, 3, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 4:  // 16
        softmax_warp_forward<input_t, output_t, acc_t, 4, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 5:  // 32
        softmax_warp_forward<input_t, output_t, acc_t, 5, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 6:  // 64
        softmax_warp_forward<input_t, output_t, acc_t, 6, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 7:  // 128
        softmax_warp_forward<input_t, output_t, acc_t, 7, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 8:  // 256
        softmax_warp_forward<input_t, output_t, acc_t, 8, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 9:  // 512
        softmax_warp_forward<input_t, output_t, acc_t, 9, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 10:  // 1024
        softmax_warp_forward<input_t, output_t, acc_t, 10, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      default:
        break;
    }
  }
}
)",
                 R"(
inline int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
  ReduceOp<acc_t> r;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
      sum[i] = r(sum[i], b);
    }
  }
}

/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/* Modifications Copyright (c) Microsoft. */

// The code below(from the definition of softmax_warp_forward to the definition of dispatch_softmax_forward) 
// is mostly copied from Pytorch PersistentSoftmax.cuh

// The softmax_warp_* methods perform softmax forward and backward propagation on samples spanning the fast dimension.
// Each sample contains element_count scalar elements. element_count can be any integer value <= 1024.
// The template arguments have the following meaning:
// One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples.
// WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small.
// A "WARP" contains "GPU_WARP_SIZE" threads, these treads are guaranteed to belong to the same warp.
// This is important because it means only __shfl_ instructions are required for reductions.
// Note that this means WARP_SIZE must be a power of two and <= architecture warp size.
// CUDA warp size is 32 for all existing GPU architecures, but there is no guarantee this will not change for future arch.
// is_log_softmax is a flag indicating whether SoftMax or LogSoftMax should be computed.
// The template can be instantiated with any floating point type for the type arguments input_t, output_t and acc_t.
// This allows SoftMax to be fused with a cast immediately following the SoftMax.
// For instance:
// input_t=half,  acc_t=float, output_t=half  => read half tensor, float accumulators, write half tensor.
// input_t=half,  acc_t=float, output_t=float => read half tensor, float accumulators, write float tensor.
// input_t_float, acc_t=float, output_t=half  => read float tensor, float accumulators, write half tensor.

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_forward(output_t* dst, const input_t* src, int batch_size, int stride, int element_count) {
  // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the batch
  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
  // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
  // the nested loops.
  // This should have no impact on performance because the loops are unrolled anyway.

  // load data from global memory
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] = src[i * element_count + it * WARP_SIZE];
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  // compute max_value
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      if (is_log_softmax) {
        sum[i] += std::exp((float)(elements[i][it] - max_value[i]));
      } else {
        elements[i][it] = std::exp((float)(elements[i][it] - max_value[i]));
        sum[i] += elements[i][it];
      }
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
    if (is_log_softmax) sum[i] = max_value[i] + std::log((float)(sum[i]));
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        if (is_log_softmax) {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] - sum[i];
        } else {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        }
      } else {
        break;
      }
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_forward(cudaStream_t stream, output_t* dst, const input_t* src, int softmax_elements, int softmax_elements_stride, int batch_count) {
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
      case 0:  // 1
        softmax_warp_forward<input_t, output_t, acc_t, 0, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 1:  // 2
        softmax_warp_forward<input_t, output_t, acc_t, 1, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 2:  // 4
        softmax_warp_forward<input_t, output_t, acc_t, 2, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 3:  // 8
        softmax_warp_forward<input_t, output_t, acc_t, 3, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 4:  // 16
        softmax_warp_forward<input_t, output_t, acc_t, 4, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 5:  // 32
        softmax_warp_forward<input_t, output_t, acc_t, 5, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 6:  // 64
        softmax_warp_forward<input_t, output_t, acc_t, 6, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 7:  // 128
        softmax_warp_forward<input_t, output_t, acc_t, 7, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 8:  // 256
        softmax_warp_forward<input_t, output_t, acc_t, 8, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 9:  // 512
        softmax_warp_forward<input_t, output_t, acc_t, 9, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      case 10:  // 1024
        softmax_warp_forward<input_t, output_t, acc_t, 10, is_log_softmax>
            <<<blocks, threads, 0, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        break;
      default:
        break;
    }
  }
}
)",
                 "");

LU_DEFINE_EXTEND(declaration::warp,
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
)",
                 R"(
// Check compute capability
#if !defined(CONSTANT_VAR)
#define CONSTANT_VAR 1
const int GPU_WARP_SIZE = 32;
const uint64_t MAX_GRID_Y = 65535;
#endif

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
)",
                 "");

LU_DEFINE(declaration::mem_eff_attn, R"(

void mem_eff_attention_1(void* output,
                   void* query,
                   void* key,
                   void* value,
                   float* accum_ptr,
                   long long* batch_size,
                   int seq_len,
                   int seq_len_kv,
                   int num_heads,
                   int head_size,
                   int head_size_v,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
                   cudaStream_t stream)
    
{

    /*
    problem_sizes0 [b, m, n, k]
    [head_number * batch_size, m, mkv, k0]
    [head_number * batch_size, seq_length, seq_length_kv, head_size]

    problem_sizes1
    [head_number * batch_size, m, k1, mkv]
    [head_number * batch_size, seq_length, head_size_v, seq_length_kv]

    m = seq_len
    n = seq_len
    k = head_size

    Q: B, M, K
    K: B, N, K
    P: B, M, N
    V: B, N, K
    O: B, M, K
    output: bs, num_head, seq_len, head_size
    */


    using ArchTag = cutlass::arch::Sm80;
    constexpr bool kIs64x64 = true;
    constexpr bool kSingleValueIteration = true;

    // Set grid size
    constexpr long long kQueriesPerBlock = kIs64x64 ? 64 : 32;
    constexpr long long kKeysPerBlock = kIs64x64 ? 64 : 128;
    if (kIs64x64 && head_size_v > kKeysPerBlock) {
        std::cerr << "WARNING: you will get better performance with `kIs64x64=false`";
    }
    if (kSingleValueIteration && head_size_v > kKeysPerBlock) {
        std::cerr << "ERROR  : Use kSingleValueIteration to keep output in RF. "         "This requires to have `head_size <= kKeysPerBlock` "         "but head_size_v=" << head_size_v << " and kKeysPerBlock=" << kKeysPerBlock << "";
        return;
    }
    if (!kSingleValueIteration && head_size_v <= kKeysPerBlock) {
       std::cerr << "WARNING: you will get better performance with `kSingleValueIteration=true'";
       }
    

    using Attention = AttentionKernel<
        cutlass::half_t, // scalar_t
        ArchTag,
        true, // memory is aligned
        kQueriesPerBlock,
        kKeysPerBlock,
        kSingleValueIteration
    >;

    int block_O_size = (*batch_size) * seq_len * num_heads * head_size_v;
    typename Attention::Params p;
    {
        // set parameters
        p.query_ptr = static_cast<cutlass::half_t*>(query);
        p.key_ptr = static_cast<cutlass::half_t*>(key);
        p.value_ptr = static_cast<cutlass::half_t*>(value);
        p.logsumexp_ptr = nullptr; // Only needed for bw
        p.output_accum_ptr = nullptr;
        if (Attention::kNeedsOutputAccumulatorBuffer) {
          p.output_accum_ptr = accum_ptr;
        }
        p.output_ptr = static_cast<cutlass::half_t*>(output);

        p.num_heads = num_heads;
        p.num_batches = *batch_size;
        p.head_dim = head_size;
        p.head_dim_value = head_size_v;
        p.num_queries = seq_len;
        p.num_keys = seq_len_kv;
        p.causal = is_causal;


        p.q_strideM = head_size;
        p.k_strideM = head_size;
        p.v_strideM = head_size_v;

        p.q_strideH = p.q_strideM * seq_len;
        p.k_strideH = p.k_strideM * seq_len_kv;
        p.v_strideH = p.v_strideM * seq_len_kv;
        p.o_strideH = head_size_v;
        p.q_strideB = p.q_strideH * num_heads;
        p.k_strideB = p.k_strideH * num_heads;
        p.v_strideB = p.v_strideH * num_heads;
        p.o_strideB = head_size_v * seq_len * num_heads;
    }

    // launch kernel
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      std::cerr << "Kernel does not support these inputs" << std::endl;
      return;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    // cudaError_t err = cudaDeviceSynchronize();

    // if (err != cudaSuccess)  {
    //   std::cerr << "Kernel execution error: " << cudaGetErrorString(err);
    //   return;
    // }
}


void mem_eff_attention_2(void* output,
                   void* query,
                   void* key,
                   void* value,
                   float* accum_ptr,
                   long long* batch_size,
                   int seq_len,
                   int seq_len_kv,
                   int num_heads,
                   int head_size,
                   int head_size_v,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
                   cudaStream_t stream)
    
{
    using ArchTag = cutlass::arch::Sm80;
    constexpr bool kIs64x64 = true;
    constexpr bool kSingleValueIteration = false;

    // Set grid size
    constexpr long long kQueriesPerBlock = kIs64x64 ? 64 : 32;
    constexpr long long kKeysPerBlock = kIs64x64 ? 64 : 128;
    if (kIs64x64 && head_size_v > kKeysPerBlock) {
        std::cerr << "WARNING: you will get better performance with `kIs64x64=false`";
    }
    if (kSingleValueIteration && head_size_v > kKeysPerBlock) {
        std::cerr << "ERROR  : Use kSingleValueIteration to keep output in RF. "         "This requires to have `head_size <= kKeysPerBlock` "         "but head_size_v=" << head_size_v << " and kKeysPerBlock=" << kKeysPerBlock << "";
        return;
    }
    if (!kSingleValueIteration && head_size_v <= kKeysPerBlock) {
       std::cerr << "WARNING: you will get better performance with `kSingleValueIteration=true'";
       }
    

    using Attention = AttentionKernel<
        cutlass::half_t, // scalar_t
        ArchTag,
        true, // memory is aligned
        kQueriesPerBlock,
        kKeysPerBlock,
        kSingleValueIteration
    >;

    int block_O_size = (*batch_size) * seq_len * num_heads * head_size_v;
    typename Attention::Params p;
    {
        // set parameters
        p.query_ptr = static_cast<cutlass::half_t*>(query);
        p.key_ptr = static_cast<cutlass::half_t*>(key);
        p.value_ptr = static_cast<cutlass::half_t*>(value);
        p.logsumexp_ptr = nullptr; // Only needed for bw
        p.output_accum_ptr = nullptr;
        if (Attention::kNeedsOutputAccumulatorBuffer) {
          p.output_accum_ptr = accum_ptr;
        }
        p.output_ptr = static_cast<cutlass::half_t*>(output);

        p.num_heads = num_heads;
        p.num_batches = *batch_size;
        p.head_dim = head_size;
        p.head_dim_value = head_size_v;
        p.num_queries = seq_len;
        p.num_keys = seq_len_kv;
        p.causal = is_causal;


        p.q_strideM = head_size;
        p.k_strideM = head_size;
        p.v_strideM = head_size_v;

        p.q_strideH = p.q_strideM * seq_len;
        p.k_strideH = p.k_strideM * seq_len_kv;
        p.v_strideH = p.v_strideM * seq_len_kv;
        p.o_strideH = head_size_v;
        p.q_strideB = p.q_strideH * num_heads;
        p.k_strideB = p.k_strideH * num_heads;
        p.v_strideB = p.v_strideH * num_heads;
        p.o_strideB = head_size_v * seq_len * num_heads;
    }

    // launch kernel
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      std::cerr << "Kernel does not support these inputs" << std::endl;
      return;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    // cudaError_t err = cudaDeviceSynchronize();

    // if (err != cudaSuccess)  {
    //   std::cerr << "Kernel execution error: " << cudaGetErrorString(err);
    //   return;
    // }
}

void mem_eff_attention_3(void* output,
                   void* query,
                   void* key,
                   void* value,
                   float* accum_ptr,
                   long long* batch_size,
                   int seq_len,
                   int seq_len_kv,
                   int num_heads,
                   int head_size,
                   int head_size_v,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
                   cudaStream_t stream)
    
{
    using ArchTag = cutlass::arch::Sm80;
    constexpr bool kIs64x64 = false;
    constexpr bool kSingleValueIteration = true;

    // Set grid size
    constexpr long long kQueriesPerBlock = kIs64x64 ? 64 : 32;
    constexpr long long kKeysPerBlock = kIs64x64 ? 64 : 128;
    if (kIs64x64 && head_size_v > kKeysPerBlock) {
        std::cerr << "WARNING: you will get better performance with `kIs64x64=false`";
    }
    if (kSingleValueIteration && head_size_v > kKeysPerBlock) {
        std::cerr << "ERROR  : Use kSingleValueIteration to keep output in RF. "         "This requires to have `head_size <= kKeysPerBlock` "         "but head_size_v=" << head_size_v << " and kKeysPerBlock=" << kKeysPerBlock << "";
        return;
    }
    if (!kSingleValueIteration && head_size_v <= kKeysPerBlock) {
       std::cerr << "WARNING: you will get better performance with `kSingleValueIteration=true'";
       }
    

    using Attention = AttentionKernel<
        cutlass::half_t, // scalar_t
        ArchTag,
        true, // memory is aligned
        kQueriesPerBlock,
        kKeysPerBlock,
        kSingleValueIteration
    >;

    int block_O_size = (*batch_size) * seq_len * num_heads * head_size_v;
    typename Attention::Params p;
    {
        // set parameters
        p.query_ptr = static_cast<cutlass::half_t*>(query);
        p.key_ptr = static_cast<cutlass::half_t*>(key);
        p.value_ptr = static_cast<cutlass::half_t*>(value);
        p.logsumexp_ptr = nullptr; // Only needed for bw
        p.output_accum_ptr = nullptr;
        if (Attention::kNeedsOutputAccumulatorBuffer) {
          p.output_accum_ptr = accum_ptr;
        }
        p.output_ptr = static_cast<cutlass::half_t*>(output);

        p.num_heads = num_heads;
        p.num_batches = *batch_size;
        p.head_dim = head_size;
        p.head_dim_value = head_size_v;
        p.num_queries = seq_len;
        p.num_keys = seq_len_kv;
        p.causal = is_causal;


        p.q_strideM = head_size;
        p.k_strideM = head_size;
        p.v_strideM = head_size_v;

        p.q_strideH = p.q_strideM * seq_len;
        p.k_strideH = p.k_strideM * seq_len_kv;
        p.v_strideH = p.v_strideM * seq_len_kv;
        p.o_strideH = head_size_v;
        p.q_strideB = p.q_strideH * num_heads;
        p.k_strideB = p.k_strideH * num_heads;
        p.v_strideB = p.v_strideH * num_heads;
        p.o_strideB = head_size_v * seq_len * num_heads;
    }

    // launch kernel
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      std::cerr << "Kernel does not support these inputs" << std::endl;
      return;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    // cudaError_t err = cudaDeviceSynchronize();

    // if (err != cudaSuccess)  {
    //   std::cerr << "Kernel execution error: " << cudaGetErrorString(err);
    //   return;
    // }
}

void mem_eff_attention_4(void* output,
                   void* query,
                   void* key,
                   void* value,
                   float* accum_ptr,
                   long long* batch_size,
                   int seq_len,
                   int seq_len_kv,
                   int num_heads,
                   int head_size,
                   int head_size_v,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
                   cudaStream_t stream)
    
{
    using ArchTag = cutlass::arch::Sm80;
    constexpr bool kIs64x64 = false;
    constexpr bool kSingleValueIteration = false;

    // Set grid size
    constexpr long long kQueriesPerBlock = kIs64x64 ? 64 : 32;
    constexpr long long kKeysPerBlock = kIs64x64 ? 64 : 128;
    if (kIs64x64 && head_size_v > kKeysPerBlock) {
        std::cerr << "WARNING: you will get better performance with `kIs64x64=false`";
    }
    if (kSingleValueIteration && head_size_v > kKeysPerBlock) {
        std::cerr << "ERROR  : Use kSingleValueIteration to keep output in RF. "         "This requires to have `head_size <= kKeysPerBlock` "         "but head_size_v=" << head_size_v << " and kKeysPerBlock=" << kKeysPerBlock << "";
        return;
    }
    if (!kSingleValueIteration && head_size_v <= kKeysPerBlock) {
       std::cerr << "WARNING: you will get better performance with `kSingleValueIteration=true'";
       }
    

    using Attention = AttentionKernel<
        cutlass::half_t, // scalar_t
        ArchTag,
        true, // memory is aligned
        kQueriesPerBlock,
        kKeysPerBlock,
        kSingleValueIteration
    >;

    int block_O_size = (*batch_size) * seq_len * num_heads * head_size_v;
    typename Attention::Params p;
    {
        // set parameters
        p.query_ptr = static_cast<cutlass::half_t*>(query);
        p.key_ptr = static_cast<cutlass::half_t*>(key);
        p.value_ptr = static_cast<cutlass::half_t*>(value);
        p.logsumexp_ptr = nullptr; // Only needed for bw
        p.output_accum_ptr = nullptr;
        if (Attention::kNeedsOutputAccumulatorBuffer) {
          p.output_accum_ptr = accum_ptr;
        }
        p.output_ptr = static_cast<cutlass::half_t*>(output);

        p.num_heads = num_heads;
        p.num_batches = *batch_size;
        p.head_dim = head_size;
        p.head_dim_value = head_size_v;
        p.num_queries = seq_len;
        p.num_keys = seq_len_kv;
        p.causal = is_causal;


        p.q_strideM = head_size;
        p.k_strideM = head_size;
        p.v_strideM = head_size_v;

        p.q_strideH = p.q_strideM * seq_len;
        p.k_strideH = p.k_strideM * seq_len_kv;
        p.v_strideH = p.v_strideM * seq_len_kv;
        p.o_strideH = head_size_v;
        p.q_strideB = p.q_strideH * num_heads;
        p.k_strideB = p.k_strideH * num_heads;
        p.v_strideB = p.v_strideH * num_heads;
        p.o_strideB = head_size_v * seq_len * num_heads;
    }

    // launch kernel
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      std::cerr << "Kernel does not support these inputs" << std::endl;
      return;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    // cudaError_t err = cudaDeviceSynchronize();

    // if (err != cudaSuccess)  {
    //   std::cerr << "Kernel execution error: " << cudaGetErrorString(err);
    //   return;
    // }
}

)");
