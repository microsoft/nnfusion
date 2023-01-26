cuda_default_header = """
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>
"""

cuda_fp16_header = """
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}

typedef long long _ll;
#define int64_t _ll
#define __int8_t_defined

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x) {          \
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

// Pack two half values.
inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// There is no make_int8 in cuda, but TVM codegen seem to use it
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

"""

rocm_default_header = """
#include <hip/hip_runtime.h>
#include <math.h>
"""

rocm_fp16_header = """
#define half _Float16
#define __float2half_rn(x) half(x)

#define htanh tanhf
#define htan tanf
#define hatan atanf
#define herf erff
#include <hip/hcc_detail/hip_fp16_math_fwd.h>
#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16
#define hexp __ocml_exp_f16

// Pack two half values.
inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// There is no make_int8 in cuda, but TVM codegen seem to use it
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
"""

cutlass_header = """
#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"

namespace cutlass {
namespace gemm {
namespace warp {

template <
  typename Shape,
  typename SMemLayoutA,
  typename LayoutA,
  typename SmemLayoutB,
  typename LayoutB,
  typename LayoutC
>
class GemmTensorOp {
public:

  using WarpShape = GemmShape<Shape::kM, Shape::kN, 4>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      cutlass::half_t,
      LayoutA,
      cutlass::half_t,
      LayoutB,
      cutlass::half_t,
      LayoutC,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    cutlass::half_t,
    SMemLayoutA,
    cutlass::half_t,
    SmemLayoutB,
    cutlass::half_t,
    LayoutC,
    Policy
  >;
  int const kKgroups = (Shape::kK + 3) / 4;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
  MmaWarp mma_op;
  typename MmaWarp::FragmentA frag_A;
  typename MmaWarp::FragmentB frag_B;
public:
  CUTLASS_HOST_DEVICE
  GemmTensorOp() {
    accum.clear();
  }
  typename MmaWarp::FragmentC accum;
  CUTLASS_DEVICE
  void operator()(TensorRefA ref_A, TensorRefB ref_B, int lane_id) {
    typename MmaWarp::IteratorA iter_A(ref_A, lane_id);
    typename MmaWarp::IteratorB iter_B(ref_B, lane_id);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups; ++k) {
      iter_A.load(frag_A);
      iter_B.load(frag_B);
      ++iter_A;
      ++iter_B;
      mma_op(accum, frag_A, frag_B, accum);
    }
  }
  CUTLASS_DEVICE
  half& operator[](size_t i) {
    return ((half*)accum.data())[i];
  }
  CUTLASS_DEVICE
  half* operator+(size_t i) {
    return (half*)accum.data() + i;
  }
};

}}}
"""
