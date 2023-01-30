old = """
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> output_wmma_accumulator[4];
  __shared__ half a_shared[2560];
  __shared__ half b_shared[2304];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> a_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_shared_wmma_matrix_b[2];
  for (int i0_1_1_init = 0; i0_1_1_init < 2; ++i0_1_1_init) {
    for (int i1_1_1_init = 0; i1_1_1_init < 2; ++i1_1_1_init) {
      nvcuda::wmma::fill_fragment(output_wmma_accumulator[((i0_1_1_init * 2) + i1_1_1_init)], 0.000000e+00f);
    }
  }
  for (int i2_0 = 0; i2_0 < 16; ++i2_0) {
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
      *(uint4*)(a_shared + ((((ax0_ax1_fused_0 * 1280) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(a + (((((((((int)blockIdx.x) >> 3) * 32768) + (ax0_ax1_fused_0 * 16384)) + (((int)threadIdx.y) * 4096)) + ((((int)threadIdx.x) >> 2) * 512)) + (i2_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    for (int ax0_ax1_fused_01 = 0; ax0_ax1_fused_01 < 2; ++ax0_ax1_fused_01) {
      *(uint4*)(b_shared + ((((ax0_ax1_fused_01 * 1152) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(b + ((((((i2_0 * 16384) + (ax0_ax1_fused_01 * 8192)) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    __syncthreads();
    for (int i2_1 = 0; i2_1 < 2; ++i2_1) {
      for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
        nvcuda::wmma::load_matrix_sync(a_shared_wmma_matrix_a[ax0_0], (&(a_shared[((((((int)threadIdx.y) >> 1) * 1280) + (ax0_0 * 640)) + (i2_1 * 16))])), 40);
      }
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
        nvcuda::wmma::load_matrix_sync(b_shared_wmma_matrix_b[ax1_0], (&(b_shared[(((i2_1 * 1152) + ((((int)threadIdx.y) & 1) * 32)) + (ax1_0 * 16))])), 72);
      }
      for (int i0_1_1 = 0; i0_1_1 < 2; ++i0_1_1) {
        for (int i1_1_1 = 0; i1_1_1 < 2; ++i1_1_1) {
          nvcuda::wmma::mma_sync(output_wmma_accumulator[((i0_1_1 * 2) + i1_1_1)], a_shared_wmma_matrix_a[i0_1_1], b_shared_wmma_matrix_b[i1_1_1], output_wmma_accumulator[((i0_1_1 * 2) + i1_1_1)]);
        }
      }
    }
  }
  for (int ax0_01 = 0; ax0_01 < 2; ++ax0_01) {
    for (int ax1_01 = 0; ax1_01 < 2; ++ax1_01) {
      nvcuda::wmma::store_matrix_sync((&(output[(((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.y) >> 1) * 16384)) + (ax0_01 * 8192)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.y) & 1) * 32)) + (ax1_01 * 16))])), output_wmma_accumulator[((ax0_01 * 2) + ax1_01)], 512, nvcuda::wmma::mem_row_major);
    }
  }
}
"""

kernel_code = """
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(half* __restrict__ a, half* __restrict__ b, half* __restrict__ output) {
  cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 16, 4>,
    cutlass::half_t,cutlass::layout::RowMajor,
    cutlass::half_t,cutlass::layout::RowMajor,
    cutlass::half_t,cutlass::layout::RowMajor
> output_cutlass_warp_mma[1];
  __shared__ half a_shared[9216];
  __shared__ half b_shared[8704];
  for (int i2_0 = 0; i2_0 < 8; ++i2_0) {
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0) {
      *(uint4*)(a_shared + ((((ax0_ax1_fused_0 * 1152) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(a + (((((((((int)blockIdx.x) >> 2) * 65536) + (ax0_ax1_fused_0 * 8192)) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (i2_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    for (int ax0_ax1_fused_01 = 0; ax0_ax1_fused_01 < 8; ++ax0_ax1_fused_01) {
      *(uint4*)(b_shared + ((((ax0_ax1_fused_01 * 1088) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(b + ((((((i2_0 * 32768) + (ax0_ax1_fused_01 * 4096)) + (((int)threadIdx.y) * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)blockIdx.x) & 3) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    __syncthreads();
    output_cutlass_warp_mma[0]({(cutlass::half_t*)(&(a_shared[((((int)threadIdx.y) >> 1) * 4608)])), 72},{(cutlass::half_t*)(&(b_shared[((((int)threadIdx.y) & 1) * 64)])), 136}, threadIdx.x);
  }
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 32; ++ax1_0) {
    *(uint2*)(output + ((((((((((((((((int)blockIdx.x) >> 2) * 65536) + ((((int)threadIdx.y) >> 1) * 32768)) + ((ax1_0 >> 4) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (((((int)threadIdx.x) & 7) >> 2) * 4096)) + (((ax1_0 & 3) >> 1) * 2048)) + ((((int)threadIdx.x) & 3) * 512)) + ((((int)blockIdx.x) & 3) * 128)) + ((((int)threadIdx.y) & 1) * 64)) + (((ax1_0 & 15) >> 3) * 32)) + ((ax1_0 & 1) * 16)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + (((ax1_0 & 7) >> 2) * 4))) = *(uint2*)(output_cutlass_warp_mma[0] + (ax1_0 * 4));
  }
}
"""

import memopt
from memopt.utils import CompileResult
from tvm import te


def gemm(n, m, k):
    """TVM expression for vector add"""
    A = te.placeholder((n, k), dtype="float16", name='a')
    B = te.placeholder((k, m), dtype="float16", name='b')
    K = te.reduce_axis((0, k))
    C = te.compute((n, m), lambda i, j: te.sum(A[i,K]*B[K,j], axis=[K]), name='output')
    return A, B, C

args = gemm(512, 512, 512)
cp = CompileResult(None, kernel_code, [32, 4, 1], [64, 1, 1], "default_function_kernel0", args)
cp.compile_and_load(memopt.arch.V100())
a = cp.get_example_outputs()[0]
print(cp.profile())
print(a)
from memopt.reference import get_reference_output

oo = get_reference_output(args)[-1].numpy()
print(abs(oo - a).max())
