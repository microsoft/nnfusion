from memopt.reference import get_ref_tensor
from memopt.schedule_rewrite import CodeGenerator
from memopt.utils import CompileResult
import memopt
import tvm
from tvm import te
import torch
import numpy as np

code = """
__device__ void top(half* __restrict__ A, half* __restrict__ B, half* __restrict__ output0, char* shared) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> output0_wmma_accumulator[8];
  half* A_shared = (half*)(shared);
  half* B_shared = (half*)(shared + 9216 * 2);
//   __shared__ half A_shared[9216];
//   __shared__ half B_shared[4608];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_shared_wmma_matrix_b[2];
  for (int x_c_outer_init = 0; x_c_outer_init < 4; ++x_c_outer_init) {
    for (int y_c_outer_init = 0; y_c_outer_init < 2; ++y_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(output0_wmma_accumulator[((x_c_outer_init * 2) + y_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax0_ax1_fused_outer = 0; ax0_ax1_fused_outer < 64; ++ax0_ax1_fused_outer) {
    A_shared[(((((ax0_ax1_fused_outer * 144) + (((int)threadIdx.z) * 72)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = A[((((((((int)blockIdx.x) * 8192) + (ax0_ax1_fused_outer * 128)) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))];
  }
  for (int ax0_ax1_fused_outer1 = 0; ax0_ax1_fused_outer1 < 32; ++ax0_ax1_fused_outer1) {
    B_shared[(((((ax0_ax1_fused_outer1 * 144) + (((int)threadIdx.z) * 72)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = B[(((((ax0_ax1_fused_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
      (void)nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_outer], ((half *)A_shared + ((((((int)threadIdx.y) * 4608) + (ax0_outer * 1152)) + (k_outer_inner * 16)))), 72);
    }
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[ax1_outer], ((half *)B_shared + ((((k_outer_inner * 1152) + (((int)threadIdx.z) * 32)) + (ax1_outer * 16)))), 72);
    }
    for (int x_c_outer = 0; x_c_outer < 4; ++x_c_outer) {
      for (int y_c_outer = 0; y_c_outer < 2; ++y_c_outer) {
        (void)nvcuda::wmma::mma_sync(output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)], A_shared_wmma_matrix_a[x_c_outer], B_shared_wmma_matrix_b[y_c_outer], output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)]);
      }
    }
  }
  __syncthreads();
  for (int x_inner_inner_outer = 0; x_inner_inner_outer < 4; ++x_inner_inner_outer) {
    for (int y_inner_inner_outer = 0; y_inner_inner_outer < 2; ++y_inner_inner_outer) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)output0 + ((((((((int)threadIdx.y) * 4608)) + (x_inner_inner_outer * 1152)) + (((int)threadIdx.z) * 32)) + (y_inner_inner_outer * 16)))), output0_wmma_accumulator[((x_inner_inner_outer * 2) + y_inner_inner_outer)], 72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
}

__device__ void down(half* __restrict__ A, half* __restrict__ B, half* __restrict__ output0, char* shared) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> output0_wmma_accumulator[8];
  half* A_shared = A;
  half* B_shared = (half*)shared;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_shared_wmma_matrix_b[2];
  for (int x_c_outer_init = 0; x_c_outer_init < 4; ++x_c_outer_init) {
    for (int y_c_outer_init = 0; y_c_outer_init < 2; ++y_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(output0_wmma_accumulator[((x_c_outer_init * 2) + y_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax0_ax1_fused_outer1 = 0; ax0_ax1_fused_outer1 < 32; ++ax0_ax1_fused_outer1) {
    B_shared[(((((ax0_ax1_fused_outer1 * 144) + (((int)threadIdx.z) * 72)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = B[(((((ax0_ax1_fused_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
      (void)nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_outer], ((half *)A_shared + ((((((int)threadIdx.y) * 4608) + (ax0_outer * 1152)) + (k_outer_inner * 16)))), 72);
    }
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[ax1_outer], ((half *)B_shared + ((((k_outer_inner * 1152) + (((int)threadIdx.z) * 32)) + (ax1_outer * 16)))), 72);
    }
    for (int x_c_outer = 0; x_c_outer < 4; ++x_c_outer) {
      for (int y_c_outer = 0; y_c_outer < 2; ++y_c_outer) {
        (void)nvcuda::wmma::mma_sync(output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)], A_shared_wmma_matrix_a[x_c_outer], B_shared_wmma_matrix_b[y_c_outer], output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)]);
      }
    }
  }
  for (int x_inner_inner_outer = 0; x_inner_inner_outer < 4; ++x_inner_inner_outer) {
    for (int y_inner_inner_outer = 0; y_inner_inner_outer < 2; ++y_inner_inner_outer) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)output0 + ((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 4096)) + (x_inner_inner_outer * 1024)) + (((int)threadIdx.z) * 32)) + (y_inner_inner_outer * 16)))), output0_wmma_accumulator[((x_inner_inner_outer * 2) + y_inner_inner_outer)], 64, nvcuda::wmma::mem_row_major);
    }
  }
}

__global__ void __launch_bounds__(128) Fused(half* input0, half* input1, half* input2, half* output0) {
  __shared__ char shared[(9216+4608)*2];
  top(input0, input1, (half*)(shared+0), shared+0);
  down((half*)(shared+0), input2, output0, shared+9216*2);
}
"""

codefull = """
__device__ void top(half* __restrict__ A, half* __restrict__ B, half* __restrict__ output0, char* shared) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> output0_wmma_accumulator[8];
  half* A_shared = (half*)(shared);
  half* B_shared = (half*)(shared + 9216 * 2);
//   __shared__ half A_shared[9216];
//   __shared__ half B_shared[4608];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_shared_wmma_matrix_b[2];
  for (int x_c_outer_init = 0; x_c_outer_init < 4; ++x_c_outer_init) {
    for (int y_c_outer_init = 0; y_c_outer_init < 2; ++y_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(output0_wmma_accumulator[((x_c_outer_init * 2) + y_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax0_ax1_fused_outer = 0; ax0_ax1_fused_outer < 64; ++ax0_ax1_fused_outer) {
    A_shared[(((((ax0_ax1_fused_outer * 144) + (((int)threadIdx.z) * 72)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = A[((((((((int)blockIdx.x) * 8192) + (ax0_ax1_fused_outer * 128)) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))];
  }
  for (int ax0_ax1_fused_outer1 = 0; ax0_ax1_fused_outer1 < 32; ++ax0_ax1_fused_outer1) {
    B_shared[(((((ax0_ax1_fused_outer1 * 144) + (((int)threadIdx.z) * 72)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = B[(((((ax0_ax1_fused_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
      (void)nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_outer], ((half *)A_shared + ((((((int)threadIdx.y) * 4608) + (ax0_outer * 1152)) + (k_outer_inner * 16)))), 72);
    }
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[ax1_outer], ((half *)B_shared + ((((k_outer_inner * 1152) + (((int)threadIdx.z) * 32)) + (ax1_outer * 16)))), 72);
    }
    for (int x_c_outer = 0; x_c_outer < 4; ++x_c_outer) {
      for (int y_c_outer = 0; y_c_outer < 2; ++y_c_outer) {
        (void)nvcuda::wmma::mma_sync(output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)], A_shared_wmma_matrix_a[x_c_outer], B_shared_wmma_matrix_b[y_c_outer], output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)]);
      }
    }
  }
  __syncthreads();
  for (int x_inner_inner_outer = 0; x_inner_inner_outer < 4; ++x_inner_inner_outer) {
    for (int y_inner_inner_outer = 0; y_inner_inner_outer < 2; ++y_inner_inner_outer) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)output0 + ((((((((int)threadIdx.y) * 4608)) + (x_inner_inner_outer * 1152)) + (((int)threadIdx.z) * 32)) + (y_inner_inner_outer * 16)))), output0_wmma_accumulator[((x_inner_inner_outer * 2) + y_inner_inner_outer)], 72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
}

__device__ void mid(half* __restrict__ A, half* __restrict__ B, half* __restrict__ output0, char* shared) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> output0_wmma_accumulator[8];
  half* A_shared = A;
  half* B_shared = (half*)shared;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_shared_wmma_matrix_b[2];
  for (int x_c_outer_init = 0; x_c_outer_init < 4; ++x_c_outer_init) {
    for (int y_c_outer_init = 0; y_c_outer_init < 2; ++y_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(output0_wmma_accumulator[((x_c_outer_init * 2) + y_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax0_ax1_fused_outer1 = 0; ax0_ax1_fused_outer1 < 32; ++ax0_ax1_fused_outer1) {
    B_shared[(((((ax0_ax1_fused_outer1 * 144) + (((int)threadIdx.z) * 72)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = B[(((((ax0_ax1_fused_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
      (void)nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_outer], ((half *)A_shared + ((((((int)threadIdx.y) * 4608) + (ax0_outer * 1152)) + (k_outer_inner * 16)))), 72);
    }
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[ax1_outer], ((half *)B_shared + ((((k_outer_inner * 1152) + (((int)threadIdx.z) * 32)) + (ax1_outer * 16)))), 72);
    }
    for (int x_c_outer = 0; x_c_outer < 4; ++x_c_outer) {
      for (int y_c_outer = 0; y_c_outer < 2; ++y_c_outer) {
        (void)nvcuda::wmma::mma_sync(output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)], A_shared_wmma_matrix_a[x_c_outer], B_shared_wmma_matrix_b[y_c_outer], output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)]);
      }
    }
  }
  __syncthreads();
  for (int x_inner_inner_outer = 0; x_inner_inner_outer < 4; ++x_inner_inner_outer) {
    for (int y_inner_inner_outer = 0; y_inner_inner_outer < 2; ++y_inner_inner_outer) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)output0 + ((((((((int)threadIdx.y) * 4608)) + (x_inner_inner_outer * 1152)) + (((int)threadIdx.z) * 32)) + (y_inner_inner_outer * 16)))), output0_wmma_accumulator[((x_inner_inner_outer * 2) + y_inner_inner_outer)], 72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
}

__device__ void down(half* __restrict__ A, half* __restrict__ B, half* __restrict__ output0, char* shared) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> output0_wmma_accumulator[8];
  half* A_shared = A;
  half* B_shared = (half*)shared;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_shared_wmma_matrix_b[2];
  for (int x_c_outer_init = 0; x_c_outer_init < 4; ++x_c_outer_init) {
    for (int y_c_outer_init = 0; y_c_outer_init < 2; ++y_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(output0_wmma_accumulator[((x_c_outer_init * 2) + y_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax0_ax1_fused_outer1 = 0; ax0_ax1_fused_outer1 < 32; ++ax0_ax1_fused_outer1) {
    B_shared[(((((ax0_ax1_fused_outer1 * 144) + (((int)threadIdx.z) * 72)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = B[(((((ax0_ax1_fused_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
      (void)nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_outer], ((half *)A_shared + ((((((int)threadIdx.y) * 4608) + (ax0_outer * 1152)) + (k_outer_inner * 16)))), 72);
    }
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[ax1_outer], ((half *)B_shared + ((((k_outer_inner * 1152) + (((int)threadIdx.z) * 32)) + (ax1_outer * 16)))), 72);
    }
    for (int x_c_outer = 0; x_c_outer < 4; ++x_c_outer) {
      for (int y_c_outer = 0; y_c_outer < 2; ++y_c_outer) {
        (void)nvcuda::wmma::mma_sync(output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)], A_shared_wmma_matrix_a[x_c_outer], B_shared_wmma_matrix_b[y_c_outer], output0_wmma_accumulator[((x_c_outer * 2) + y_c_outer)]);
      }
    }
  }
  for (int x_inner_inner_outer = 0; x_inner_inner_outer < 4; ++x_inner_inner_outer) {
    for (int y_inner_inner_outer = 0; y_inner_inner_outer < 2; ++y_inner_inner_outer) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)output0 + ((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 4096)) + (x_inner_inner_outer * 1024)) + (((int)threadIdx.z) * 32)) + (y_inner_inner_outer * 16)))), output0_wmma_accumulator[((x_inner_inner_outer * 2) + y_inner_inner_outer)], 64, nvcuda::wmma::mem_row_major);
    }
  }
}

__global__ void __launch_bounds__(128) Fused(half* input0, half* input1, half* input2, half* output0) {
  __shared__ char shared[(9216+4608)*2];
  top(input0, input1, (half*)(shared+0), shared+0);
  down((half*)(shared+0), input2, output0, shared+9216*2);
}
"""

def refernce(M, K, N, device=0, seed=0):
    torch.cuda.set_device(device)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    A = get_ref_tensor([M, K], device, "half")
    B = get_ref_tensor([K, N], device, "half")
    C = get_ref_tensor([K, N], device, "half")
    D = torch.matmul(torch.matmul(A, B), C)
    return D.cpu().numpy()

if __name__ == "__main__":
    M, K, N = 1920 * 1080, 64, 64
    args = [te.placeholder((M, K), dtype='float16', name="input0"),
            te.placeholder((K, N), dtype='float16', name="input1"),
            te.placeholder((K, N), dtype='float16', name="input2"),
            te.placeholder((M, N), dtype='float16', name="output0")]
    cp = CompileResult(None, codefull, [32, 2, 2], [16200, 1, 1], "Fused", args)
    cp.append_host_call()
    cp.compile_and_load()
    print(cp.profile())
    out = cp.get_example_outputs()
    ref = refernce(M, K, N)
    print(np.max(np.abs(out-ref)))
    print(out, ref)
