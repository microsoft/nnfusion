import memopt
import numpy as np
import torch
import tvm
from memopt.reference import get_ref_tensor
from memopt.utils import CompileResult
from tvm import te
from tvm.topi.cuda.tensor_intrin import (intrin_wmma_gemm,
                                         intrin_wmma_load_matrix_A,
                                         intrin_wmma_load_matrix_W,
                                         intrin_wmma_store_matrix)


def matmul(M, K, N):
    A = te.placeholder((M, K), dtype='float16', name="A")
    B = te.placeholder((K, N), dtype='float16', name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    C = te.compute((M, N), lambda *i: te.if_then_else(C[i]>=0, C[i], tvm.tir.const(0, "float16")), name="output0")
    return [A, B, C]

def cooperative_fetch_2d(shared, sch, block_size, strides):
    assert len(block_size) == 3
    axes = sch[shared].op.axis
    sch[shared].storage_align(axes[0], strides - 1, strides)
    fused = sch[shared].fuse(*axes)
    oo, temp_0 = sch[shared].split(fused, factor=block_size[0] * block_size[1] * block_size[2])
    inner_z, temp_1 = sch[shared].split(temp_0, factor=block_size[0] * block_size[1])
    inner_y, inner_x = sch[shared].split(temp_1, factor=block_size[0])
    sch[shared].reorder(oo, inner_z, inner_y, inner_x)
    sch[shared].bind(inner_x, te.thread_axis("threadIdx.x"))
    sch[shared].bind(inner_y, te.thread_axis("threadIdx.y"))
    sch[shared].bind(inner_z, te.thread_axis("threadIdx.z"))

def get_schedule(A, B, R):

    s = te.create_schedule(R.op)
    C = s[R].op.input_tensors[0]
    s[C].compute_inline()
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, "shared", [C])
    wmma_m, wmma_n, wmma_k = 16, 16, 16
    rstep_size = 4

    m_tile = [128, 64, wmma_m]
    n_tile = [64, 32, wmma_n]
    block_size = [32, int(m_tile[0] / m_tile[1]), int(n_tile[0] / n_tile[1])]

    offset = 8
    AF_stride = [wmma_k, 1]
    AS_stride = [rstep_size * wmma_k + offset, 1]
    BF_stride = [n_tile[1], 1]
    BS_stride = [n_tile[0] + offset, 1]
    CF_stride = [n_tile[1], 1]
    CS_stride = [n_tile[0] + offset, 1]

    # Schedule for output
    block_i, mc = s[R].split(R.op.axis[0], factor=m_tile[0])
    block_j, nc = s[R].split(R.op.axis[1], factor=n_tile[0])
    s[R].reorder(block_i, block_j, mc, nc)
    block_fused = s[R].fuse(block_i, block_j)
    t = s[R].fuse(mc, nc)
    t, tx = s[R].split(t, factor=block_size[0])
    t, ty = s[R].split(t, factor=block_size[1])
    t, tz = s[R].split(t, factor=block_size[2])
    s[R].bind(tz, te.thread_axis("threadIdx.z"))
    s[R].bind(ty, te.thread_axis("threadIdx.y"))
    s[R].bind(tx, te.thread_axis("threadIdx.x"))
    s[R].bind(block_fused, te.thread_axis("blockIdx.x"))

    # schedule for block
    s[CS].compute_at(s[R], block_fused)
    mc, nc = CS.op.axis
    s[CS].storage_align(mc, CS_stride[0] - 1, CS_stride[0])
    mm, mmii = s[CS].split(mc, factor=m_tile[1])
    nn, nnii = s[CS].split(nc, factor=n_tile[1])
    mmii, mmi = s[CS].split(mmii, factor=m_tile[2])
    nnii, nni = s[CS].split(nnii, factor=n_tile[2])
    s[CS].reorder(mm, nn, mmii, nnii, mmi, nni)
    s[CS].bind(mm, te.thread_axis("threadIdx.y"))
    s[CS].bind(nn, te.thread_axis("threadIdx.z"))

    # Schedule for wmma computation
    s[CF].compute_at(s[CS], nn)
    warp_i, _ii = s[CF].split(CF.op.axis[0], factor=m_tile[2])
    warp_j, _jj = s[CF].split(CF.op.axis[1], factor=n_tile[2])
    k = CF.op.reduce_axis[0]
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=rstep_size)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    m, m_ii = s[AF].split(AF.op.axis[0], factor=wmma_m)
    i, i_jj = s[AF].split(AF.op.axis[1], factor=wmma_k)
    s[AF].reorder(m, i, m_ii, i_jj)

    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    i, i_ii = s[BF].split(BF.op.axis[0], factor=wmma_k)
    n, n_ii = s[BF].split(BF.op.axis[1], factor=wmma_n)
    s[BF].reorder(i, n, i_ii, n_ii)

    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=A.dtype)
    BL_gemm = te.placeholder((wmma_k, wmma_n), name="BL_gemm", dtype=B.dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(C.dtype) * BL_gemm[k_gemm, jj].astype(C.dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    # schedule shared
    s[AS].compute_at(s[CF], ko)
    s[BS].compute_at(s[CF], ko)
    cooperative_fetch_2d(AS, s, block_size, AS_stride[0])
    cooperative_fetch_2d(BS, s, block_size, BS_stride[0])

    s[AF].tensorize(
        m_ii,
        intrin_wmma_load_matrix_A(
            AF_stride, AS_stride, shape, "row_major", (wmma_m, wmma_k), (wmma_m, wmma_k), A.dtype
        ),
    )
    s[BF].tensorize(
        i_ii,
        intrin_wmma_load_matrix_W(
            BF_stride, BS_stride, shape, "row_major", (wmma_k, wmma_n), (wmma_k, wmma_n), B.dtype
        ),
    )
    s[CF].tensorize(
        _ii, intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape)
    )
    s[CS].tensorize(
        mmi,
        intrin_wmma_store_matrix(
            CS_stride, CF_stride, shape, C.dtype, (wmma_m, wmma_n), (wmma_m, wmma_n)
        ),
    )

    return s

def refernce(M, K, N, device=0, seed=0):
    torch.cuda.set_device(device)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    A = get_ref_tensor([M, K], device, "half")
    B = get_ref_tensor([K, N], device, "half")
    C = torch.relu(torch.matmul(A, B))
    return C.cpu().numpy()

if __name__ == "__main__":
    M, K, N = 1920 * 1080, 64, 64
    args = matmul(M, K, N)
    s = get_schedule(*args)
    with memopt.Scope(s):
        func = tvm.build(s, args, "cuda")
        source = func.imported_modules[0].get_source()
        source = source[source.index('extern "C" __global__'):]
        print(memopt.get_scope().block_size, memopt.get_scope().grid_size)
        cp = CompileResult(None, source, memopt.get_scope().block_size, memopt.get_scope().grid_size, "default_function_kernel0", args)
        print(cp.code)
    cp.compile_and_load()
    ref = refernce(M, K, N)
    out = cp.get_example_outputs()[0]
    print(np.max(np.abs(out-ref)))
    print(cp.profile())

    print(out, ref)

