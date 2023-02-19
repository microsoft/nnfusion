from typing import Callable, List

import tvm
from tvm import te, tir
from tvm.script import tir as T
from tvm.tir import TensorIntrin


def register_cutlass_warp_mma(warp_M, warp_N, warp_K, layoutA, layoutB):
    cls_code = f"""cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<{warp_M}, {warp_N}, {warp_K}>,
    {layoutA.smem_layout_name()},
    {layoutB.smem_layout_name()}
>"""
    return cls_code

def register_volta_cutlass_warp_mma(warp_M, warp_N, warp_K, layoutA, layoutB):
    cls_code = f"""cutlass::gemm::warp::VoltaGemmTensorOp<
    cutlass::gemm::GemmShape<{warp_M}, {warp_N}, {warp_K}>,
    {layoutA.smem_layout_name()}, {layoutA.local_layout_name()},
    {layoutB.smem_layout_name()}, {layoutB.local_layout_name()},
    cutlass::layout::RowMajor
>"""
    return cls_code

def register_cutlass_warp_init_intrin(m_dim: int, n_dim: int, dtype: str, cls_name: str,
    num_warp_m: int, num_warp_n: int) -> str:
    """Generator of mma intrins"""
    zero = tir.IntImm("int32", 0).astype(dtype)
    @T.prim_func
    def desc(c: T.handle) -> None:
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, scope="cutlass.warp.mma"
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("init"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = zero
    num_warp = num_warp_m * num_warp_n
    get_warp_idx_m = lambda warp_idx: warp_idx // num_warp_n
    get_warp_idx_n = lambda warp_idx: warp_idx % num_warp_n
    @T.prim_func
    def impl(c: T.handle) -> None:
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, scope="cutlass.warp.mma"
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            warp_idx = T.env_thread("threadIdx.y")
            T.launch_thread(warp_idx, num_warp)
            T.evaluate(
                T.cutlass_init_fragment(
                    C.data,
                    cls_name,
                    get_warp_idx_m(warp_idx),
                    get_warp_idx_n(warp_idx),
                    dtype="handle",
                )
            )
    TensorIntrin.register("mma_fill", desc, impl, override=True)
    return "mma_fill"

def register_gemm_intrin(m_dim: int, n_dim: int, k_dim: int, in_dtype: str, out_dtype: str,
                         transpose_A: bool, transpose_B: bool, layoutA, layoutB) -> str:
    """Generator of cutlass gemm intrins"""
    A_shape_0, A_shape_1 = m_dim, k_dim
    B_shape_0, B_shape_1 = k_dim, n_dim
    if transpose_A:
        A_shape_0, A_shape_1 = A_shape_1, A_shape_0
    if transpose_B:
        B_shape_0, B_shape_1 = B_shape_1, B_shape_0
    def maybe_swap_A(i, j):
        if transpose_A:
            return j, i
        return i, j
    def maybe_swap_B(i, j):
        if transpose_B:
            return j, i
        return i, j
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (A_shape_0, A_shape_1), in_dtype, elem_offset=te.var(), scope="shared"
        )
        B = T.match_buffer(
            b, (B_shape_0, B_shape_1), in_dtype, elem_offset=te.var(), scope="shared",
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), out_dtype, scope="cutlass.warp.mma"
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:A_shape_0, 0:A_shape_1], B[0:B_shape_0, 0:B_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    ai, ak = maybe_swap_A(vii, vkk)
                    bk, bj = maybe_swap_B(vkk, vjj)
                    C[vii, vjj] = C[vii, vjj] + A[ai, ak] * B[bk, bj]
    read_ptr = lambda buffer: buffer.access_ptr("r", offset=-buffer.elem_offset)
    stride_A = layoutA.get_stride()
    stride_B = layoutB.get_stride()
    @T.prim_func
    def impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (A_shape_0, A_shape_1), in_dtype, elem_offset=te.var(), scope="shared"
        )
        B = T.match_buffer(
            b, (B_shape_0, B_shape_1), in_dtype, elem_offset=te.var(), scope="shared"
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), out_dtype, scope="cutlass.warp.mma"
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:A_shape_0, 0:A_shape_1], B[0:B_shape_0, 0:B_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.cutlass_warp_mma(
                    C.data,
                    "prologue",
                    read_ptr(A),
                    stride_A,
                    read_ptr(B),
                    stride_B,
                    dtype="handle",
                )
            )
            T.evaluate(
                T.cutlass_warp_mma(C.data, "body", read_ptr(A), read_ptr(B), dtype="handle")
            )
            T.evaluate(
                T.cutlass_warp_mma(C.data, "epilogue", dtype="handle")
            )
    TensorIntrin.register("mma_sync", desc, impl, override=True)
    return "mma_sync"
