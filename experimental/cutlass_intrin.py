from typing import Callable, List

import tvm
from tvm import tir
from tvm.runtime import convert
from tvm.script import tir as T
from tvm.tir import TensorIntrin


def register_cutlass_warp_mma(warp_M, warp_N, warp_K, SMemLayoutA, layoutA, SMemLayoutB, layoutB):
    cls_code = f"""cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<{warp_M}, {warp_N}, {warp_K}>,
    {SMemLayoutA}, {layoutA},
    {SMemLayoutB}, {layoutB},
    cutlass::layout::RowMajor
>"""
    tvm.register_func("get_cutlass_warp_mma", lambda : cls_code, override=True)
    tvm.register_func("get_cutlass_warp_mma_size", lambda : warp_M*warp_N, override=True)

def get_fragment_index(buffer, m_dim, n_dim):
    """Compute wmma fragment index using elem_offset of the buffer"""
    frag_size = convert(m_dim * n_dim)
    return buffer.elem_offset // frag_size + (buffer.elem_offset % frag_size) // n_dim

def register_mma_fill_intrin(m_dim: int, n_dim: int, dtype: str, LayoutC) -> str:
    """Generator of mma intrins"""
    zero = tir.IntImm("int32", 0).astype(dtype)
    warp_size = 32
    C_fraglen = m_dim * n_dim // 32
    @T.prim_func
    def desc(c: T.handle) -> None:
        C = T.match_buffer(
            c, (warp_size, C_fraglen), dtype, scope="cutlass.warp.mma"
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:warp_size, 0:C_fraglen])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("init"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    thread_id_C, local_id_C = LayoutC(vii, vjj)
                    C[thread_id_C, local_id_C] = zero

    @T.prim_func
    def impl(c: T.handle) -> None:
        C = T.match_buffer(
            c, (warp_size, C_fraglen), dtype, scope="cutlass.warp.mma"
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:warp_size, 0:C_fraglen])
            T.evaluate(
                T.cutlass_init_fragment(
                    C.data,
                    get_fragment_index(C, m_dim, n_dim),
                    dtype="handle",
                )
            )
    TensorIntrin.register("mma_fill", desc, impl)
    return "mma_fill"

def register_gemm_intrin(m_dim: int, n_dim: int, k_dim: int, in_dtype: str, out_dtype: str,
                         transpose_A: bool, transpose_B: bool,
                         LayoutA, LayoutB, LayoutC, num_warp_m, num_warp_n) -> str:
    """Generator of cutlass gemm intrins"""
    warp_size = 32
    C_fraglen = m_dim * n_dim // 32
    A_shape_0, A_shape_1 = m_dim, k_dim
    B_shape_0, B_shape_1 = k_dim, n_dim
    if transpose_A:
        A_shape_0, A_shape_1 = A_shape_1, A_shape_0
    if transpose_B:
        B_shape_0, B_shape_1 = B_shape_1, B_shape_0
    C_layout_func = LayoutC.get()
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (A_shape_0, A_shape_1), in_dtype, offset_factor=16, scope="shared"
        )
        B = T.match_buffer(
            b, (B_shape_0, B_shape_1), in_dtype, offset_factor=16, scope="shared",
        )
        C = T.match_buffer(
            c, (warp_size, C_fraglen), out_dtype, scope="cutlass.warp.mma"
        )

        with T.block("root"):
            T.reads(C[0:warp_size, 0:C_fraglen], A[0:A_shape_0, 0:A_shape_1], B[0:B_shape_0, 0:B_shape_1])
            T.writes(C[0:warp_size, 0:C_fraglen])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    thread_id_C, local_id_C = C_layout_func(vii, vjj)
                    C[thread_id_C, local_id_C] = C[thread_id_C, local_id_C] + A[vii, vkk] * B[vkk, vjj]


    A_access_ptr = lambda buffer, warp_idx: LayoutA.get_access_ptr(buffer, warp_idx // num_warp_n)
    B_access_ptr = lambda buffer, warp_idx: LayoutB.get_access_ptr(buffer, warp_idx % num_warp_n)
    A_param = lambda s: LayoutA.get_param(s)
    B_param = lambda s: LayoutB.get_param(s)
    num_warp = num_warp_m * num_warp_n
    @T.prim_func
    def impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        As = T.var("int32")
        Bs = T.var("int32")
        _1 = T.var("int32")
        _2 = T.var("int32")
        A = T.match_buffer(
            a, (A_shape_0, A_shape_1), in_dtype, offset_factor=16, scope="shared", strides=[As, _1]
        )
        B = T.match_buffer(
            b, (B_shape_0, B_shape_1), in_dtype, offset_factor=16, scope="shared", strides=[Bs, _2]
        )
        C = T.match_buffer(
            c, (warp_size, C_fraglen), out_dtype, scope="cutlass.warp.mma"
        )

        with T.block("root"):
            T.reads(C[0:warp_size, 0:C_fraglen], A[0:A_shape_0, 0:A_shape_1], B[0:B_shape_0, 0:B_shape_1])
            T.writes(C[0:warp_size, 0:C_fraglen])
            warp_idx = T.env_thread("threadIdx.y")
            T.launch_thread(warp_idx, num_warp)
            T.evaluate(
                T.cutlass_warp_mma(
                    C.data,
                    get_fragment_index(C, m_dim, n_dim),
                    A_access_ptr(A, warp_idx), A_param(As),
                    B_access_ptr(B, warp_idx), B_param(Bs),
                    dtype="handle",
                )
            )
    TensorIntrin.register("mma_sync", desc, impl)
    return "mma_sync"
