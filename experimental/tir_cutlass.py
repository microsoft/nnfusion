import memopt
import numpy as np
import tvm
from cutlass_intrin import *
from layout import *
from memopt.utils import CompileResult
from tvm import te

tvm.register_func("tvm_callback_cuda_compile", override=True)(lambda x:"")

def gemm(n, m, k):
    """TVM expression for vector add"""
    A = te.placeholder((n, k), dtype="float16", name='a')
    B = te.placeholder((k, m), dtype="float16", name='b')
    K = te.reduce_axis((0, k))
    C = te.compute((n, m), lambda i, j: te.sum(A[i,K]*B[K,j], axis=[K]), name='output')
    return A, B, C

def sche_gemm(sch: tvm.tir.Schedule):
    C = sch.get_block("output")

    block_size_M, block_size_N = 256, 128
    warp_size_M, warp_size_N = 128, 64
    chunk_size = 32
    warp_size = 32
    num_warp = (block_size_M * block_size_N) // (warp_size_M * warp_size_N)

    ax_M, ax_N, ax_K = sch.get_loops(C)
    grid_M, block_M = sch.split(ax_M, factors=[None, block_size_M])
    grid_N, block_N = sch.split(ax_N, factors=[None, block_size_N])
    sch.reorder(grid_M, grid_N, block_M, block_N)
    grid = sch.fuse(grid_M, grid_N)
    sch.bind(grid, "blockIdx.x")

    grid, ax_M, ax_N, ax_K = sch.get_loops(C)
    K_outer, K_inner = sch.split(ax_K, factors=[None, chunk_size])
    warp_M, inner_M = sch.split(ax_M, factors=[None, warp_size_M])
    warp_N, inner_N = sch.split(ax_N, factors=[None, warp_size_N])
    sch.reorder(warp_M, warp_N, K_outer, inner_M, inner_N, K_inner)
    warp = sch.fuse(warp_M, warp_N)
    sch.bind(warp, "threadIdx.y")

    layoutA = RowMajorVoltaTensorOpMultiplicandCrosswise(block_size_M, chunk_size)
    layoutB = RowMajorVoltaTensorOpMultiplicandBCongruous(chunk_size, block_size_N)
    # layoutA = RowMajorLayout(block_size_M, chunk_size)
    # layoutB = RowMajorLayout(chunk_size, block_size_N)

    for idx in [0, 1]:
        layout = layoutA if idx==0 else layoutB
        SS = sch.cache_read(C, idx, "shared")
        sch.compute_at(SS, K_outer)
        if layout.requires_padding():
            pad_size = 4 if idx == 0 else 8 # m8n8k4
            layout.set_pad(pad_size)
            sch.storage_align(SS, 0, axis=-2, factor=2*pad_size, offset=pad_size)
        fused = sch.fuse(*sch.get_loops(SS)[-2:])
        vectorize_size = layout.get_vectorize()
        oo, idx_y, idx_x, vec = sch.split(fused, [None, num_warp, warp_size, vectorize_size])
        sch.bind(idx_x, "threadIdx.x")
        sch.bind(idx_y, "threadIdx.y")
        sch.vectorize(vec)
        sch.annotate(SS, "tir.manifest_shared_memory_local_stage", 1)
        # sch.unroll(oo)

    cls_code = register_volta_cutlass_warp_mma(warp_size_M, warp_size_N, chunk_size, layoutA, layoutB)
    C_warp = sch.cache_write(C, 0, "cutlass.warp.mma")
    sch.reverse_compute_at(C_warp, warp)

    sch.decompose_reduction(C, sch.get_loops(C)[2])
    block_init_c = sch.get_block("output_init")
    layoutC = voltaFragmentCLayout32x32(warp_size_M, warp_size_N)

    sch.transform_loop(C_warp, 2, layoutC)
    sch.bind(sch.get_loops(C_warp)[-2], "threadIdx.x")
    oo, vec = sch.split(sch.get_loops(C_warp)[-1], factors=[None, layoutC.get_vectorize()])
    sch.vectorize(vec)
    sch.unroll(oo)
    sch.annotate(sch.get_loops(C)[2], "software_pipeline_stage", [0, 0, 0, 0, 1])
    sch.annotate(sch.get_loops(C)[2], "software_pipeline_order", [0, 3, 1, 4, 2])
    sch.tensorize(sch.get_loops(block_init_c)[-2],
        register_cutlass_warp_init_intrin(warp_size_M, warp_size_N, "float16",
        cls_code, block_size_M // warp_size_M, block_size_N // warp_size_N)
    )
    sch.tensorize(sch.get_loops(C)[-3],
        register_gemm_intrin(
            warp_size_M, warp_size_N, chunk_size, "float16", "float16", False, False,
            layoutA, layoutB)
    )
    layout_pass = ApplyLayoutPass({"a_shared": layoutA, "b_shared": layoutB, "output_cutlass.warp.mma": layoutC.fragment_offset})
    passes = [
        layout_pass.get_pass(),
    ]
    grid = [np.prod(args[-1].shape) // block_size_M // block_size_N, 1, 1]
    block = [warp_size, num_warp, 1]
    return grid, block, passes


args = gemm(8192, 8192, 8192)
workload = te.create_prim_func(args)
ir_module = tvm.IRModule({"main": workload})
sch = tvm.tir.Schedule(ir_module)
from memopt.IRpass import *

grid, block, passes = sche_gemm(sch)
with tvm.transform.PassContext(config={"tir.add_lower_pass": passes}, disabled_pass=["tir.UnrollLoop"]):
    mod = tvm.build(sch.mod["main"], target="cuda")
kernel_code = mod.imported_modules[0].get_source()
kernel_code = kernel_code[kernel_code.index('extern "C" __global__ void'):]

print(kernel_code)
cp = CompileResult(None, kernel_code, block, grid, "default_function_kernel0", args)
cp.compile_and_load(memopt.arch.V100())
a = cp.get_example_outputs()[0]
print(a)
print(cp.profile())

# from memopt.reference import get_reference_output

# oo = get_reference_output(args)[-1].numpy()
# print(oo)
# print(abs(oo - a).max())
