import memopt
import tvm
from memopt.utils import CompileResult
from tvm import te
from tvm.tir.tensor_intrin.cuda import (WMMA_FILL_16x16x16_F16_INTRIN,
                                        WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
                                        WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
                                        WMMA_SYNC_16x16x16_f16f16f16_INTRIN)

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

    ax_N, ax_M, ax_K = sch.get_loops(C)
    grid_N, block_N = sch.split(ax_N, factors=[None, 64])
    grid_M, block_M = sch.split(ax_M, factors=[None, 64])
    sch.reorder(grid_N, grid_M, block_N, block_M)
    grid = sch.fuse(grid_N, grid_M)
    sch.bind(grid, "blockIdx.x")

    grid, ax_N, ax_M, ax_K = sch.get_loops(C)
    K_outer, K_inner, wmma_K = sch.split(ax_K, factors=[None, 4, 16])
    warp_N, va_N, wmma_N = sch.split(ax_N, factors=[None, 2, 16])
    warp_M, va_M, wmma_M = sch.split(ax_M, factors=[None, 2, 16])
    sch.reorder(warp_N, warp_M, K_outer, K_inner, va_N, va_M, wmma_N, wmma_M, wmma_K)
    warp = sch.fuse(warp_N, warp_M)
    sch.bind(warp, "threadIdx.y")

    for idx in [0, 1]:
        SS = sch.cache_read(C, idx, "shared")
        sch.compute_at(SS, K_outer)
        ax_M, ax_N = sch.get_loops(SS)[-2:]
        fused = sch.fuse(ax_M, ax_N)
        oo, idx_y, idx_x, vec = sch.split(fused, [None, 4, 32, 8])
        sch.bind(idx_x, "threadIdx.x")
        sch.bind(idx_y, "threadIdx.y")
        sch.vectorize(vec)
        sch.storage_align(SS, 0, axis=-2, factor=32, offset=8)

    A_warp = sch.cache_read(C, 0, "wmma.matrix_a")
    B_warp = sch.cache_read(C, 1, "wmma.matrix_b")
    sch.compute_at(A_warp, K_inner)
    sch.compute_at(B_warp, K_inner)

    C_warp = sch.cache_write(C, 0, "wmma.accumulator")
    sch.reverse_compute_at(C_warp, warp)

    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)

    sch.decompose_reduction(C, sch.get_loops(C)[2])
    block_init_c = sch.get_block("output_init")
    def tile_wmma_fragment(block_read, height, width):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, width])
        sch.reorder(i0, j0, i1, j1)
        return i1
    tile_wmma_fragment(A_warp, 16, 16)
    tile_wmma_fragment(B_warp, 16, 16)
    sch.tensorize(sch.get_loops(A_warp)[-2], "wmma_load_16x16x16_f16_a")
    sch.tensorize(sch.get_loops(B_warp)[-2], "wmma_load_16x16x16_f16_b")
    sch.tensorize(sch.get_loops(block_init_c)[-2], WMMA_FILL_16x16x16_F16_INTRIN)
    sch.tensorize(sch.get_loops(C_warp)[-2], WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN)
    sch.tensorize(sch.get_loops(C)[-3], WMMA_SYNC_16x16x16_f16f16f16_INTRIN)
    # print(sch.mod["main"].script())
    # exit(0)

def sche_gemm_shared(sch: tvm.tir.Schedule):
    C = sch.get_block("output")

    ax_N, ax_M, ax_K = sch.get_loops(C)
    grid_N, block_N = sch.split(ax_N, factors=[None, 64])
    grid_M, block_M = sch.split(ax_M, factors=[None, 64])
    sch.reorder(grid_N, grid_M, block_N, block_M)
    grid = sch.fuse(grid_N, grid_M)
    sch.bind(grid, "blockIdx.x")
    C_shared = sch.cache_write(C, 0, "shared")
    sch.reverse_compute_at(C_shared, grid)
    ax_M, ax_N = sch.get_loops(C_shared)[-2:]
    fused = sch.fuse(ax_M, ax_N)
    oo, idx_y, idx_x, vec = sch.split(fused, [None, 4, 32, 8])
    sch.bind(idx_x, "threadIdx.x")
    sch.bind(idx_y, "threadIdx.y")
    sch.vectorize(vec)
    sch.storage_align(C, 0, axis=-2, factor=32, offset=8)

    grid, ax_N, ax_M, ax_K = sch.get_loops(C)
    K_outer, K_inner, wmma_K = sch.split(ax_K, factors=[None, 2, 16])
    warp_N, va_N, wmma_N = sch.split(ax_N, factors=[None, 2, 16])
    warp_M, va_M, wmma_M = sch.split(ax_M, factors=[None, 2, 16])
    sch.reorder(warp_N, warp_M, K_outer, K_inner, va_N, va_M, wmma_N, wmma_M, wmma_K)
    warp = sch.fuse(warp_N, warp_M)
    sch.bind(warp, "threadIdx.y")

    for idx in [0, 1]:
        SS = sch.cache_read(C, idx, "shared")
        sch.compute_at(SS, K_outer)
        ax_M, ax_N = sch.get_loops(SS)[-2:]
        fused = sch.fuse(ax_M, ax_N)
        oo, idx_y, idx_x, vec = sch.split(fused, [None, 4, 32, 8])
        sch.bind(idx_x, "threadIdx.x")
        sch.bind(idx_y, "threadIdx.y")
        sch.vectorize(vec)
        sch.storage_align(SS, 0, axis=-2, factor=32, offset=8)

    A_warp = sch.cache_read(C, 0, "wmma.matrix_a")
    B_warp = sch.cache_read(C, 1, "wmma.matrix_b")
    sch.compute_at(A_warp, K_inner)
    sch.compute_at(B_warp, K_inner)

    C_warp = sch.cache_write(C, 0, "wmma.accumulator")
    sch.reverse_compute_at(C_warp, warp)
    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)

    sch.decompose_reduction(C, sch.get_loops(C)[2])
    block_init_c = sch.get_block("output_init")
    def tile_wmma_fragment(block_read, height, width):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, width])
        sch.reorder(i0, j0, i1, j1)
        return i1
    tile_wmma_fragment(A_warp, 16, 16)
    tile_wmma_fragment(B_warp, 16, 16)
    sch.tensorize(sch.get_loops(A_warp)[-2], "wmma_load_16x16x16_f16_a")
    sch.tensorize(sch.get_loops(B_warp)[-2], "wmma_load_16x16x16_f16_b")
    sch.tensorize(sch.get_loops(block_init_c)[-2], WMMA_FILL_16x16x16_F16_INTRIN)
    sch.tensorize(sch.get_loops(C_warp)[-2], WMMA_STORE_16x16x16_F16_SHARED_INTRIN)
    sch.tensorize(sch.get_loops(C)[-3], WMMA_SYNC_16x16x16_f16f16f16_INTRIN)
    # print(sch.mod["main"].script())
    # exit(0)

def sche_gemm_shared_opt(sch: tvm.tir.Schedule):
    C = sch.get_block("output")

    ax_N, ax_M, ax_K = sch.get_loops(C)
    grid_N, block_N = sch.split(ax_N, factors=[None, 64])
    grid_M, block_M = sch.split(ax_M, factors=[None, 64])
    sch.reorder(grid_N, grid_M, block_N, block_M)
    grid = sch.fuse(grid_N, grid_M)
    sch.bind(grid, "blockIdx.x")

    C_shared = sch.cache_write(C, 0, "shared")
    C_warp = sch.cache_write(C, 0, "wmma.accumulator")

    grid, ax_N, ax_M, ax_K = sch.get_loops(C)
    K_outer, K_inner, wmma_K = sch.split(ax_K, factors=[None, 2, 16])
    warp_N, va_N, wmma_N = sch.split(ax_N, factors=[None, 2, 16])
    warp_M, va_M, wmma_M = sch.split(ax_M, factors=[None, 2, 16])
    sch.reorder(warp_N, warp_M, K_outer, K_inner, va_N, va_M, wmma_N, wmma_M, wmma_K)
    warp = sch.fuse(warp_N, warp_M)
    sch.bind(warp, "threadIdx.y")

    for idx in [0, 1]:
        SS = sch.cache_read(C, idx, "shared")
        sch.compute_at(SS, K_outer)
        ax_M, ax_N = sch.get_loops(SS)[-2:]
        fused = sch.fuse(ax_M, ax_N)
        oo, idx_y, idx_x, vec = sch.split(fused, [None, 4, 32, 8])
        sch.bind(idx_x, "threadIdx.x")
        sch.bind(idx_y, "threadIdx.y")
        sch.vectorize(vec)
        sch.storage_align(SS, 0, axis=-2, factor=32, offset=8)

    A_warp = sch.cache_read(C, 0, "wmma.matrix_a")
    B_warp = sch.cache_read(C, 1, "wmma.matrix_b")
    sch.compute_at(A_warp, K_inner)
    sch.compute_at(B_warp, K_inner)

    sch.reverse_compute_at(C_warp, warp)

    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)
    sch.reverse_compute_at(C_shared, sch.get_loops(C_warp)[-3])
    ax_M, ax_N = sch.get_loops(C_shared)[-2:]
    fused = sch.fuse(ax_M, ax_N)
    oo, idx_x, vec = sch.split(fused, [None, 32, 8])
    sch.bind(idx_x, "threadIdx.x")
    sch.vectorize(vec)
    sch.storage_align(C_warp, 0, axis=-2, factor=32, offset=8)

    sch.decompose_reduction(C, sch.get_loops(C)[2])
    block_init_c = sch.get_block("output_init")
    def tile_wmma_fragment(block_read, height, width):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, width])
        sch.reorder(i0, j0, i1, j1)
        return i1
    tile_wmma_fragment(A_warp, 16, 16)
    tile_wmma_fragment(B_warp, 16, 16)
    sch.tensorize(sch.get_loops(A_warp)[-2], "wmma_load_16x16x16_f16_a")
    sch.tensorize(sch.get_loops(B_warp)[-2], "wmma_load_16x16x16_f16_b")
    sch.tensorize(sch.get_loops(block_init_c)[-2], WMMA_FILL_16x16x16_F16_INTRIN)
    sch.tensorize(sch.get_loops(C_warp)[-2], WMMA_STORE_16x16x16_F16_SHARED_INTRIN)
    sch.tensorize(sch.get_loops(C)[-3], WMMA_SYNC_16x16x16_f16f16f16_INTRIN)

args = gemm(512, 512, 512)
workload = te.create_prim_func(gemm(512, 512, 512))
ir_module = tvm.IRModule({"main": workload})
sch = tvm.tir.Schedule(ir_module)
sche_gemm(sch)
mod = tvm.build(sch.mod["main"], target="cuda")
kernel_code = mod.imported_modules[0].get_source()

kernel_code = kernel_code[kernel_code.index('extern "C" __global__ void'):]
print(kernel_code)
cp = CompileResult(None, kernel_code, [32, 4, 1], [64, 1, 1], "default_function_kernel0", args)
cp.compile_and_load(memopt.arch.V100())
a = cp.get_example_outputs()
print(cp.profile())
print(a)

# from memopt.reference import get_reference_output
# oo = get_reference_output(args)
# print(oo[-1])
