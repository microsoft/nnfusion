import tvm
import welder
from tvm import te
from welder.utils import CompileResult

tvm.register_func("tvm_callback_cuda_compile", override=True)(lambda x:"")

def gemm(n, m, k):
    """TVM expression for vector add"""
    A = te.placeholder((n, k), name='a')
    B = te.placeholder((k, m), name='b')
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
    K_outer, K_inner = sch.split(ax_K, factors=[None, 32])
    va_N, thread_N = sch.split(ax_N, factors=[None, 16])
    va_M, thread_M = sch.split(ax_M, factors=[None, 8])
    sch.reorder(K_outer, K_inner, va_N, va_M, thread_N, thread_M)
    va = sch.fuse(va_N, va_M)
    thread = sch.fuse(thread_N, thread_M)
    sch.bind(va, "vthread.x")
    sch.bind(thread, "threadIdx.x")

    for idx in [0, 1]:
        SS = sch.cache_read(C, idx, "shared")
        if idx == 0:
            sch.transform_layout(SS, buffer=("write",0), index_map=lambda i, j: [j, i])
        sch.compute_at(SS, K_outer)
        ax_M, ax_N = sch.get_loops(SS)[-2:]
        fused = sch.fuse(ax_M, ax_N)
        oo, ii = sch.split(fused, [None, 128])
        sch.bind(ii, "threadIdx.x")

    C_L = sch.cache_write(C, 0, "local")
    sch.reverse_compute_at(C_L, grid)
    _, ax_N, ax_M = sch.get_loops(C_L)
    va_N, thread_N = sch.split(ax_N, factors=[None, 16])
    va_M, thread_M = sch.split(ax_M, factors=[None, 8])
    sch.reorder(va_N, va_M, thread_N, thread_M)
    thread = sch.fuse(thread_N, thread_M)
    va = sch.fuse(va_N, va_M)
    sch.bind(va, "vthread.x")
    sch.bind(thread, "threadIdx.x")

    sch.decompose_reduction(C, sch.get_loops(C)[1])
    block_init_c = sch.get_block("output_init")
    initc_grid, initc_va, initc_thread = sch.get_loops(block_init_c)
    sch.bind(initc_va, "vthread.x")
    sch.bind(initc_thread, "threadIdx.x")
    # print(sch.mod["main"].script())
    # exit()

args = gemm(512, 512, 512)
workload = te.create_prim_func(gemm(512, 512, 512))
ir_module = tvm.IRModule({"main": workload})
sch = tvm.tir.Schedule(ir_module)
sche_gemm(sch)
mod = tvm.build(sch.mod["main"], target="cuda")
kernel_code = mod.imported_modules[0].get_source()

cp = CompileResult(None, kernel_code, [128, 1, 1], [64, 1, 1], "default_function_kernel0", args)
cp.compile_and_load(welder.arch.V100())
a = cp.get_example_outputs()
print(cp.profile())
print(a)
from welder.reference import get_reference_output

oo = get_reference_output(args)
print(oo[-1])
