import tvm
from tvm import te
from .tc_intrin import (
    init_intrin_strides,
    intrin_wmma_load_matrix,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)
import math

def schedule_tensorcore(tvm_schedule, rprog, C, verbose=False):
    """
        Schedule dense operator using Tensorcore
    """
    #if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
    #    s[B].compute_inline()
    #batch, out_dim = get_const_tuple(C.shape)
    Mdim, Ndim = C.shape
    s = tvm_schedule
    A, B = s[C].op.input_tensors
    data_dtype = A.dtype
    out_dtype = C.dtype

    # Explicit memory access
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")
    #CS = s.cache_read(CF, "shared", [C])

    if verbose:
        print("0========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")

    # Extract Sokoban scheduling information
    warp_size = 32
    wmma_m, wmma_n = rprog.GetTile(2).SDimensions()
    wmma_k = rprog.GetTile(2).RDimensions()[0]
    warp_m, warp_n = rprog.GetTile(1).SDimensions()
    block_m, block_n = rprog.GetTile(0).SDimensions()
    rstep_size = rprog.GetTile(0).RDimensions()[0] // wmma_k

    block_row_warps = block_m // warp_m
    block_col_warps = block_n // warp_n
    warp_row_tiles = warp_m // wmma_m
    warp_col_tiles = warp_n // wmma_n
    offset = 8
    offsetCS = 0
    vec = 1

    # Define the stride of intrin functions
    #CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    #AS_stride = [AS_align, 1]
    #BS_stride = [BS_align, 1]
    #AF_stride = [wmma_k, 1]
    #BF_stride = [wmma_k, 1]
    AF_stride, AS_stride = init_intrin_strides([wmma_m, wmma_k], warp_row_tiles, block_row_warps, rstep_size, offset, "row_major")
    BF_stride, BS_stride = init_intrin_strides([wmma_k, wmma_n], warp_col_tiles, block_col_warps, rstep_size, offset, "col_major")
    AS_align = AS_stride[0]
    BS_align = BS_stride[0]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    C_stride = [Ndim, 1]

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for dense computation
    block_factor_m = wmma_m * warp_row_tiles * block_row_warps
    block_factor_n = wmma_n * warp_col_tiles * block_col_warps
    m, n = C.op.axis
        
    block_i, mc = s[C].split(m, factor=block_factor_m)
    block_j, nc = s[C].split(n, factor=block_factor_n)
    mm, mmi = s[C].split(mc, factor=wmma_m)
    nn, nni = s[C].split(nc, factor=wmma_n)
    mm, mmii = s[C].split(mm, factor=warp_row_tiles)
    nn, nnii = s[C].split(nn, factor=warp_col_tiles)
    s[C].reorder(block_i, block_j, mm, nn, mmii, nnii, mmi, nni)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    if verbose:
        print("i========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")
    s[C].bind(mm, thread_y)
    s[C].bind(nn, thread_z)
    s[C].unroll(mmii)
    s[C].unroll(nnii)

    if verbose:
        print("ii========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")

    # Schedule for wmma computation
    s[CF].compute_at(s[C], nn)
    warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    (k,) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=rstep_size)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)
    s[CF].unroll(ki)
    s[CF].unroll(warp_i)
    s[CF].unroll(warp_j)

    if verbose:
        print("iii========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")
    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    m, i = AF.op.axis
    m, m_ii = s[AF].split(m, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(m, i, m_ii, i_jj)
    s[AF].unroll(m)
    s[AF].unroll(i)

    if verbose:
        print("iv========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")
    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    i, n = BF.op.axis
    i, i_ii = s[BF].split(i, factor=wmma_k)
    n, n_ii = s[BF].split(n, factor=wmma_n)
    s[BF].reorder(i, n, i_ii, n_ii)
    s[BF].unroll(i)
    s[BF].unroll(n)

    if verbose:
        print("v========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")
    # Schedule for A's(B's) shared memory load
        
    def shared_shedule(stage, strides):
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        t, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].unroll(t)
        s[stage].vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    if verbose:
        print("vi========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")

    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=data_dtype)
    BL_gemm = te.placeholder((wmma_k, wmma_n), name="BL_gemm", dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[k_gemm, jj].astype(out_dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    # lower the computation loops down to TensorCore hardware intrinsics
    # by mapping the dense tensorcore to tensor intrinsics
    s[AF].tensorize(
        m_ii,
        intrin_wmma_load_matrix(
            (wmma_m, wmma_n, wmma_k), (warp_row_tiles, warp_col_tiles), (block_row_warps, block_col_warps), rstep_size, AF_stride, AS_stride, "wmma.matrix_a", "row_major", data_dtype
        ),
    )
    if verbose:
        print("vii========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")

    s[BF].tensorize(
        i_ii,
        intrin_wmma_load_matrix(
            (wmma_m, wmma_n, wmma_k), (warp_row_tiles, warp_col_tiles), (block_row_warps, block_col_warps), rstep_size, BF_stride, BS_stride, "wmma.matrix_b", "col_major", data_dtype
        ),
    )
    if verbose:
        print("viii========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")

    s[CF].tensorize(
        _ii, intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape)
    )
    if verbose:
        print("ix========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")

    s[C].tensorize(
        mmi,
        intrin_wmma_store_matrix(
            C_stride, CF_stride, shape, out_dtype
        ),
    )
    if verbose:
        print("x========================================================================")
        print(tvm.lower(s, [A, B, C]))
        print("========================================================================")
    return s

BACKEND = "tvm"

def tc_mm_main_template(source, M, K, N, grid_x, grid_y, block_x, block_y, block_z, times):
    if BACKEND == "antares":
        kernel_name = "template_op_kernel0"
    if BACKEND == "tvm":
        kernel_name = "default_function_kernel0"
    return '#include <cuda_runtime.h>\n' \
    '#include <stdio.h>\n' \
    '#include <stdlib.h>\n' \
    '#include "cu_helper.h"\n' \
    '#include <cuda_fp16.h>\n' \
    '#include <mma.h>\n' \
    '#include <string>\n' \
    '\n' \
    'int M = {}, K = {}, N = {};\n' \
    '\n' \
    '{}' \
    '\n' \
    'int main(int argc, char *argv[])\n' \
    '{{\n' \
    '    std::string path;\n' \
    '    int input_size0 = M * K;\n' \
    '    int input_size1 = K * N;\n' \
    '    int output_size = N * M;\n' \
    '\n' \
    '    checkCudaErrors(cuInit(0));\n' \
    '    CUdevice device;\n' \
    '    checkCudaErrors(cuDeviceGet(&device, 0));\n' \
    '    CUcontext context;\n' \
    '    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));\n' \
    '\n' \
    '    half *Ah, *Bh;\n' \
    '    half *Ad, *Bd, *Cd;\n' \
    '    Ah = (half*)malloc(input_size0 * sizeof(half));\n' \
    '    Bh = (half*)malloc(input_size1 * sizeof(half));\n' \
    '\n' \
    '    cudaMalloc((void **)&Ad, input_size0 * sizeof(half));\n' \
    '    cudaMalloc((void **)&Bd, input_size1 * sizeof(half));\n' \
    '    cudaMalloc((void **)&Cd, output_size * sizeof(half));\n' \
    '\n' \
    '    srand(1);\n' \
    '    for (int i = 0; i < input_size0; ++ i)\n' \
    '        Ah[i] = __float2half(1);\n' \
    '    for (int i = 0; i < input_size1; ++ i)\n' \
    '        Bh[i] = __float2half(1);\n' \
    '\n' \
    '    cudaMemcpy(Ad, Ah, input_size0 * sizeof(half), cudaMemcpyHostToDevice);\n' \
    '    cudaMemcpy(Bd, Bh, input_size1 * sizeof(half), cudaMemcpyHostToDevice);\n' \
    '\n' \
    '    dim3 grid({}, {}, 1);\n' \
    '    dim3 block({}, {}, {});\n' \
    '\n' \
    '    int numBlocks;\n' \
    '    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, {}, {}, 0);\n' \
    '    fprintf(stderr, \"Active blocks per SM = %d\\n\", numBlocks);\n ' \
    '\n' \
    '    for (int i = 0; i < {}; ++i)\n' \
    '    {{\n' \
	'        {}<<<grid, block>>>((half*)Ad, (half*)Bd, (half*)Cd);\n' \
    '        cudaDeviceSynchronize();\n' \
    '    }}\n' \
    '}}\n'.format(M, K, N, source, grid_x, grid_y, block_x, block_y, block_z, kernel_name, block_x * block_y * block_z, times, kernel_name)


def get_tc_mm_source(A, B, C, rprog):
    s = te.create_schedule(C.op)
    s = schedule_tensorcore(s, rprog, C)
    func = tvm.build(s, [A, B, C], "cuda")
    source = func.imported_modules[0].get_source()
    # get rid of prior definitions
    start_pos = source.find("extern")
    return source[start_pos:]

def get_tc_block_size(block_rTile, warp_rTile):
    block_x = 32
    block_y = block_rTile.SDimensions()[0] // warp_rTile.SDimensions()[0]
    block_z = block_rTile.SDimensions()[1] // warp_rTile.SDimensions()[1]
    return block_x, block_y, block_z

def get_tc_grid_size(M, N, block_rTile):
    m = math.ceil(M / block_rTile.SDimensions()[0])
    n = math.ceil(N / block_rTile.SDimensions()[1])
    return m, n
