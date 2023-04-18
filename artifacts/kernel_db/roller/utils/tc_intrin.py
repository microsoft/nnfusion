import tvm
from tvm import te

def init_intrin_strides(wmma_shape, warp_s, block_s, rstep_size, offset, layout):
    wmma_x, wmma_y = wmma_shape
    #warp_m, warp_n = warp_shape
    #block_m, block_n = block_shape
    if layout == "row_major":
        # row-major input tensor
        # block-level subtensor shape = [block_size_m, wmma_k]
        # warp-level subtensor shape = [warp_size_m, wmma_k]
        # fragment shape = [warp_m, wmma_k]
        wmma_m, wmma_k = wmma_x, wmma_y
        F_stride = [wmma_k, 1]
        S_stride = [rstep_size * wmma_k + offset, 1]
    if layout == "col_major":
        # col-major input tensor
        # block-level subtensor shape = [wmma_k, block_size_n]
        # warp-level subtensor shape = [wmma_k, warp_size_n]
        # fragment shape = [wmma_k, warp_n]
        wmma_k, wmma_n = wmma_x, wmma_y
        F_stride = [wmma_n * warp_s, 1]
        S_stride = [wmma_n * warp_s * block_s + offset, 1]
    return F_stride, S_stride

def intrin_wmma_load_matrix(wmma_shape, warp_shape, block_shape, rstep, stride_dst, stride_src, scope, layout, data_dtype):
    wmma_m, wmma_n, wmma_k = wmma_shape
    warp_m, warp_n = warp_shape
    block_m, block_n = block_shape
    if layout == "row_major":
        buffer_shape = (wmma_m, wmma_k)
        #stride_dst = [wmma_k, 1]
        #stride_src = [wmma_k, 1]
    elif layout == "col_major":
        buffer_shape = (wmma_k, wmma_n)
        #stride_dst = [wmma_n * warp_n, 1]
        #stride_src = [wmma_n * warp_n * block_n, 1]
    
    A = te.placeholder(buffer_shape, name="A", dtype=data_dtype)
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", strides=stride_src,
        data_alignment=32, offset_factor=8
        )
    C = te.compute(buffer_shape, lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope=scope, strides=stride_dst,
        data_alignment=32, offset_factor=8
        )

    if layout == "row_major":
        warp_index = (BC.elem_offset % (warp_m * wmma_m * wmma_k * rstep)) // (wmma_m * wmma_k)
    elif layout == "col_major":
        warp_index = (BC.elem_offset % (warp_n * wmma_n * rstep)) // wmma_n

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                wmma_m,
                wmma_n,
                wmma_k,
                warp_index,
                BA.access_ptr("r"),
                stride_src[0],
                layout,
            )
        )
        return ib.get()
 
    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_store_matrix(strides_dst, strides_from, F_shape, out_dtype):
    """Intrin function for storing the results from wmma.accumulator to global"""
    wmma_m, wmma_n, wmma_k = F_shape
    A = te.placeholder((wmma_m, wmma_n), name="A", dtype=out_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", data_alignment=32, offset_factor=8, strides=strides_from
    )
    C = te.compute((wmma_m, wmma_n), lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="global", data_alignment=32, offset_factor=8, strides=strides_dst)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_n
        warp_index = BA.elem_offset // row + BA.elem_offset % row // wmma_n
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                wmma_m,
                wmma_n,
                wmma_k,
                warp_index,
                BC.access_ptr("w"),
                strides_dst[0],
                "row_major",
            )
        )
        return ib.get()
    
    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, strides_A, strides_B, strides_C, shape):
    """Intrin for wmma fill_fragment and mma_sync
    Parameters
    ----------
    AL_gemm : tvm.te.placeholder
        wmma matrix A
    WL_gemm : tvm.te.placeholder
        wmma matrix B
    CL_compute : tvm.te.compute
        The definition of wmma gemm
    """
    wmma_m, wmma_n, wmma_k = shape
    A = AL_gemm
    B = BL_gemm
    C = CL_compute

    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        name="BA",
        scope="wmma.matrix_a",
        data_alignment=32,
        offset_factor=8,
        strides=strides_A,
    )
    BB = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        name="BB",
        scope="wmma.matrix_b",
        data_alignment=32,
        offset_factor=8,
        strides=strides_B,
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        data_alignment=32,
        offset_factor=8,
        strides=strides_C,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def warp_idnex(offset, row, col):
            row = row * col
            return offset // row + offset % row // col

        warp_index_A = warp_idnex(BA.elem_offset, wmma_m, wmma_k)
        warp_index_B = warp_idnex(BB.elem_offset, wmma_k, wmma_n)
        warp_index_C = warp_idnex(BC.elem_offset, wmma_m, wmma_n)

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_fill_fragment",
                    BC.data,
                    wmma_m,
                    wmma_n,
                    wmma_k,
                    warp_index_C,
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    warp_index_C,
                    BA.data,
                    warp_index_A,
                    BB.data,
                    warp_index_B,
                    BC.data,
                    warp_index_C,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})
