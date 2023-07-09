import tvm
from tvm import te


def intrin_wmma_load_matrix_A(strides_dst, strides_from, shape, layout, A_shape, C_shape, in_dtype):
    """Intrin function for loading data from shared memory to wmma.matrix_a"""
    wmma_m, wmma_n, wmma_k = shape

    load_matrix_stride = 1
    for stride in reversed(strides_from):
        if stride > 1:
            load_matrix_stride = stride
            break

    A = te.placeholder(A_shape, name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", strides=strides_from, elem_offset=te.var(),
    )
    C = te.compute(C_shape, lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_a",
        strides=strides_dst,
        elem_offset=te.var(),
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_k
        if layout == "row_major":
            warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_k
        else:
            warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_m
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
                load_matrix_stride,
                layout,
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_W(strides_dst, strides_from, shape, layout, A_shape, C_shape, in_dtype):
    """Intrin function for loading data from shared memory to wmma.matrix_b"""
    wmma_m, wmma_n, wmma_k = shape
    load_matrix_stride = 1
    for stride in reversed(strides_from):
        if stride > 1:
            load_matrix_stride = stride
            break

    A = te.placeholder(A_shape, name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", strides=strides_from, elem_offset=te.var(),
    )
    C = te.compute(C_shape, lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_b",
        strides=strides_dst,
        elem_offset=te.var(),
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_n * wmma_k
        if layout == "row_major":
            warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_n
        else:
            warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_k
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
                load_matrix_stride,
                layout,
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_store_matrix(strides_dst, strides_from, shape, out_dtype, A_shape, C_shape, scope):
    """Intrin function for storing the results from wmma.accumulator to shared"""
    wmma_m, wmma_n, wmma_k = shape
    store_matrix_stride = 1
    assert scope in ["shared", "global"]
    for stride in reversed(strides_dst):
        if stride > 1:
            store_matrix_stride = stride
            break
    A = te.placeholder(A_shape, name="A", dtype=out_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        scope="wmma.accumulator",
        strides=strides_from,
        elem_offset=te.var(),
    )
    C = te.compute(C_shape, lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope=scope, strides=[te.var() for _ in C_shape], elem_offset=te.var()
    )

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
                store_matrix_stride,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(AL_gemm, WL_gemm, CL_compute, strides_A, strides_W, strides_Conv, shape, layoutA, layoutB):
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
    B = WL_gemm
    C = CL_compute

    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        name="BA",
        scope="wmma.matrix_a",
        elem_offset=te.var(),
        strides=strides_A,
    )
    BB = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        name="BB",
        scope="wmma.matrix_b",
        elem_offset=te.var(),
        strides=strides_W,
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        elem_offset=te.var(),
        strides=strides_Conv,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def warp_idnex(offset, row, col):
            row = row * col
            return offset // row + offset % row // col

        if layoutA == "row_major":
            warp_index_A = warp_idnex(BA.elem_offset, wmma_m, wmma_k)
        else:
            warp_index_A = warp_idnex(BA.elem_offset, wmma_k, wmma_m)
        if layoutB == "row_major":
            warp_index_B = warp_idnex(BB.elem_offset, wmma_k, wmma_n)
        else:
            warp_index_B = warp_idnex(BB.elem_offset, wmma_n, wmma_k)
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
