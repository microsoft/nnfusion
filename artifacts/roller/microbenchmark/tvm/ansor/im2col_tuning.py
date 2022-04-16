import sys
import tvm
from tvm import te, auto_scheduler, topi
from tvm.contrib import nvcc
from tvm.topi import nn
import os, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
#from antares.lang.codegen import *

profile_mode = "profile_time"
#profile_mode = "profile_metrics"

BACKEND = "tvm"
#BACKEND = "antares"

def equal_const_int(expr, value):
    """Returns if expr equals value.
    Parameters
    ----------
    expr : tvm.Expr
        The input expression.
    Returns
    -------
    equal : bool
        Whether they equals.
    """
    if isinstance(expr, Integral):
        return expr == value
    if not isinstance(expr, tvm.tir.IntImm):
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
        return False
    return expr.value == value

def pad(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput"):
    """Pad Input with zeros.
    Parameters
    ----------
    data : tvm.te.Tensor
        n-D input, can be any layout.
    pad_before : list / tuple of n ints
        Pad width on each dimension to pad the before the axis begin.
    pad_after : list / tuple of n ints, optional
        Pad width each dimension to pad the after the axis end.
    pad_value : float, optional
        The value to be padded.
    name : str, optional
        The name prefix operators generated
    Returns
    -------
    Output : tvm.te.Tensor
        n-D, the same layout as Input.
    """
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError(
            "Input dimension and pad_before dismatch : %d vs %d" % (n, len(pad_before))
        )
    if len(pad_after) != n:
        raise ValueError("Input dimension and pad_after dismatch : %d vs %d" % (n, len(pad_before)))
    ana = tvm.arith.Analyzer()
    dshape = []
    for dim in data.shape:
        if isinstance(dim, tvm.tir.Any):
            dshape.append(tvm.te.size_var("dim"))
        else:
            dshape.append(dim)
    out_shape = tuple(ana.simplify(dshape[i] + pad_before[i] + pad_after[i]) for i in range(n))
    pad_value = (
        pad_value
        if isinstance(pad_value, tvm.tir.PrimExpr)
        else tvm.tir.const(pad_value, data.dtype)
    )

    def _pad(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                index_tuple.append(indices[i])
            else:
                index_tuple.append(indices[i] - pad_before[i])
                not_zero.append(indices[i] >= pad_before[i])
                not_zero.append(indices[i] < data.shape[i] + pad_before[i])
        if not_zero:
            not_zero = tvm.tir.all(*not_zero)
            return tvm.tir.if_then_else(not_zero, data(*index_tuple), pad_value)
        return data(*index_tuple)

    return te.compute(out_shape, _pad, name=name)

def simplify(expr):
    """Simplify the expression if it is Expr, directly return if it is int.
    Parameters
    ----------
    expr : Expr or int
        The input.
    Returns
    -------
    out : Expr or int
        The simplified output
    """
    return tvm.arith.Analyzer().simplify(expr) if isinstance(expr, tvm.tir.PrimExpr) else expr

def get_pad_tuple(padding, kernel):
    """Common code to get the pad option
    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    kernel : tuple of int
        Conv kernel size
    Returns
    -------
    pad_top : int
        Padding size on top
    pad_left : int
        Padding size on left
    pad_down : int
        Padding size on down.
    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left

def conv2d_im2col(batch, channel, num_filter, kernel, stride, in_height, in_width, padding, dilation = 1):
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(kernel, int):
        kernel_h = kernel_w = kernel
    else:
        kernel_h, kernel_w = kernel

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    M = batch * out_height * out_width
    N = num_filter
    K = channel * kernel_h * kernel_w

    Data_Matform = te.placeholder((M, K), name="data")
    Filter_Matform = te.placeholder((K, N), name="filter")
    k = te.reduce_axis((0, K), name="k")
    Out = te.compute((M, N), lambda y, x: te.sum(Data_Matform[y, k] * Filter_Matform[k, x], axis=k))

    return Out, Data_Matform, Filter_Matform

@auto_scheduler.register_workload
def conv2d_im2col_func(N, C, F, K, S, H, W, P):
    conv, data_matform, filter_matform = conv2d_im2col(N, C, F, K, S, H, W, P)
    return [data_matform, filter_matform, conv]


if __name__ == "__main__":
    #sys.path.append("/usr/local/cuda-10.2/bin/")
    os.system("export PATH=$PATH:/usr/local/cuda-10.2/bin/")
    #N, C, F, K, S, H, W, P = 64, 3, 64, 7, 2, 230, 230, 0
    #N, C, F, K, S, H, W, P = 64, 64, 64, 3, 1, 56, 56, 1
    #N, C, F, K, S, H, W, P = 64, 128, 128, 3, 1, 28, 28, 1
    N, C, F, K, S, H, W, P = 64, 168, 168, 1, 1, 42, 42, 0 
    #N, C, F, K, S, H, W, P = 64, 336, 336, 1, 1, 21, 21, 0
    #N, C, F, K, S, H, W, P = 64, 672, 672, 1, 1, 11, 11, 0
    #N, C, F, K, S, H, W, P = 64, 32, 64, 1, 1, 112, 112, 0
    #N, C, F, K, S, H, W, P = 64, 128, 128, 1, 1, 56, 56, 0
    D = 1
    if len(sys.argv) == 11:
        N = int(sys.argv[1])
        C = int(sys.argv[2])
        H = int(sys.argv[3])
        W = int(sys.argv[4])
        F = int(sys.argv[5])
        K = int(sys.argv[6])
        K = int(sys.argv[7])
        S = int(sys.argv[8])
        D = int(sys.argv[9])
        P = int(sys.argv[10])
    print("N, C, H, W, F, K, S, D, P:", N, C, H, W, F, K, S, D, P)

    #s[conv.op].fuse(*list(s[conv].op.reduce_axis))
    #fused_axis = s[conv.op].fuse(s[conv].op.axis[2], s[conv].op.axis[3])
    #s[conv.op].split(fused_axis, 32)

    #print(tvm.lower(s, [conv, data, kernel], simple_mode = True))
    #func = tvm.build(s, [data, kernel, conv], "cuda")
    #source = func.imported_modules[0].get_source()
    #print(source)
    
    # fuse axes, all reduce axes and hw axes of image
    target = tvm.target.Target("cuda")

    task = auto_scheduler.SearchTask(
        func=conv2d_im2col_func, args=(N, C, F, K, S, H, W, P), target=target
    )
    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "conv2d.json"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,  # change this to 1000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    task.tune(tune_option)
    sch, args = task.apply_best(log_file)

    # Kill the measurement process
    del measure_ctx

    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
