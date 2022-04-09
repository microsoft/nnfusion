from tvm import te
import tvm

def tvm_matmul(n, m, k):
    A = te.placeholder((n, k), name="A")
    B = te.placeholder((k, m), name="B")
    k = te.reduce_axis((0, k), name="k")
    C = te.compute((n, m), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k))
    return (A, B, C)

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

def conv2d_nchw(Input, Filter, stride, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.
    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]
    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    out_channel = num_filter
    out_height = (in_height - dilated_kernel_h) // stride_h + 1
    out_width = (in_width - dilated_kernel_w) // stride_w + 1
    k = te.reduce_axis((0, in_channel * kernel_h * kernel_w), name="k")
    #ry = te.reduce_axis((0, kernel_h), name="ry")
    #rx = te.reduce_axis((0, kernel_w), name="rx")
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: te.sum(
            Input[nn, k // (kernel_h * kernel_w), yy * stride_h + (k % (kernel_h * kernel_w) // kernel_w) * dilation_h, xx * stride_w + (k % kernel_w) * dilation_w].astype(
                out_dtype
            )
            * Filter[ff, k // (kernel_h * kernel_w), k % (kernel_h * kernel_w) // kernel_w, k % kernel_w].astype(out_dtype),
            axis=[k],
        ),
        tag="conv2d_nchw",
    )

def depthwise_conv2d_nchw(Input, Filter, stride, dilation, out_dtype=None):
    """Depthwise convolution nchw forward operator.
    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    Filter : tvm.te.Tensor
        4-D with shape [in_channel, channel_multiplier, filter_height, filter_width]
    stride : int or a list/tuple of two ints
        The spatial stride, or (stride_height, stride_width).
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype: str, optional
        Output data type
    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    # shape of dilated kernel
    filter_channel, channel_multiplier, filter_height, filter_width = Filter.shape

    dilated_kernel_h = (filter_height - 1) * dilation_h + 1
    dilated_kernel_w = (filter_width - 1) * dilation_w + 1

    out_channel = in_channel * channel_multiplier
    out_height = (in_height - dilated_kernel_h) // stride_h + 1
    out_width = (in_width - dilated_kernel_w) // stride_w + 1
    # depthconv stage
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, filter_height * filter_width), name="k")
    #di = te.reduce_axis((0, filter_height), name="di")
    #dj = te.reduce_axis((0, filter_width), name="dj")
    Conv = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j: te.sum(
            (
                Input[
                    b,
                    idxdiv(c, channel_multiplier),
                    #i * stride_h + di * dilation_h,
                    #j * stride_w + dj * dilation_w,
                    i * stride_h + (k // filter_width) * dilation_h,
                    j * stride_w + (k % filter_height) * dilation_w,
                ].astype(out_dtype)
                * Filter[
                    idxdiv(c, channel_multiplier), idxmod(c, channel_multiplier), k // filter_width, k % filter_height
                ].astype(out_dtype)
            ),
            axis=[k],
        ),
    )
    return Conv

def padding(X, pd, pt, pl, pr, val=0):
    """Pad X with the given value in 2-D
    ph, pw : height and width padding
    val : padding value, default 0
    """
    assert len(X.shape) >= 2
    nh, nw = X.shape[-2], X.shape[-1]
    return te.compute(
            (*X.shape[0:-2], nh+pd+pt, nw+pl+pr),
            lambda *i: te.if_then_else(
                te.any(i[-2]<pd, i[-2]>=nh+pt, i[-1]<pl, i[-1]>=nw+pr),
                val, X[i[:-2]+(i[-2]-pd, i[-1]-pl)]),
            name='PaddedX')

def tvm_conv(N, C, H, W, F, K, S, D, P="SAME"):
    pt, pl, pd, pr = get_pad_tuple(P, (K, K))
    X = te.placeholder((N, C, H, W), name="X")
    PaddedX = padding(X, pd, pt, pl, pr) if (pd + pt) * (pl + pr) != 0 else X
    kernel = te.placeholder((F, C, K, K), name="K")
    conv = conv2d_nchw(PaddedX, kernel, S, D)
    return (X, kernel, conv)

def tvm_depthwise_conv(N, C, H, W, K, S, D, P="SAME", M=1):
    pt, pl, pd, pr = get_pad_tuple(P, (K, K))
    X = te.placeholder((N, C, H, W), name="X")
    PaddedX = padding(X, pd, pt, pl, pr) if (pd + pt) * (pl + pr) != 0 else X
    kernel = te.placeholder((C, M, K, K), name="K")
    conv = depthwise_conv2d_nchw(PaddedX, kernel, S, D)
    return (X, kernel, conv)
