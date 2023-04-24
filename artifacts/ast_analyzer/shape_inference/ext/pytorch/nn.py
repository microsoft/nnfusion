import math
import numpy as np

from   chainer.utils import type_check

from   ast_analyzer.shape_inference.ext.utils         import *
from   ast_analyzer.shape_inference.types             import *
from   ast_analyzer.shape_inference.ext.pytorch.utils import *

__all__ = [ 'ty_TorchConv'
          , 'ty_TorchPooling'
          , 'ty_TorchAdaptivePooling'
          , 'ty_TorchDimPad'
          , 'ty_TorchBatchNorm'
          , 'ty_TorchInstanceNorm'
          , 'ty_TorchLSTMCell'
          , 'ty_TorchLinear'
          , 'ty_TorchEmbed'
          , 'ty_TorchNNCrossEntropyLoss'
          , 'ty_TorchPixelShuffle'
          , 'ty_TorchInterpolate'
          ]


# Convolution

def get_conv_outsize(x, kernel_size, stride, padding, dilation):
    return (x + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def get_conv_transpose_outsize(
        x, kernel_size, stride, padding, dilation, output_padding):
    return (x - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + \
            output_padding + 1

class ty_TorchConv():
    def __init__(self, dim, transpose=False):
        self.dim = dim
        self.transpose = transpose

    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args

        check_dtype(obj, x_type.dtype)
        assert x_type.ndim == self.dim + 2, \
                "Conv: dimension mismatch"
        assert x_type.shape[1] == obj.in_channels, \
                "Conv: shape[1] != in_channels"

        return self.infer_return(x_type, obj)

    def infer_return(self, x_type, obj):
        kernel_size = make_tuple(obj.kernel_size, self.dim)
        stride      = make_tuple(obj.stride,      self.dim)
        padding     = make_tuple(obj.padding,     self.dim)
        dilation    = make_tuple(obj.dilation,    self.dim)

        if self.transpose:
            output_padding = make_tuple(obj.output_padding, self.dim)

        shape = [0] * self.dim
        for i in range(self.dim):
            if self.transpose:
                shape[i] = get_conv_transpose_outsize(
                        x_type.shape[i + 2], kernel_size[i], stride[i],
                        padding[i], dilation[i], output_padding[i])
            else:
                shape[i] = get_conv_outsize(
                        x_type.shape[i + 2], kernel_size[i], stride[i],
                        padding[i], dilation[i])
        ret_shape = (x_type.shape[0], obj.out_channels) + tuple(shape)
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


# Pooling

class ty_TorchPooling():
    def __init__(self, dim):
        self.dim = dim

    # TOOD(momohatt): in_channels, out_channels
    def __call__(self, ty_args, ty_kwargs):
        x_type, kernel_size_type = ty_args
        assert x_type.ndim == self.dim + 2
        assert x_type.dtype.kind == 'f' # TODO

        kernel_size = extract_value_from_ty(kernel_size_type)
        stride, _    = get_kwarg(ty_kwargs, 'stride', default=kernel_size)
        padding, _   = get_kwarg(ty_kwargs, 'padding', default=0)
        ceil_mode, _ = get_kwarg(ty_kwargs, 'ceil_mode', default=False)
        return self.infer_return(x_type, kernel_size, stride, padding, ceil_mode)

    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.ndim == self.dim + 2
        check_dtype(obj, x_type.dtype)

        kernel_size = obj.kernel_size
        stride = obj.stride
        padding = obj.padding
        ceil_mode = obj.ceil_mode
        return self.infer_return(x_type, kernel_size, stride, padding, ceil_mode)

    def infer_return(self, x_type, kernel_size, stride, padding, ceil_mode):
        padding = make_tuple(padding, self.dim)
        kernel_size = make_tuple(kernel_size, self.dim)
        stride = make_tuple(stride, self.dim)
        shape = [0] * (self.dim + 2)

        shape[0] = x_type.shape[0]
        shape[1] = x_type.shape[1]
        if ceil_mode:
            for i in range(self.dim):
                shape[i + 2] = math.ceil((x_type.shape[i + 2] + padding[i] * 2 - kernel_size[i]) / stride[i]) + 1
        else:
            for i in range(self.dim):
                shape[i + 2] = (x_type.shape[i + 2] + padding[i] * 2 - kernel_size[i]) // stride[i] + 1

        return TyTorchTensor(x_type.dtype, shape=tuple(shape))


class ty_TorchAdaptivePooling():
    def __init__(self, dim):
        self.dim = dim

    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        output_size = obj.output_size
        shape = x_type.shape[:-self.dim] + wrap_shape(output_size)
        return TyTorchTensor(x_type.dtype, shape=shape)


# Padding

class ty_TorchDimPad():
    def __init__(self, dim, is_const=False):
        self.dim = dim
        self.is_const = is_const

    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.ndim == self.dim + 2

        if self.is_const:
            if type(obj.value) is int:
                assert x_type.dtype.kind == 'i'
            elif type(obj.value) is float:
                assert x_type.dtype.kind == 'f'

        padding = make_tuple(obj.padding, self.dim + 2)
        return self.infer_return(x_type, padding)

    def infer_return(self, x_type, padding):
        shape = list(x_type.shape)
        for i in range(self.dim):
            shape[i + 2] = shape[i + 2] + padding[- (2 * i + 1)] + \
                    padding[- (2 * i + 2)]
        return TyTorchTensor(x_type.dtype, shape=shape)


# Non-linear activations (weighted sum, nonlinearity)

# Non-linear activations (other)

# Normalization

class ty_TorchBatchNorm():
    def __init__(self, dim):
        self.dim = dim

    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.ndim == self.dim + 2 or x_type.ndim == 2 and self.dim == 1, \
                "BatchNorm: dimension mismatch"
        assert x_type.shape[1] == obj.num_features, \
                "BatchNorm: shape[1] != num_features"
        check_dtype(obj, x_type.dtype)
        return x_type


class ty_TorchInstanceNorm():
    def __init__(self, dim):
        self.dim = dim

    def nn(self, obj, ty_args, ty_kwargs):
        # TODO: Need any check?
        x_type, = ty_args
        assert x_type.ndim == self.dim + 2
        check_dtype(obj, x_type.dtype)
        return x_type


# Recurrent

class ty_TorchLSTMCell():
    def nn(self, obj, ty_args, ty_kwargs):
        input_size = obj.input_size
        hidden_size = obj.hidden_size
        input_type = ty_args[0]
        assert isinstance(ty_args[1], TyTuple)
        assert ty_args[1].is_fixed_len
        h_0_type, c_0_type = ty_args[1].get_tys()

        batch = input_type.shape[0]
        assert input_type.shape[1] == input_size
        assert h_0_type.shape[0] == batch
        assert h_0_type.shape[1] == hidden_size
        assert c_0_type.shape[0] == batch
        assert c_0_type.shape[1] == hidden_size

        return TyTuple([copy_ty(h_0_type), copy_ty(c_0_type)])


# Transformer

# Linear

class ty_TorchLinear():
    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        check_dtype(obj, x_type.dtype)
        assert x_type.shape[-1] == obj.in_features
        return self.infer_return_shape(x_type, obj.out_features)

    def infer_return_shape(self, x_type, out_features):
        out_shape = x_type.shape[:-1] + (out_features,)
        return TyTorchTensor(x_type.dtype, shape=out_shape)


# Dropout

# Sparse

class ty_TorchEmbed():
    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.dtype == np.int64, "dtype of input must be int64"
        embedding_dim = obj.embedding_dim
        shape = x_type.shape + (embedding_dim,)
        # TODO(momohatt): Use global dtype
        return TyTorchTensor(np.float32, shape=shape)

    def __call__(self, ty_args, ty_kwargs):
        x_type, weight_type = ty_args
        assert x_type.dtype == np.int64, "dtype of input must be int64"
        assert weight_type.dtype.kind == 'f'
        assert weight_type.ndim == 2
        embedding_dim = weight_type.shape[1]
        shape = x_type.shape + (embedding_dim,)
        return TyTorchTensor(weight_type.dtype, shape=shape)


# Distance functions

# Loss functions

class ty_TorchNNCrossEntropyLoss():
    def nn(self, _, ty_args, ty_kwargs):
        x_type, t_type = ty_args
        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x', 't')))
        return self.infer_return(x_type, t_type)

    def __call__(self, ty_args, ty_kwargs):
        x_type, t_type = ty_args
        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x', 't')))
        return self.infer_return(x_type, t_type)

    def check_type_forward(self, in_types):
        x_type, t_type = in_types

        type_check.expect(
                x_type.dtype.kind == 'f',
                t_type.dtype.kind == 'i',
                t_type.ndim == x_type.ndim - 1,
                x_type.shape[0] == t_type.shape[0],
                x_type.shape[2:] == t_type.shape[1:]
                )

    def infer_return(self, x_type, t_type):
        return TyTorchTensor(x_type.dtype, shape=())


# Vision

class ty_TorchPixelShuffle():
    def nn(self, obj, ty_args, ty_kwargs):
        upscale_factor = obj.upscale_factor
        x_type, = ty_args
        assert x_type.ndim == 4
        return self.infer_return(x_type, upscale_factor)

    def infer_return(self, x_type, upscale_factor):
        shape = list(x_type.shape)
        shape[1] //= upscale_factor ** 2
        shape[2] *= upscale_factor
        shape[3] *= upscale_factor
        return TyTorchTensor(x_type.dtype, shape=shape)


class ty_TorchInterpolate():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.ndim >= 3 and x_type.ndim <= 5
        size, lacks_size = get_kwarg(ty_kwargs, 'size', None)
        scale_factor, lacks_scale_factor = get_kwarg(ty_kwargs, 'scale_factor', None)
        assert not lacks_size and not lacks_scale_factor
        return self.infer_return(x_type, size, scale_factor)

    def infer_return(self, x_type, size, scale_factor):
        shape = list(x_type.shape)
        if size is not None:
            if isinstance(size, tuple):
                for (i, x) in zip(range(2, x_type.ndim), size):
                    shape[i] = x
            else:
                for i in range(2, x_type.ndim):
                    shape[i] = size
        elif scale_factor is not None:
            if isinstance(scale_factor, tuple):
                for (i, r) in zip(range(2, x_type.ndim), scale_factor):
                    shape[i] = math.floor(r * shape[i])
            else:
                for i in range(2, x_type.ndim):
                    shape[i] = math.floor(scale_factor * shape[i])
        return TyTorchTensor(x_type.dtype, shape=wrap_shape(shape))


# DataParallel (multi-GPU, distributed)

# Utilities

