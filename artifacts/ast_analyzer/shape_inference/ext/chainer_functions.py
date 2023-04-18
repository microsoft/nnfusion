import chainer
from   chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
import math

from   chainer.utils.conv import get_conv_outsize
from   chainer.utils import type_check

from   ast_analyzer.shape_inference.ext.common import *
from   ast_analyzer.shape_inference.ext.utils  import *
from   ast_analyzer.shape_inference.shape_elem import *
from   ast_analyzer.shape_inference.types      import *
from   ast_analyzer.shape_inference.utils      import all_same

__all__ = [ 'chainer_attr_ty', 'chainer_func_ty', 'chainer_callable_ty' ]


class ty_ChainerVariable():
    def __call__(self, ty_args, ty_kwargs):
        # XXX: data=Noneはkwargsでないと仮定
        assert len(ty_args) > 0
        data_type, = ty_args
        return TyChainerVariable(data_type.dtype, shape=data_type.shape)


class ty_ChainerPooling2d():
    # max_pooling_2d / average_pooling_2d
    def __init__(self, cover_all=None):
        self.cover_all = cover_all

    def __call__(self, ty_args, ty_kwargs):
        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x', 'ksize')))

        x_type, ksize_type = ty_args
        ksize = extract_value_from_ty(ksize_type)
        stride, _ = get_kwarg(ty_kwargs, 'stride', default=ksize)
        pad, _ = get_kwarg(ty_kwargs, 'pad', default=0)

        if self.cover_all is None:
            self.cover_all, _ = get_kwarg(ty_kwargs, 'cover_all', default=True)

        return self.infer_return(x_type, ksize, stride, pad)

    def check_type_forward(self, in_types):
        x_type = in_types[0]

        type_check.expect(
                x_type.dtype.kind == 'f',
                )

        # assert isinstance(ksize_type, TyNum)

    def infer_return(self, x_type, ksize, stride, pad):
        pad = make_pair(pad)
        ksize = make_pair(ksize)
        stride = make_pair(stride)

        shape_0 = x_type.shape[0]
        shape_1 = x_type.shape[1]
        if self.cover_all:
            shape_2 = math.ceil((x_type.shape[2] + pad[0] * 2 - ksize[0]) / stride[0]) + 1
            shape_3 = math.ceil((x_type.shape[3] + pad[1] * 2 - ksize[1]) / stride[1]) + 1
        else:
            shape_2 = (x_type.shape[2] + pad[0] * 2 - ksize[0]) // stride[0] + 1
            shape_3 = (x_type.shape[3] + pad[1] * 2 - ksize[1]) // stride[1] + 1

        return TyChainerVariable(x_type.dtype,
                shape=(shape_0, shape_1, shape_2, shape_3))


class ty_ChainerSoftmaxCrossEntropy():
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
        return TyChainerVariable(x_type.dtype, shape=())


class ty_ChainerIdentical():
    # functions that doesn't change shapes or dtypes

    def __init__(self, is_float_only=True):
        self.is_float_only = is_float_only

    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        if self.is_float_only:
            assert x_type.dtype.kind == 'f'
        return copy_ty(x_type)


# ========================= chainer.functions.array ============================

class ty_ChainerConcat():
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        assert isinstance(xs_type, (TyList, TyTuple))
        x_type = xs_type.get()
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', default=1)

        if lacks_axis:
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        if lacks_value(xs_type):
            ret_shape = list(x_type.shape)
            ret_shape[self.axis] = None
            return TyChainerVariable(x_type.dtype, shape=ret_shape)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.make_variable(self.axis, 'axis'))

        type_check.expect(
            -in_types[0].ndim <= self.axis,
            self.axis < in_types[0].ndim
        )
        ndim = type_check.eval(in_types[0].ndim)
        axis = self.axis % ndim
        for i in range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in range(0, ndim):
                if d == axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        ret_shape = list(xs_type[0].shape)
        ret_shape[self.axis] = sum([x_type.shape[self.axis] for x_type in xs_type])
        return TyChainerVariable(dtype=xs_type[0].dtype, shape=ret_shape)


class ty_ChainerStack():
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        assert isinstance(xs_type, (TyList, TyTuple))
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', default=1)

        if lacks_value(xs_type) or lacks_axis:
            x_type = xs_type.get()
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim + 1)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        self.axis = xs_type.get().ndim + 1 - abs(self.axis)
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(
            -in_types[0].ndim - 1 <= self.axis,
            self.axis <= in_types[0].ndim
        )

        # XXX: modified
        for i in range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].shape == in_types[i].shape,
            )

        # XXX: the following doesn't work
        # dtype = in_types[0].dtype
        # shape = in_types[0].shape
        # for x_type in in_types[1:]:
        #     type_check.expect(
        #         x_type.dtype == dtype,
        #         x_type.shape == shape,
        #     )

    def infer_return(self, xs_type):
        ret_shape = list(xs_type[0].shape)
        ret_shape.insert(self.axis, ShapeElem(xs_type.size()))
        return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)


class ty_ChainerHstack():
    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        assert isinstance(xs_type, (TyList, TyTuple))

        if isinstance(xs_type, TyList) or not xs_type.is_fixed_len:
            x_type = xs_type.get()
            ret_shape = list(x_type.shape)
            if x_type.ndim < 2:
                ret_shape = (None,)
            else:
                ret_shape = (x_type.shape[0], None) + x_type.shape[2:]
            return TyChainerVariable(x_type.dtype, shape=ret_shape)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check._argname((in_types[0],), ('x0',))

        ndim = type_check.eval(in_types[0].ndim)
        for i in range(1, type_check.eval(in_types.size())):
            type_check._argname((in_types[i],), ('x{}'.format(i),))
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                continue
            for d in range(0, ndim):
                if d == 1:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        if xs_type[0].ndim <= 1:
            return ty_ChainerConcat(axis=0).infer_return(xs_type)
        return ty_ChainerConcat(axis=1).infer_return(xs_type)


class ty_ChainerVstack():
    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args

        assert isinstance(xs_type, (TyList, TyTuple))
        if isinstance(xs_type, TyList) or not xs_type.is_fixed_len:
            x_type = xs_type.get()
            if x_type.ndim < 2:
                ret_shape = (None, x_type.shape[0])
            else:
                ret_shape = (None,) + x_type.shape[1:]
            return TyChainerVariable(x_type.dtype, shape=ret_shape)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        ndim = type_check.eval(in_types[0].ndim)
        for i in range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                type_check.expect(in_types[0].shape == in_types[i].shape)
                continue
            for d in range(1, ndim):
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        if xs_type[0].ndim <= 1:
            return ty_ChainerStack(axis=0).infer_return(xs_type)
        return ty_ChainerConcat(axis=0).infer_return(xs_type)


class ty_ChainerExpandDims():
    def __call__(self, ty_args, ty_kwargs):
        x_type, axis_type = ty_args
        self.axis = extract_value_from_ty(axis_type)

        if self.axis is None:
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim + 1)

        self.check_type_forward(make_multiple_tc_variable((x_type,), ('x',)))
        return self.infer_return(x_type)

    def check_type_forward(self, in_types):
        x_type, = in_types
        if self.axis >= 0:
            type_check.expect(x_type.ndim >= self.axis)
        else:
            type_check.expect(x_type.ndim >= -self.axis - 1)

    def infer_return(self, x_type):
        if self.axis < 0:
            self.axis = x_type.ndim + 1 - abs(self.axis)
        ret_shape = list(x_type.shape)
        ret_shape.insert(self.axis, 1)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerBroadcastTo():
    def __call__(self, ty_args, ty_kwargs):
        x_type, shape_type = ty_args

        assert shape_type.is_fixed_len

        out_shape = wrap_shape(extract_value_from_ty(shape_type))

        # TODO: use check_type_forward
        ndim = len(out_shape)
        assert x_type.ndim <= ndim

        for i in range(-1, - x_type.ndim - 1, -1):
            assert x_type.shape[i] == out_shape[i] or x_type.shape[i] == 1

        return TyChainerVariable(x_type.dtype, shape=out_shape)


class ty_ChainerReshape():
    def __call__(self, ty_args, ty_kwargs):
        x_type, shape_type = ty_args

        assert shape_type.is_fixed_len

        self.shape = extract_value_from_ty(shape_type)
        return self.infer_return(x_type)

    def infer_return(self, x_type):
        ret_shape = calculate_reshape(x_type.shape, self.shape)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerRepeat():
    def __init__(self):
        self.axis = None

    def __call__(self, ty_args, ty_kwargs):
        x_type, repeats_type = ty_args
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', 0)

        if lacks_value(repeats_type) or lacks_axis:
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        repeats = extract_value_from_ty(repeats_type)
        self.check_type_forward(x_type, repeats)
        return self.infer_return(x_type, repeats)

    def check_type_forward(self, x_type, repeats):
        # XXX: This is not taken from chainer nor numpy
        if isinstance(repeats, int):
            assert self.axis < x_type.ndim
            return
        assert x_type.shape[self.axis] == len(repeats), "repeat"

    def infer_return(self, x_type, repeats):
        if isinstance(repeats, int):
            if x_type.ndim < 1:
                ret_shape = (repeats,)
            else:
                ret_shape = list(x_type.shape)
                ret_shape[self.axis] = x_type.shape[self.axis] * repeats
        else:
            ret_shape = list(x_type.shape)
            ret_shape[self.axis] = sum(repeats)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerSqueeze():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args

        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', None)
        if isinstance(self.axis, int):
            self.axis = (self.axis,)

        if is_incomplete_shape(x_type.shape):
            # TODO: use ty_kwargs['axis'].size()
            if lacks_axis or self.axis is None:
                assert False, "chainer.fucntions.squeeze: cannot guess ndim of return type"

        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x',)))

        if self.axis is not None:
            for i in self.axis:
                assert x_type.shape[i] == 1, "chainer.fucntions.squeeze: invalid axis"
        return self.infer_return(x_type)

    def check_type_forward(self, in_types):
        # type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        if self.axis is not None:
            for x in self.axis:
                if x >= 0:
                    type_check.expect(x < x_type.ndim)
                else:
                    type_check.expect(-x_type.ndim <= x)

    def infer_return(self, x_type):
        if isinstance(self.axis, tuple):
            ret_shape = remove_dims(x_type.shape, self.axis)
        else:
            ret_shape = [s for s in x_type.shape if s != 1]
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerSum():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', default=None)
        self.keepdims, lacks_keepdims = \
                get_kwarg(ty_kwargs, 'keepdims', default=False)

        if isinstance(self.axis, int):
            self.axis = (self.axis,)

        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x',)))

        if self.axis is None:
            self.axis = tuple(range(x_type.ndim))

        return self.infer_return(x_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types[0].dtype.kind == 'f')

        if self.axis is None:
            return

        for axis in self.axis:
            if axis >= 0:
                type_check.expect(
                    axis < in_types[0].ndim,
                )
            else:
                type_check.expect(
                    -axis - 1 < in_types[0].ndim,
                )

    def infer_return(self, x_type):
        if self.keepdims:
            ret_shape = list(x_type.shape)
            for i in self.axis:
                ret_shape[i] = 1
            return TyChainerVariable(x_type.dtype, shape=ret_shape)

        ret_shape = remove_dims(x_type.shape, self.axis)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerSwapAxes():
    def __call__(self, ty_args, ty_kwargs):
        x_type, axis1_type, axis2_type = ty_args

        if lacks_value(axis1_type) or lacks_value(axis2_type):
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        self.axis1 = extract_value_from_ty(axis1_type)
        self.axis2 = extract_value_from_ty(axis2_type)

        self.check_type_forward(type_check.make_variable(x_type, 'x'))
        return self.infer_return(x_type)

    def check_type_forward(self, x_type):
        type_check.expect(
                self.axis1 < x_type.ndim,
                self.axis2 < x_type.ndim
                )

    def infer_return(self, x_type):
        ret_shape = list(x_type.shape)
        ret_shape[self.axis1], ret_shape[self.axis2] = \
                ret_shape[self.axis2], ret_shape[self.axis1]
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerSeparate():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', 0)

        if lacks_axis:
            return TyTuple(TyChainerVariable(x_type.dtype, ndim=x_type.ndim-1))

        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x',)))
        return self.infer_return(x_type)

    def check_type_forward(self, in_types):
        x_type = in_types[0]
        if self.axis >= 0:
            type_check.expect(self.axis < x_type.ndim)
        else:
            type_check.expect(-self.axis <= x_type.ndim)

    def infer_return(self, x_type):
        n = x_type.shape[self.axis]
        ret_shape = x_type.shape[:self.axis] + x_type.shape[self.axis + 1:]
        ret_ty = TyChainerVariable(x_type.dtype, shape=ret_shape)
        if not n.has_value():
            return TyTuple(ret_ty)
        return TyTuple([ret_ty] * n.value)


class ty_ChainerSplitAxis():
    def __call__(self, ty_args, ty_kwargs):
        x_type, _, axis_type = ty_args

        self.axis = axis_type.value

        if isinstance(ty_args[1], TyNum):
            sections = ty_args[1].value
            return self.infer_return(x_type, sections, is_indices=False)

        # 1-D array
        indices_type = ty_args[1]
        assert isinstance(indices_type, TyTensor)

        assert indices_type.ndim == 1
        n = indices_type.shape[0].value
        return self.infer_return(x_type, n + 1, is_indices=True)

    # TODO: check_type_forward

    def infer_return(self, x_type, n_split, is_indices):
        if n_split is None:
            if self.axis is None:
                return TyTuple(TyChainerVariable(x_type.dtype, ndim=x_type.ndim))
            ret_shape = list(x_type.shape)
            ret_shape[self.axis] = None
            return TyTuple(TyChainerVariable(x_type.dtype, shape=ret_shape))
        ret_shape = list(x_type.shape)
        if is_indices:
            ret_shape[self.axis] = None
        else:
            ret_shape[self.axis] = ret_shape[self.axis] // n_split
        return TyTuple(
            [TyChainerVariable(x_type.dtype, shape=ret_shape)] * n_split)


class ty_ChainerPad():
    def __call__(self, ty_args, ty_kwargs):
        x_type, pad_width_type, mode_type = ty_args

        assert isinstance(mode_type, TyString), \
                "chainer.functions.pad: mode_type should be string"
        self.check_type_forward(make_multiple_tc_variable(ty_args[:1], ('x',)))

        if lacks_value(pad_width_type):
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        assert pad_width_type.size() > 0, \
                "chainer.functions.pad: pad_width is not specified"

        pad_width = extract_value_from_ty(pad_width_type)
        if isinstance(pad_width, int):
            pad_width = make_pair(pad_width)
        if isinstance(pad_width[0], int):
            pad_width = pad_width * x_type.ndim
        for pad in pad_width:
            assert len(pad) == 2, "chainer.functions.pad: pad_width is invalid"
        return self.infer_return(x_type, pad_width)

    def check_type_forward(self, in_types):
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f')

    def infer_return(self, x_type, pad_width):
        ret_shape = list(x_type.shape)
        for i in range(x_type.ndim):
            ret_shape[i] += pad_width[i][0] + pad_width[i][1]
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerPadSequence():
    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        self.length, lacks_length = get_kwarg(ty_kwargs, 'length', None)

        if isinstance(xs_type, TyList) or \
                (isinstance(xs_type, TyTuple) and not xs_type.is_fixed_len):
            ret_shape = list((None,) * (xs_type.get().ndim + 1))
            if not lacks_length:
                ret_shape[1] = self.length
            return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)

        self.check_type_forward(type_check.make_variable(xs_type, 'xs'))
        return self.infer_return(xs_type, lacks_length)

    def check_type_forward(self, xs_type):
        for i in range(xs_type.size().eval()):
            type_check.expect(
                xs_type[i].ndim > 0,
                xs_type[i].shape[1:] == xs_type[0].shape[1:],
                xs_type[i].dtype == xs_type[0].dtype
            )

    def infer_return(self, xs_type, lacks_length):
        n = ShapeElem(xs_type.size())
        ret_shape = list((n,) + xs_type.get().shape)

        if lacks_length:
            ret_shape[1] = None
            return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)
        if self.length is not None:
            ret_shape[1] = self.length
            return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)

        shape_0s = [t.shape[0] for t in xs_type.get_tys()]
        ret_shape[1] = max(shape_0s)
        return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)


class ty_ChainerLocalResponseNormalization():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args

        self.check_type(x_type)
        return self.infer_return(x_type)

    def check_type(self, x_type):
        x = type_check.Variable(x_type, 'x')

        type_check.expect(
            x.dtype.kind == 'f',
            x.ndim >= 2,
        )
        assert len(x_type.shape) >= 2

    def infer_return(self, x_type):
        return TyChainerVariable(dtype=x_type.dtype, shape=x_type.shape)


# ================================= Links ======================================

class ty_ChainerLinear():
    def __call__(self, linear, ty_args, ty_kwargs):
        x_type, = ty_args
        self.n_batch_axes, lacks_n_batch_axes = \
                get_kwarg(ty_kwargs, 'n_batch_axes', default=1)

        if linear.b is not None:
            assert x_type.dtype == linear.b.dtype
        if lacks_n_batch_axes:
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        assert self.n_batch_axes >= 1

        out_shape = self.infer_return_shape(linear, x_type)
        return TyChainerVariable(x_type.dtype, shape=out_shape)

    def infer_return_shape(self, linear, x_type):
        if x_type.ndim == self.n_batch_axes + 1:
            x_shape = x_type.shape
        else:
            x_shape = calculate_reshape(
                    x_type.shape, x_type.shape[:self.n_batch_axes] + (-1,))

        if linear.in_size is not None:
            assert x_shape[-1] == linear.in_size

        return wrap_shape(x_shape[:-1] + (linear.out_size,))


class ty_ChainerConvolution2D():
    def __call__(self, conv, ty_args, ty_kwargs):
        x_type, = ty_args

        assert x_type.dtype.kind == 'f'
        if conv.b is not None:
            assert x_type.dtype == conv.b.dtype
        assert x_type.ndim == 4

        if conv.in_channels is not None:
            assert x_type.shape[1] == conv.in_channels

        return self.infer_return(conv, x_type)

    def infer_return(self, conv, x_type):
        ksize = make_pair(conv.ksize)
        stride = make_pair(conv.stride)
        pad = make_pair(conv.pad)
        dilate = make_pair(conv.dilate)

        shape_2 = get_conv_outsize(
                x_type.shape[2], ksize[0], stride[0], pad[0], d=dilate[0])
        shape_3 = get_conv_outsize(
                x_type.shape[3], ksize[1], stride[1], pad[1], d=dilate[1])
        ret_shape = (x_type.shape[0], conv.out_channels, shape_2, shape_3)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerBatchNormalization():
    def __call__(self, obj, ty_args, ty_kwargs):
        assert False


class ty_ChainerEmbedID():
    def __call__(self, embed, ty_args, ty_kwargs):
        assert isinstance(ty_args[0], TyTensor)
        x_type, = ty_args

        assert x_type.dtype.kind == 'i'
        assert x_type.ndim >= 1
        ret_shape = x_type.shape + (ShapeElem(embed.W.shape[1]),)

        if not is_incomplete_shape(x_type.shape):
            assert all([t < embed.W.shape[0] for t in x_type.shape])
        return TyChainerVariable(embed.W.dtype, shape=ret_shape)


class ty_ChainerNStepBiLSTM():
    def __call__(self, nblstm, ty_args, ty_kwargs):
        hx_type, cx_type, xs_type = ty_args
        if isinstance(xs_type, TyList):
            xs_len = None
        else:
            assert isinstance(xs_type, TyTuple)
            xs_len = xs_type.size()

        if isinstance(hx_type, TyTensor):
            hx_shape = hx_type.shape
            hx_dtype = hx_type.dtype
        else:
            hx_shape = (nblstm.n_layers * 2, xs_len, nblstm.out_size)
            hx_dtype = xs_type.get().dtype

        if isinstance(cx_type, TyTensor):
            cx_shape = cx_type.shape
            cx_dtype = cx_type.dtype
        else:
            cx_shape = (nblstm.n_layers * 2, xs_len, nblstm.out_size)
            cx_dtype = hx_dtype

        hy_type = TyChainerVariable(hx_dtype, shape=hx_shape)
        cy_type = TyChainerVariable(cx_dtype, shape=cx_shape)

        assert hx_shape[0] // 2 == nblstm.n_layers
        assert hx_shape == cx_shape
        N = hx_shape[2]

        if isinstance(xs_type, TyList) or not xs_type.is_fixed_len:
            # TODO
            ys_shape = (xs_type.get().shape[0], 2 * N)
            ys_type = TyList(TyChainerVariable(xs_type.get().dtype, shape=ys_shape))
            return TyTuple([hy_type, cy_type, ys_type])

        xs_dtypes = [t.dtype for t in xs_type.get_tys()]
        xs_shapes = [t.shape for t in xs_type.get_tys()]
        assert all_same(xs_dtypes)

        # TODO(momohatt): nblstm doesn't have attribute in_size
        # assert all([i == nblstm.in_size for _, i in xs_shapes])
        ys_shapes = [(l, 2 * N) for l, _ in xs_shapes]
        ys_type = TyList([TyChainerVariable(xs_dtypes[0], shape=s) for s in ys_shapes])

        return TyTuple([hy_type, cy_type, ys_type])


chainer_attr_ty = {
        'shape'  : ty_Shape,
        'size'   : ty_Size,
        'dtype'  : ty_DType,
        }


chainer_func_ty = {
        chainer.Variable               : ty_ChainerVariable(),
        cuda.to_cpu                    : ty_ChainerIdentical(is_float_only=False),
        F.average_pooling_2d           : ty_ChainerPooling2d(cover_all=False),
        F.broadcast_to                 : ty_ChainerBroadcastTo(),
        F.concat                       : ty_ChainerConcat(),
        F.dropout                      : ty_ChainerIdentical(),
        F.expand_dims                  : ty_ChainerExpandDims(),
        F.hstack                       : ty_ChainerHstack(),
        F.local_response_normalization : ty_ChainerLocalResponseNormalization(),
        F.max_pooling_2d               : ty_ChainerPooling2d(),
        F.pad                          : ty_ChainerPad(),
        F.pad_sequence                 : ty_ChainerPadSequence(),
        F.relu                         : ty_ChainerIdentical(),
        F.reshape                      : ty_ChainerReshape(),
        F.repeat                       : ty_ChainerRepeat(),
        F.separate                     : ty_ChainerSeparate(),
        F.sigmoid                      : ty_ChainerIdentical(),
        F.split_axis                   : ty_ChainerSplitAxis(),
        F.squeeze                      : ty_ChainerSqueeze(),
        F.softmax                      : ty_ChainerIdentical(),
        F.softmax_cross_entropy        : ty_ChainerSoftmaxCrossEntropy(),
        F.stack                        : ty_ChainerStack(),
        F.sum                          : ty_ChainerSum(),
        F.swapaxes                     : ty_ChainerSwapAxes(),
        F.tanh                         : ty_ChainerIdentical(),
        F.vstack                       : ty_ChainerVstack(),
        }


chainer_callable_ty = {
        L.Linear             : ty_ChainerLinear(),
        L.Convolution2D      : ty_ChainerConvolution2D(),
        L.BatchNormalization : ty_ChainerBatchNormalization(),
        L.EmbedID            : ty_ChainerEmbedID(),
        L.NStepBiLSTM        : ty_ChainerNStepBiLSTM(),
        }
