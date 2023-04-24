import numpy as np
import math

from   chainer.utils import type_check

from   ast_analyzer.shape_inference.types      import *
from   ast_analyzer.shape_inference.shape_elem import *


def size_of_shape(shape):
    size = 1
    for i in shape:
        size *= i
    return size


def make_tuple(x, n):
    if isinstance(x, tuple):
        return x
    return (x,) * n

def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    return x


def get_kwarg(ty_kwargs, key, default):
    if key in ty_kwargs.keys():
        # when unable to get the correct value, returns None
        return extract_value_from_ty(ty_kwargs[key]), lacks_value(ty_kwargs[key])
    return default, False


def extract_kwarg(ty_kwargs, key, default):
    if key in ty_kwargs.keys():
        return extract_kwarg(ty_kwargs[key])
    return default


def make_multiple_tc_variable(ty_args, names):
    assert len(ty_args) == len(names)
    return [type_check.Variable(t, n) for t, n in zip(ty_args, names)]


def calculate_reshape(in_shape, out_shape):
    # in_shape must be wrapped
    if is_incomplete_shape(in_shape):
        if any([i == -1 for i in out_shape]):
            return wrap_shape([i if i != -1 else None for i in out_shape])
        return out_shape
    in_shape = unwrap_shape(in_shape)
    fill = abs(size_of_shape(in_shape) // size_of_shape(out_shape))
    ret_shape = tuple([i if i != -1 else fill for i in out_shape])
    assert size_of_shape(in_shape) == size_of_shape(ret_shape)
    return wrap_shape(ret_shape)


def remove_dims(shape, dims_to_remove):
    # dims_to_remove can have negative indices
    dims_to_remove = [d % len(shape) for d in dims_to_remove]
    return tuple([shape[i] for i in range(len(shape)) if i not in dims_to_remove])
