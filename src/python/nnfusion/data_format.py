# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes

from . import dtypes


class DataFormat(object):
    """ Data format for NNFusion executor.

    Attributes:
        pointer: Ctypes pointer
        point_type: Ctypes pointer type
        shape: A sequence of ints representing tensor shape.
        dtype: A str representing tensor type, reference dtypes.str2dtype
        reference: Keep a reference to origin data structure
    """
    def __init__(self, pointer, pointer_type, shape, dtype, reference=None):
        self._pointer = pointer
        self._pointer_type = pointer_type
        self._shape = tuple(shape)
        self._dtype = dtype
        self._reference = reference

    @property
    def pointer(self):
        return self._pointer

    @property
    def pointer_type(self):
        return self._pointer_type

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def reference(self):
        return self._reference


def cast_pytorch_tensor(pytorch_tensor):
    if not pytorch_tensor.is_contiguous():
        raise Exception(
            "Cannot cast incontiguous tensor, please use tensor.detach().clone().contiguous() before casting."
        )
    tensor_addr = pytorch_tensor.storage().data_ptr()
    shape = pytorch_tensor.shape
    dtype = str(pytorch_tensor.dtype).split(".")[-1]
    pointer_type = ctypes.POINTER(dtypes.str2type[dtype].c_type)
    pointer = ctypes.cast(tensor_addr, pointer_type)
    reference = pytorch_tensor
    return DataFormat(pointer, pointer_type, shape, dtype, reference)


def cast_numpy_array(numpy_array):
    dtype = numpy_array.dtype.name
    pointer_type = ctypes.POINTER(dtypes.str2type[dtype].c_type)
    pointer = numpy_array.ctypes.data_as(pointer_type)
    shape = numpy_array.shape
    reference = numpy_array
    return DataFormat(pointer, pointer_type, shape, dtype, reference)
