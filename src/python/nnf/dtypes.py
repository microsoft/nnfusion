# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes
import torch
import numpy
import collections

TypeObject = collections.namedtuple(
    "TypeObject", ["type_str", "c_type", "torch_type", "numpy_type"])

c_float = ctypes.c_float
c_float_p = ctypes.POINTER(ctypes.c_float)
c_float_p_p = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
c_int = ctypes.c_int
c_int_p = ctypes.POINTER(ctypes.c_int)
c_int_p_p = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
c_int64 = ctypes.c_int64
c_int64_p = ctypes.POINTER(ctypes.c_int64)

str2type = {
    "float":
    TypeObject._make(["float32", ctypes.c_float, torch.float32,
                      numpy.float32]),
    "float32":
    TypeObject._make(["float32", ctypes.c_float, torch.float32,
                      numpy.float32]),
    "double":
    TypeObject._make(
        ["float64", ctypes.c_double, torch.float64, numpy.float64]),
    "float64":
    TypeObject._make(["float64", ctypes.c_float, torch.float32,
                      numpy.float32]),
    "int8":
    TypeObject._make(["int8", ctypes.c_int8, torch.int8, numpy.int8]),
    "int16":
    TypeObject._make(["int16", ctypes.c_int16, torch.int16, numpy.int16]),
    "int32":
    TypeObject._make(["int32", ctypes.c_int32, torch.int32, numpy.int32]),
    "int64":
    TypeObject._make(["int64", ctypes.c_int64, torch.int64, numpy.int64]),
    "uint8":
    TypeObject._make(["uint8", ctypes.c_uint8, torch.uint8, numpy.uint8]),
    "uint16":
    TypeObject._make(["uint8", ctypes.c_uint16, None, numpy.uint16]),
    "uint32":
    TypeObject._make(["uint8", ctypes.c_uint32, None, numpy.uint32]),
    "uint64":
    TypeObject._make(["uint8", ctypes.c_uint64, None, numpy.uint64]),
}

dtype2ctype = {
    torch.float32: c_float_p,
    torch.int32: c_int_p,
    torch.int64: c_int64_p
}


def tensor_ptr(tensor):
    if tensor.dtype not in dtype2ctype:
        raise Exception("Dtype is not suppported: {}".format(tensor.dtype))
    tensor_addr = tensor.storage().data_ptr()
    return ctypes.cast(tensor_addr, dtype2ctype[tensor.dtype])


def get_data_addr(tensors):
    return tuple(tensor_ptr(t) for t in tensors)


def deduce_signatrue(tensors):
    return tuple(dtype2ctype[t.dtype] for t in tensors)
