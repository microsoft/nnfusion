# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes
import torch
import numpy
import collections

TypeObject = collections.namedtuple(
    "TypeObject", ["type_str", "c_type", "torch_type", "numpy_type", "n_byte"])

str2type = {
    "half":
        TypeObject._make(["float16", ctypes.c_uint16, torch.float16,
                          numpy.float16, 2]),
    "float16":
        TypeObject._make(["float16", ctypes.c_uint16, torch.float16,
                          numpy.float16, 2]),
    "float":
    TypeObject._make(["float32", ctypes.c_float, torch.float32,
                      numpy.float32, 4]),
    "float32":
    TypeObject._make(["float32", ctypes.c_float, torch.float32,
                      numpy.float32, 4]),
    "double":
    TypeObject._make(
        ["float64", ctypes.c_double, torch.float64, numpy.float64, 8]),
    "float64":
    TypeObject._make(["float64", ctypes.c_double, torch.float64,
                      numpy.float64, 8]),
    "int8":
    TypeObject._make(["int8", ctypes.c_int8, torch.int8, numpy.int8, 1]),
    "int16":
    TypeObject._make(["int16", ctypes.c_int16, torch.int16, numpy.int16, 2]),
    "int32":
    TypeObject._make(["int32", ctypes.c_int32, torch.int32, numpy.int32, 4]),
    "int64":
    TypeObject._make(["int64", ctypes.c_int64, torch.int64, numpy.int64, 8]),
    "uint8":
    TypeObject._make(["uint8", ctypes.c_uint8, torch.uint8, numpy.uint8, 1]),
    "uint16":
    TypeObject._make(["uint16", ctypes.c_uint16, None, numpy.uint16, 2]),
    "uint32":
    TypeObject._make(["uint32", ctypes.c_uint32, None, numpy.uint32, 4]),
    "uint64":
    TypeObject._make(["uint64", ctypes.c_uint64, None, numpy.uint64, 8]),
}
