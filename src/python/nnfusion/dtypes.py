# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes
import torch
import numpy
import collections

TypeObject = collections.namedtuple(
    "TypeObject", ["type_str", "c_type", "torch_type", "numpy_type"])

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
