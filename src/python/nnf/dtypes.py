# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes
import torch
c_float = ctypes.c_float
c_float_p = ctypes.POINTER(ctypes.c_float)
c_float_p_p = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
c_int = ctypes.c_int
c_int_p = ctypes.POINTER(ctypes.c_int)
c_int_p_p = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
c_int64 = ctypes.c_int64
c_int64_p = ctypes.POINTER(ctypes.c_int64)

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
