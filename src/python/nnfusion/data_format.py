# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes
import torch

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
        if len(shape) > 0:
            self._shape = tuple(shape)
        else:
            self._shape = (1, )
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

class HLSLTensor(object):
    antares_lib = None

    @classmethod
    def init_antares_lib(cls, antares_dll_path):
        if cls.antares_lib is None:    
            # cls.antares_lib = ctypes.cdll.LoadLibrary(r"D:\project\nnfusion_rt_pow\nnfusion_rt\dxcompute_codegen\Direct3DWinNN_seperate_dll\x64\Release\antares.dll")
            cls.antares_lib = ctypes.cdll.LoadLibrary(antares_dll_path)
            # alloc
            cls.antares_lib.dxMemAlloc.argtypes = [ctypes.c_uint64]
            cls.antares_lib.dxMemAlloc.restype = ctypes.c_void_p
            # free
            cls.antares_lib.dxMemFree.argtypes = [ctypes.c_void_p]
            cls.antares_lib.dxMemFree.restype = ctypes.c_int32
            # H2D
            cls.antares_lib.dxMemcpyHtoDAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
            cls.antares_lib.dxMemcpyHtoDAsync.restype = ctypes.c_int32
            # D2H
            cls.antares_lib.dxMemcpyDtoHAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
            cls.antares_lib.dxMemcpyDtoHAsync.restype = ctypes.c_int32
            # Sync
            cls.antares_lib.dxStreamSynchronize.argtypes = [ctypes.c_void_p]
            cls.antares_lib.dxStreamSynchronize.restype = ctypes.c_int32
        return

    @classmethod
    def build_from_torch(cls, pytorch_tensor):
        pytorch_tensor = pytorch_tensor.contiguous()
        shape = pytorch_tensor.shape
        pt_type = str(pytorch_tensor.dtype).split(".")[-1]
        dtype = dtypes.str2type[pt_type].type_str
        hlsl_tensor = HLSLTensor(shape, dtype)
        if hlsl_tensor.size > 0:
            cls.antares_lib.dxMemcpyHtoDAsync(hlsl_tensor.pointer, ctypes.cast(pytorch_tensor.data_ptr(), ctypes.c_void_p), hlsl_tensor.size, None)
            cls.antares_lib.dxStreamSynchronize(None)
        return hlsl_tensor

    def __init__(self, shape, dtype) -> None:
        if self.antares_lib is None:
            raise Exception("Please init antares lib firstly(e.g. creating a executor instance antomatically init antares lib")
        self.shape = shape
        self.dtype = dtypes.str2type[dtype].type_str
        num_element = 1
        for dim in shape:
            num_element *= dim
        element_size = dtypes.str2type[dtype].n_byte
        self.size = num_element * element_size
        if self.size > 0:
            self.pointer = self.antares_lib.dxMemAlloc(self.size)
        else:
            self.pointer = None
        

    def __del__(self):
        if hasattr(self, "pointer") and self.pointer:
            self.antares_lib.dxMemFree(self.pointer)
            # access violation reading 0x0000000000000020
            # possibliy caused by dxFinialize reset stream when release nnf_rt
            # self.antares_lib.dxStreamSynchronize(None)
            self.pointer == ctypes.c_void_p(None)
    
    def __str__(self):
        return self.to_pytorch_tensor().__str__()

    def to_pytorch_tensor(self):
        res = torch.empty(self.shape, dtype=dtypes.str2type[self.dtype].torch_type)        
        self.antares_lib.dxMemcpyDtoHAsync(ctypes.cast(res.data_ptr(), ctypes.c_void_p), self.pointer, self.size, None)       
        self.antares_lib.dxStreamSynchronize(None)
        return res
        
def cast_hlsl_tensor(hlsl_tensor):
    pointer_type = ctypes.POINTER(dtypes.str2type[hlsl_tensor.dtype].c_type)
    pointer = ctypes.cast(hlsl_tensor.pointer, pointer_type)
    shape = hlsl_tensor.shape
    dtype = hlsl_tensor.dtype
    reference = hlsl_tensor
    return DataFormat(pointer, pointer_type, shape, dtype, reference)

def cast_pytorch_tensor(pytorch_tensor):
    if not pytorch_tensor.is_contiguous():
        raise Exception(
            "Cannot cast incontiguous tensor, please use tensor.detach().clone().contiguous() before casting."
        )
    tensor_addr = pytorch_tensor.data_ptr()
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
