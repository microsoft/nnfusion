import numpy as np

import onnx
from onnx import TensorProto, ValueInfoProto
from ast_analyzer.shape_inference.shape_elem import unwrap_shape
from ast_analyzer.shape_inference.types import tyobj_to_dtype
from ast_analyzer.shape_inference.types import *

__all__ = [
    'type_np_to_onnx',
    'type_onnx_to_np',
    'type_to_value_info',
    'scaler_to_value_info',
    'value_info_with_name',
    'np_inst_to_value_info',
    'have_real_func'
]

np2onnx = {
    np.single: TensorProto.FLOAT,
    np.float32: TensorProto.FLOAT,
    np.uint8: TensorProto.UINT8,
    np.int8: TensorProto.INT8,
    np.uint16: TensorProto.UINT16,
    np.int16: TensorProto.INT16,
    np.int32: TensorProto.INT32,
    np.int64: TensorProto.INT64,
    # skip TensorProto.STRING
    np.bool_: TensorProto.BOOL,
    np.float16: TensorProto.FLOAT16,
    np.double: TensorProto.FLOAT, # HACK DOUBLE
    np.float64: TensorProto.FLOAT, # HACK DOUBLE
    np.uint32: TensorProto.UINT32,
    np.uint64: TensorProto.UINT64,
    np.complex64: TensorProto.COMPLEX64,
    np.complex128: TensorProto.COMPLEX128,
}


def type_np_to_onnx(np_dtype):
    # cannot use np2onnx[np_dtype], so hack it
    for np_ty in np2onnx.keys():
        if np_ty == np_dtype:
            return np2onnx[np_ty]
    raise NotImplementedError(
        np_dtype + " is not supported in type_np_to_onnx")


def type_onnx_to_np(onnx_dtype):
    if 'float16' in onnx_dtype:
        return np.float16
    elif 'float' in onnx_dtype:
        return np.float32
    elif 'double' in onnx_dtype:
        return np.float32 # HACK DOUBLE
    elif 'int8' in onnx_dtype:
        return np.int8
    elif 'int16' in onnx_dtype:
        return np.int16
    elif 'int32' in onnx_dtype:
        return np.int32
    elif 'int64' in onnx_dtype:
        return np.int64
    elif 'uint8' in onnx_dtype:
        return np.uint8
    elif 'uint16' in onnx_dtype:
        return np.uint16
    elif 'bool' in onnx_dtype:
        return np.bool_
    else:
        raise NotImplementedError(
            onnx_dtype + " is not supported in type_onnx_to_np")
    return np.float32


# str, np.array, ExportEngine -> ValueInfoProto
def np_inst_to_value_info(name, np_inst, visitor):
    value_info = onnx.helper.make_tensor_value_info(
        name, type_np_to_onnx(np_inst.dtype), np_inst.shape)
    visitor.register_value_info(value_info)
    return value_info


# str, Type, ExportEngine -> ValueInfoProto
def type_to_value_info(name, ty, visitor):
    if isinstance(ty, TyTensor):
        value_info = onnx.helper.make_tensor_value_info(
            name, type_np_to_onnx(ty.dtype), unwrap_shape(ty.shape, True))
    elif isinstance(ty, TyNum):
        value_info = onnx.helper.make_tensor_value_info(
            name, type_np_to_onnx(tyobj_to_dtype(ty)), [])
    elif isinstance(ty, TyTuple):
        assert(ty.is_fixed_len)
        value_info = onnx.helper.make_tensor_value_info(
            name, TensorProto.INT64, [ty.size()])
    else:
        raise NotImplementedError(
            str(ty) + " is not supported in type_to_value_info")

    visitor.register_value_info(value_info)
    return value_info


# str, onnx_type, ExportEngine -> ValueInfoProto
def scaler_to_value_info(name, onnx_ty, visitor):
    value_info = onnx.helper.make_tensor_value_info(name, onnx_ty, [])
    visitor.register_value_info(value_info)
    return value_info


# ValueInfoProto, str, ExportEngine -> ValueInfoProto
def value_info_with_name(value_info, name, visitor):
    new_info = ValueInfoProto()
    new_info.CopyFrom(value_info)
    new_info.name = name
    visitor.register_value_info(new_info)
    return new_info


def have_real_func(model):
    if len(model.graph.input) == 0:
        return False
    found = False
    op_ignore = ['Gather']
    for node in model.graph.node:
        print(node.op_type)
        if node.op_type not in op_ignore:
            found = True
            break
    return found