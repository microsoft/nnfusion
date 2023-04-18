import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnx import helper
from ast_analyzer.shape_inference.shape_elem import unwrap_shape
from ast_analyzer.shape_inference.types import *
import astunparse

import gast as ast
import numpy as np
import math
from functools import reduce

from .utils import type_to_value_info, type_np_to_onnx, np_inst_to_value_info
from .node import OnnxNodes

__all__ = ['pytorch_func_export']


def export_tensor_arg(arg_node, visitor, name) -> OnnxNodes:
    visitor.set_output_name(arg_node, name)
    onnx_node = visitor.visit(arg_node)
    assert(len(onnx_node.out_node) == 1)
    return onnx_node


def export_obj_tensor_arg(call_node, visitor, name) -> OnnxNodes:
    assert isinstance(call_node.func, ast.Attribute)
    return export_tensor_arg(call_node.func.value, visitor, name)


def export_tuple_arg(arg_node, visitor, name, n_out) -> OnnxNodes:
    visitor.set_output_name(arg_node, name)
    onnx_node = visitor.visit(arg_node)
    assert(len(onnx_node.out_node) == n_out)
    return onnx_node


def export_layer(attr_node, visitor, name, expect_out, func_inst) -> OnnxNodes:
    visitor.set_output_name(attr_node, name)
    onnx_node = visitor.export_layer(attr_node, func_inst)
    assert(len(onnx_node.out_node) in expect_out)
    return onnx_node


def get_keywords(node, whitelist):
    ret = {}
    for key in node.keywords:
        assert(key.arg in whitelist)
        ret[key.arg] = key
    return ret


def export_tensor_args(node, visitor, name, is_tensor, n_args): # -> List[OnnxNodes]
    ret = []
    if is_tensor:
        ret.append(export_obj_tensor_arg(node, visitor, name + '0'))
        for i in range(1, n_args):
            ret.append(export_tensor_arg(node.args[i-1], visitor, name + str(i)))
    else:
        for i in range(0, n_args):
            ret.append(export_tensor_arg(node.args[i], visitor, name + str(i)))
    return tuple(ret)

class Exporter():
    def __init__(self, is_tensor = False):
        self.is_tensor = is_tensor

class ExportTorchTensorOfShape(Exporter):
    def __init__(self, value, is_tensor = False):
        assert(is_tensor == False)
        super().__init__(is_tensor)
        self.value = value

    def __call__(self, node, func, visitor) -> OnnxNodes:
        # type inference already considered dtype
        _ = get_keywords(node, ('dtype', 'device'))
        ty_call = visitor.get_type_of_node(node)
        if not ty_call.is_fixed_shape():
            raise NotImplementedError
        
        if self.value is None: # full_like
            ty_value = visitor.get_type_of_node(node.args[1])
            if not isinstance(ty_value, TyNum):
                raise NotImplementedError
            ty_value.coerce_value()
            if ty_value.value is None:
                raise NotImplementedError
            value =  ty_value.value
        else:
            value = self.value

        tensor_value = onnx.helper.make_tensor(
            visitor.gen_name(), type_np_to_onnx(ty_call.dtype), [1], [value])
        shape_name = visitor.gen_name()
        shape_arr = np.array(unwrap_shape(ty_call.shape, True), dtype=np.int64)
        shape_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            value=onnx.helper.make_tensor(
                name=visitor.gen_name(),
                data_type=onnx.TensorProto.INT64,
                dims=[len(unwrap_shape(ty_call.shape, True))],
                vals=shape_arr,
            )
        )
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node('ConstantOfShape', inputs=[
                                     shape_name], outputs=[name], value=tensor_value)
        return OnnxNodes([shape_node, onnx_node], [name], {
            name: type_to_value_info(name, ty_call, visitor),
            shape_name: np_inst_to_value_info(shape_name, shape_arr, visitor)
        })


class ExportTorchConstLike(Exporter):
    def __init__(self, value, is_tensor):
        super().__init__(is_tensor)
        self.value = value
    
    def __call__(self, node, func, visitor) -> OnnxNodes:
        ty_call = visitor.get_type_of_node(node)

        if not ty_call.is_fixed_shape():
            raise NotImplementedError
        if self.value is None: # full_like
            ty_value = visitor.get_type_of_node(node.args[1])
            if not isinstance(ty_value, TyNum):
                raise NotImplementedError
            ty_value.coerce_value()
            if ty_value.value is None:
                raise NotImplementedError
            value =  ty_value.value
        else:
            value = self.value
        tensor_value = onnx.helper.make_tensor(
            visitor.gen_name(), type_np_to_onnx(ty_call.dtype), [1], [value])
        shape_name = visitor.gen_name()
        shape_arr = np.array(unwrap_shape(ty_call.shape, True), dtype=np.int64)
        shape_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            value=onnx.helper.make_tensor(
                name=visitor.gen_name(),
                data_type=onnx.TensorProto.INT64,
                dims=[len(unwrap_shape(ty_call.shape, True))],
                vals=shape_arr,
            )
        )
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node('ConstantOfShape', inputs=[
                                     shape_name], outputs=[name], value=tensor_value)
        return OnnxNodes([shape_node, onnx_node], [name], {
            name: type_to_value_info(name, ty_call, visitor),
            shape_name: np_inst_to_value_info(shape_name, shape_arr, visitor)
        })


class ExportTorchMatmul(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        transA = 0
        transB = 0
        if isinstance(node.args[0], ast.Call) and node.args[0]._func_inst == torch.transpose and len(visitor.get_type_of_node(node.args[0]).shape) == 2:
            transA = 1
        if isinstance(node.args[1], ast.Call) and node.args[1]._func_inst == torch.transpose and len(visitor.get_type_of_node(node.args[0]).shape) == 2:
            transB = 1
        assert(transA + transB < 2)
        if transA or transB:
            # TODO: solve this quick hack
            arg_ty = []
            if self.is_tensor:
                raise NotImplementedError()
            if transA:
                arg0 = export_tensor_arg(node.args[0].args[0], visitor, visitor.gen_name())
                ty_arg = visitor.get_type_of_node(node.args[0].args[0])
                arg_ty.append(ty_arg)
            else:
                arg0 = export_tensor_arg(node.args[0], visitor, visitor.gen_name())
                arg_ty.append(visitor.get_type_of_node(node.args[0]))
            if transB:
                arg1 = export_tensor_arg(node.args[1].args[0], visitor, visitor.gen_name())
                ty_arg = visitor.get_type_of_node(node.args[1].args[0])
                arg_ty.append(ty_arg)
            else:
                arg1 = export_tensor_arg(node.args[1], visitor, visitor.gen_name())
                arg_ty.append(visitor.get_type_of_node(node.args[1]))
            assert(len(arg0.out_node) == 1)
            assert(len(arg1.out_node) == 1)
            assert(isinstance(ty_arg, TyTensor))

            old_shape = []
            for ty in ty_arg.shape:
                assert(ty.has_value())
                old_shape.append(ty.value)
            new_shape = (reduce(lambda x, y: x*y, old_shape[:-1], 1), old_shape[-1])

            shape_name = visitor.gen_name()
            shape_arr = np.array(new_shape, dtype=np.int64)
            shape_node = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[shape_name],
                value=onnx.helper.make_tensor(
                    name=visitor.gen_name(),
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(new_shape)],
                    vals=shape_arr,
                )
            )

            # print("[op] Reshape between", old_shape, new_shape)
            name_tmp = visitor.gen_name()
            if transA:
                reshape_node = helper.make_node('Reshape', inputs=[arg0.out_node[0], shape_name], outputs=[name_tmp])
            else:
                reshape_node = helper.make_node('Reshape', inputs=[arg1.out_node[0], shape_name], outputs=[name_tmp])

            name = visitor.get_or_gen_name(node)
            ty_call = visitor.get_type_of_node(node)

            onnx_node = helper.make_node(
                'Gemm',
                inputs=[name_tmp, arg1.out_node[0]] if transA else [arg0.out_node[0], name_tmp],
                outputs=[name],
                transA=transA,
                transB=transB,
                alpha=1.0,
                beta=0.0
            )

            ty_call = visitor.get_type_of_node(node)
            # print("[OP] Gemm: {} {} -> {} transA = {} transB = {}".format(arg_ty[0].shape, arg_ty[1].shape, ty_call.shape, transA, transB))
            final_shape = []
            for ty in ty_call.shape:
                assert(ty.has_value())
                final_shape.append(ty.value)

            shape2_name = visitor.gen_name()
            shape2_arr = np.array(final_shape, dtype=np.int64)
            shape2_node = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[shape2_name],
                value=onnx.helper.make_tensor(
                    name=visitor.gen_name(),
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(final_shape)],
                    vals=shape2_arr,
                )
            )

            name_final = visitor.gen_name()
            final_node = helper.make_node('Reshape', inputs=[name, shape2_name], outputs=[name_final])

            value_info = type_to_value_info(name_final, ty_call, visitor)

            new_node = arg0 + arg1
            new_node.def_nodes.append(shape_node)
            new_node.def_nodes.append(reshape_node)
            new_node.def_nodes.append(onnx_node)
            new_node.def_nodes.append(shape2_node)
            new_node.set_output(final_node, name_final, value_info)
            return new_node
        else:
            arg0, arg1 = export_tensor_args(node, visitor, 'matmul_arg', self.is_tensor, 2)
            ret_nodes = arg0 + arg1

            name = visitor.get_or_gen_name(node)
            onnx_node = helper.make_node(
                'MatMul', [ret_nodes.out_node[0], ret_nodes.out_node[1]], [name])
            ty_call = visitor.get_type_of_node(node)
            # print("[OP] Gemm: {} {} -> {} transA = {} transB = {}".format(visitor.get_type_of_node(node.args[0]).shape, visitor.get_type_of_node(node.args[1]).shape, ty_call.shape, 0, 0))
            value_info = type_to_value_info(name, ty_call, visitor)
            ret_nodes.set_output(onnx_node, name, value_info)
            return ret_nodes


class ExportTorchElementWise(Exporter):
    def __init__(self, op, is_tensor):
        super().__init__(is_tensor)
        self.op = op

    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        arg0 = export_tensor_args(node, visitor, self.op + '_arg0', self.is_tensor, 1)[0]
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            self.op,
            inputs=[arg0.out_node[0]],
            outputs=[name]
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchCompare(Exporter):
    def __init__(self, op, is_tensor):
        super().__init__(is_tensor)
        self.op = op
    
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        arg0, arg1 = export_tensor_args(node, visitor, self.op + '_arg0', self.is_tensor, 2)
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            self.op,
            inputs=[arg0.out_node[0], arg1.out_node[0]],
            outputs=[name]
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0 += arg1
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchWhere(Exporter):
    def __call__(self, node, func, visitor):
        assert(len(node.keywords) == 0)
        cond, true_node, false_node = export_tensor_args(node, visitor, 'where_arg0', self.is_tensor, 3)
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Where',
            inputs=[cond.out_node[0], true_node.out_node[0], false_node.out_node[0]],
            outputs=[name]
        )

        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        cond += true_node
        cond += false_node
        cond.set_output(onnx_node, name, value_info)
        return cond


class ExportTorchShape(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        arg0 = export_tensor_args(node, visitor, "shape_arg", self.is_tensor, 1)[0]
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Shape',
            inputs=[arg0.out_node[0]],
            outputs=[name]
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchSum(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        keys = get_keywords(node, ('dim', 'keepdim'))
        if len(node.args) + self.is_tensor > 1:
            raise NotImplementedError
        arg0 = export_tensor_args(node, visitor, 'sum_arg0', self.is_tensor, 1)[0]
        name = visitor.get_or_gen_name(node)
        keepdims = False
        if "keepdim" in keys:
            keepdims = visitor.get_type_of_node(keys['keepdim'].value).value
            assert keepdims in (True, False)
        if 'dim' in keys:
            dims = visitor.get_type_of_node(keys['dim'].value)
            assert(isinstance(dims, TyTuple))
            dims = list(eval(astunparse.unparse(keys['dim'].value), {}))
            onnx_node = helper.make_node(
                'ReduceSum',
                inputs=[arg0.out_node[0]],
                outputs=[name],
                axes=dims,
                keepdims=keepdims
            )
        else:
            onnx_node = helper.make_node(
                'ReduceSum',
                inputs=[arg0.out_node[0]],
                outputs=[name],
                keepdims=keepdims
            )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchMax(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        keys = get_keywords(node, ('dim', 'keepdim'))
        if len(node.args) + self.is_tensor > 1:
            raise NotImplementedError
        arg0 = export_tensor_args(node, visitor, 'max_arg0', self.is_tensor, 1)[0]
        name = visitor.get_or_gen_name(node)
        keepdims = False
        if "keepdim" in keys:
            keepdims = visitor.get_type_of_node(keys['keepdim'].value).value
            assert keepdims in (True, False)
        if 'dim' in keys:
            dims = visitor.get_type_of_node(keys['dim'].value)
            assert(isinstance(dims, TyTuple))
            dims = list(eval(astunparse.unparse(keys['dim'].value), {}))
            onnx_node = helper.make_node(
                'ReduceMax',
                inputs=[arg0.out_node[0]],
                outputs=[name],
                axes=dims,
                keepdims=keepdims
            )
        else:
            onnx_node = helper.make_node(
                'ReduceMax',
                inputs=[arg0.out_node[0]],
                outputs=[name],
                keepdims=keepdims
            )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchMean(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        keys = get_keywords(node, ('dim', 'keepdim'))
        if len(node.args) + self.is_tensor > 1:
            raise NotImplementedError
        arg0 = export_tensor_args(node, visitor, 'mean_arg0', self.is_tensor, 1)[0]
        name = visitor.get_or_gen_name(node)
        keepdims = False
        if "keepdim" in keys:
            keepdims = visitor.get_type_of_node(keys['keepdim'].value).value
            assert keepdims in (True, False)
        if 'dim' in keys:
            dims = visitor.get_type_of_node(keys['dim'].value)
            assert(isinstance(dims, TyTuple))
            dims = list(eval(astunparse.unparse(keys['dim'].value), {}))
            onnx_node = helper.make_node(
                'ReduceMean',
                inputs=[arg0.out_node[0]],
                outputs=[name],
                axes=dims,
                keepdims=keepdims
            )
        else:
            onnx_node = helper.make_node(
                'ReduceMean',
                inputs=[arg0.out_node[0]],
                outputs=[name],
                keepdims=keepdims
            )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchTranspose(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        assert(len(node.args) + self.is_tensor == 3)
        arg0 = export_tensor_args(node, visitor, 'sum_arg0', self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)
        if self.is_tensor:
            tensor_ty = visitor.get_type_of_node(node.func.value)
        else:
            tensor_ty = visitor.get_type_of_node(node.args[0])
        assert(isinstance(tensor_ty, TyTensor))
        arg1_ty = visitor.get_type_of_node(node.args[1 - self.is_tensor])
        arg2_ty = visitor.get_type_of_node(node.args[2 - self.is_tensor])
        assert(arg1_ty.is_int())
        assert(arg2_ty.is_int())
        shape = list(range(tensor_ty.ndim))
        dim0 = arg1_ty.value
        dim1 = arg2_ty.value
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Transpose',
            inputs=[arg0.out_node[0]],
            outputs=[name],
            perm=shape
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchSplit(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 1)
        assert(node.keywords[0].arg == 'dim')
        assert(len(node.args) + self.is_tensor == 2)
        dim_ty = visitor.get_type_of_node(node.keywords[0].value)
        assert(dim_ty.is_int())
        dim = dim_ty.value
        arg0 = export_tensor_args(node, visitor, 'split_arg0', self.is_tensor, 1)[0]
        arg1_ty = visitor.get_type_of_node(node.args[1 - self.is_tensor])
        name = visitor.get_or_gen_name(node)
        if isinstance(arg1_ty, TyNum):
            assert(arg1_ty.is_int())
            raise NotImplementedError
        else:
            assert(isinstance(arg1_ty, TyTuple))
            axis = extract_value_from_ty(dim_ty)
            names = [visitor.gen_name(name) for i in range(arg1_ty.size())]
            onnx_node = helper.make_node(
                'Split',
                inputs=[arg0.out_node[0]],
                outputs=names,
                axis=axis,
                split=extract_value_from_ty(arg1_ty)
            )
            ty_call = visitor.get_type_of_node(node)
            value_info = {}
            for i, ty in enumerate(ty_call.get_tys()):
                value_info[names[i]] = type_to_value_info(
                    names[i], ty, visitor)
            arg0.set_outputs([onnx_node], names, value_info)
            return arg0


class ExportTorchCast(Exporter):
    def __init__(self, onnx_type, is_tensor = False):
        super().__init__(is_tensor)
        self.onnx_type = onnx_type
    
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        assert(len(node.args) + self.is_tensor == 1)
        arg0 = export_tensor_args(node, visitor, "expand_arg", self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)

        ty_call = visitor.get_type_of_node(node)
        if not (isinstance(ty_call, TyNum) or (isinstance(ty_call, TyTensor) and ty_call.is_fixed_shape())):
            raise NotImplementedError
        
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Cast', inputs=[arg0.out_node[0]], outputs=[name],
            to=int(self.onnx_type)
        )

        arg0.set_output(onnx_node, name, type_to_value_info(
            name, ty_call, visitor))
        return arg0


class ExportTorchExpandAs(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        assert(len(node.args) + self.is_tensor == 2)
        arg0 = export_tensor_args(node, visitor, "expand_arg", self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)

        ty_call = visitor.get_type_of_node(node)
        if not ty_call.is_fixed_shape():
            raise NotImplementedError

        shape_name = visitor.gen_name()
        shape_arr = np.array(unwrap_shape(ty_call.shape, True), dtype=np.int64)
        shape_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            value=onnx.helper.make_tensor(
                name=visitor.gen_name(),
                data_type=onnx.TensorProto.INT64,
                dims=[len(unwrap_shape(ty_call.shape, True))],
                vals=shape_arr,
            )
        )
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Expand', inputs=[arg0.out_node[0], shape_name], outputs=[name]
        )

        return OnnxNodes(arg0.def_nodes + [shape_node, onnx_node], [name], {
            name: type_to_value_info(name, ty_call, visitor),
            shape_name: np_inst_to_value_info(shape_name, shape_arr, visitor)
        }.update(arg0.def_value_infos))


class ExportTorchCat(Exporter):
    def __init__(self, is_tensor):
        assert(is_tensor == False)
        super().__init__(is_tensor)

    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.args) == 1)
        ty_arg0 = visitor.get_type_of_node(node.args[0])
        assert(isinstance(ty_arg0, TyTuple))
        n_out = ty_arg0.size()
        if n_out is None:
            raise NotImplementedError()
        arg0 = export_tuple_arg(node.args[0], visitor, 'cat_tuple', n_out)
        if len(node.keywords) > 0:
            assert(len(node.keywords) == 1)
            assert(node.keywords[0].arg == 'dim')
            dim_ty = visitor.get_type_of_node(node.keywords[0].value)
            assert(dim_ty.is_int())
            dim = dim_ty.value
            if not isinstance(dim, int):
                raise NotImplementedError
        else:
            dim = 0

        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Concat',
            inputs=arg0.out_node,
            outputs=[name],
            axis=dim
        )

        ty_call = visitor.get_type_of_node(node)
        arg0.set_output(onnx_node, name, type_to_value_info(
            name, ty_call, visitor))
        return arg0


class ExportTorchView(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        ty_args = [visitor.get_type_of_node(nd) for nd in node.args[1-int(self.is_tensor):]]
        new_shape = []
        for ty in ty_args:
            assert(ty.is_int())
            if ty.value is None:
                raise NotImplementedError
            new_shape.append(ty.value)
        arg0 = export_tensor_args(node, visitor, "expand_arg", self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)
        shape_name = visitor.gen_name()
        shape_arr = np.array(new_shape, dtype=np.int64)
        shape_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            value=onnx.helper.make_tensor(
                name=visitor.gen_name(),
                data_type=onnx.TensorProto.INT64,
                dims=[len(new_shape)],
                vals=shape_arr,
            )
        )
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node('Reshape', inputs=[arg0.out_node[0], shape_name], outputs=[name])
        ty_call = visitor.get_type_of_node(node)
        return OnnxNodes(arg0.def_nodes + [shape_node, onnx_node], [name], {
            **arg0.def_value_infos,
            name: type_to_value_info(name, ty_call, visitor),
            shape_name: np_inst_to_value_info(shape_name, shape_arr, visitor)
        })


class ExportTorchReshape(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        new_shape = []
        ty_args = visitor.get_type_of_node(node.args[1 - self.is_tensor])
        for ty in ty_args:
            assert(ty.is_int())
            if ty.value is None:
                raise NotImplementedError
            new_shape.append(ty.value)
        arg0 = export_tensor_args(node, visitor, "expand_arg", self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)
        # print("[op] torch reshape", astunparse.unparse(node))
        shape_name = visitor.gen_name()
        shape_arr = np.array(new_shape, dtype=np.int64)
        shape_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            value=onnx.helper.make_tensor(
                name=visitor.gen_name(),
                data_type=onnx.TensorProto.INT64,
                dims=[len(new_shape)],
                vals=shape_arr,
            )
        )
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node('Reshape', inputs=[arg0.out_node[0], shape_name], outputs=[name])
        ty_call = visitor.get_type_of_node(node)
        return OnnxNodes(arg0.def_nodes + [shape_node, onnx_node], [name], {
            name: type_to_value_info(name, ty_call, visitor),
            shape_name: np_inst_to_value_info(shape_name, shape_arr, visitor)
        })


class ExportTorchPermute(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        assert(len(node.keywords) == 0)
        assert(len(node.args) + self.is_tensor == 2)
        arg0 = export_tensor_args(node, visitor, 'permute_arg0', self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)
        perm = []
        ty_args = visitor.get_type_of_node(node.args[1 - self.is_tensor])
        for ty in ty_args:
            assert(ty.is_int())
            if ty.value is None:
                raise NotImplementedError
            perm.append(ty.value)
        
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Transpose',
            inputs=[arg0.out_node[0]],
            outputs=[name],
            perm=perm
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchSoftmax(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        arg0 = export_tensor_args(node, visitor, 'softmax_arg0', self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)
        assert(len(node.keywords) == 1 and node.keywords[0].arg == 'dim')
        ty_dim = visitor.get_type_of_node(node.keywords[0].value)
        assert(ty_dim.is_int())
        if ty_dim.value is None:
            raise NotImplementedError
        dim = ty_dim.value

        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Softmax',
            inputs=[arg0.out_node[0]],
            outputs=[name],
            axis=dim
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchLogSoftmax(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        arg0 = export_tensor_args(node, visitor, 'log_softmax_arg0', self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)
        assert(len(node.keywords) == 1 and node.keywords[0].arg == 'dim')
        ty_dim = visitor.get_type_of_node(node.keywords[0].value)
        assert(ty_dim.is_int())
        if ty_dim.value is None:
            raise NotImplementedError
        dim = ty_dim.value

        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'LogSoftmax',
            inputs=[arg0.out_node[0]],
            outputs=[name],
            axis=dim
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchFlatten(Exporter):
    def __call__(self, node, func, visitor) -> OnnxNodes:
        arg0 = export_tensor_args(node, visitor, 'flatten_arg0', self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)
        assert(len(node.keywords) == 1 and node.keywords[0].arg == 'start_dim')
        ty_dim = visitor.get_type_of_node(node.keywords[0].value)
        assert(ty_dim.is_int())
        if ty_dim.value is None:
            raise NotImplementedError
        dim = ty_dim.value

        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Flatten',
            inputs=[arg0.out_node[0]],
            outputs=[name],
            axis=dim
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchIdentity(Exporter):
     def __call__(self, node, func, visitor):
        arg0 = export_tensor_args(node, visitor, 'flatten_arg0', self.is_tensor, 1)[0]
        return arg0


class ExportTorchArgmax(Exporter):
    def __call__(self, node, func, visitor):
        arg0 = export_tensor_args(node, visitor, 'argmax_arg0', self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)
        assert(len(node.args) + self.is_tensor == 2)
        assert(len(node.keywords) == 0 or (len(node.keywords) == 1 and node.keywords[0].arg == 'keepdim'))
        if len(node.keywords) == 1:
            keepdim = node.keywords[0].value.value
        else:
            keepdim = False
        ty_axis = visitor.get_type_of_node(node.args[1 - self.is_tensor])
        assert isinstance(ty_axis, TyNum)
        ty_axis.coerce_value()
        assert ty_axis.is_int() and ty_axis.value is not None
        axis = ty_axis.value
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'ArgMax',
            inputs=[arg0.out_node[0]],
            outputs=[name],
            axis=axis,
            keepdims=keepdim
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg0.set_output(onnx_node, name, value_info)
        return arg0


class ExportTorchAll(Exporter):
    def __call__(self, node, func, visitor):
        arg0 = export_tensor_args(node, visitor, 'argmax_arg0', self.is_tensor, 1)[0]
        assert(len(arg0.out_node) == 1)
        i8name = visitor.gen_name()
        cast_i8_node = helper.make_node(
            'Cast', inputs=[arg0.out_node[0]], outputs=[i8name], to=int(onnx.TensorProto.INT32)
        )

        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'ReduceMin',
            inputs=[i8name],
            outputs=[name],
            keepdims=0
        )

        boolname =  visitor.gen_name()
        cast_bool_node = helper.make_node(
            'Cast', inputs=[name], outputs=[boolname], to=int(onnx.TensorProto.BOOL)
        )

        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(boolname, ty_call, visitor)
        arg0.def_nodes.append(cast_i8_node)
        arg0.def_nodes.append(onnx_node)
        arg0.set_output(cast_bool_node, boolname, value_info)
        return arg0


class ExportTorchTopK(Exporter):
    def __call__(self, node, func, visitor):
        assert(len(node.args) + self.is_tensor == 2)
        assert(len(node.keywords) == 0) 
        arg0 = export_tensor_args(node, visitor, 'argmax_arg0', self.is_tensor, 1)[0]
        ty_k = visitor.get_type_of_node(node.args[1 - self.is_tensor])
        assert isinstance(ty_k, TyNum)
        ty_k.coerce_value()
        assert ty_k.is_int() and ty_k.value is not None
        k = ty_k.value
        k_name = visitor.gen_name()
        k_node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[k_name],
            value=helper.make_tensor(
                name=k_name,
                data_type=onnx.TensorProto.INT64,
                dims=[1],
                vals=[k]
            )
        )
        ret_nodes = arg0
        ret_nodes.def_nodes.append(k_node)

        name = visitor.get_or_gen_name(node)
        output_names = [name, name + '_indices']
        onnx_node = helper.make_node(
            'TopK',
            inputs=[arg0.out_node[0], k_name],
            outputs=output_names,
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = {}
        for i, ty in enumerate(ty_call.get_tys()):
            value_info[output_names[i]] = type_to_value_info(output_names[i], ty, visitor)
        ret_nodes.set_outputs([onnx_node], output_names, value_info)
        return ret_nodes

# ---------------------------------------------------------------------

class ExportTorchLinear:
    def __call__(self, node, func, visitor):
        assert(len(node.args) == 1)
        assert(len(node.keywords) == 0)
        arg = export_tensor_arg(node.args[0], visitor, 'linear_arg0')
        assert(len(arg.out_node) == 1)
        layer = export_layer(node.func, visitor, 'linear_layer', (2,), node._func_inst)

        ty_arg = visitor.get_type_of_node(node.args[0])
        assert(isinstance(ty_arg, TyTensor))
        old_shape = []
        for ty in ty_arg.shape:
            assert(ty.has_value())
            old_shape.append(ty.value)
        new_shape = (reduce(lambda x, y: x*y, old_shape[:-1], 1), old_shape[-1])

        shape_name = visitor.gen_name()
        shape_arr = np.array(new_shape, dtype=np.int64)
        shape_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            value=onnx.helper.make_tensor(
                name=visitor.gen_name(),
                data_type=onnx.TensorProto.INT64,
                dims=[len(new_shape)],
                vals=shape_arr,
            )
        )
        name_tmp = visitor.gen_name()
        # print("[op] reshape from linear", old_shape, new_shape, name_tmp)
        reshape_node = helper.make_node('Reshape', inputs=[arg.out_node[0], shape_name], outputs=[name_tmp])

        name = visitor.get_or_gen_name(node)
        ty_call = visitor.get_type_of_node(node)

        onnx_node = helper.make_node(
            'Gemm',
            inputs=[name_tmp, layer.out_node[0], layer.out_node[1]],
            outputs=[name],
            transB=1,
            alpha=1.0,
            beta=1.0,
        )

        ty_call = visitor.get_type_of_node(node)
        # print("[OP] Gemm: {} {} -> {} transA = {} transB = {}".format(ty_arg.shape, func.weight.data.cpu().numpy().shape, ty_call.shape, 0, 1))
        final_shape = []
        for ty in ty_call.shape:
            assert(ty.has_value())
            final_shape.append(ty.value)

        shape2_name = visitor.gen_name()
        shape2_arr = np.array(final_shape, dtype=np.int64)
        shape2_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape2_name],
            value=onnx.helper.make_tensor(
                name=visitor.gen_name(),
                data_type=onnx.TensorProto.INT64,
                dims=[len(final_shape)],
                vals=shape2_arr,
            )
        )

        name_final = visitor.gen_name()
        final_node = helper.make_node('Reshape', inputs=[name, shape2_name], outputs=[name_final])

        value_info = type_to_value_info(name_final, ty_call, visitor)

        new_node = arg + layer
        new_node.def_nodes.append(shape_node)
        new_node.def_nodes.append(reshape_node)
        new_node.def_nodes.append(onnx_node)
        new_node.def_nodes.append(shape2_node)
        new_node.set_output(final_node, name_final, value_info)
        return new_node


class ExportTorchDropout:
     def __call__(self, node, func, visitor):
        assert(len(node.args) == 1)
        assert(len(node.keywords) == 0)
        arg = export_tensor_arg(node.args[0], visitor, 'dropout_node')
        return arg


class ExportTorchLayerNorm:
    def __call__(self, node, func, visitor):
        raise NotImplementedError()


class ExportTorchConv2d:
    def __call__(self, node, func, visitor):
        assert(len(node.args) == 1)
        assert(len(node.keywords) == 0)
        arg_ty = visitor.get_type_of_node(node.args[0])
        assert(isinstance(arg_ty, TyTensor))
        arg = export_tensor_arg(node.args[0], visitor, 'conv_node')
        assert(len(arg.out_node) == 1 or len(arg.out_node) == 2)
        layer = export_layer(node.func, visitor, 'conv_layer', (1, 2), node._func_inst)
        name = visitor.get_or_gen_name(node)
        for i in range(len(func.padding)):
            # else: not implemented
            assert(func.padding[i] == func.padding[0])
        onnx_node = helper.make_node(
            'Conv',
            inputs=[arg.out_node[0]] + layer.out_node,
            outputs=[name],
            dilations=func.dilation,
            group=func.groups,
            kernel_shape=func.kernel_size,
            pads=(func.padding[0],) * len(arg_ty.shape),
            strides=func.stride
        )
        ty_call = visitor.get_type_of_node(node)
        # print("[OP] Conv: {} {} -> {} dilation {} group {} kernel_shape {} pads {} strides {}".format(arg_ty.shape, func.weight.data.cpu().numpy().shape, ty_call.shape, func.dilation, func.groups, func.kernel_size, (func.padding[0],) * len(arg_ty.shape), func.stride))
        value_info = type_to_value_info(name, ty_call, visitor)
        arg.set_output(onnx_node, name, value_info)
        return arg


class ExportTorchBatchNorm2d:
    def __call__(self, node, func, visitor):
        assert(len(node.args) == 1)
        assert(len(node.keywords) == 0)
        arg = export_tensor_arg(node.args[0], visitor, 'batchnorm_node')
        assert(len(arg.out_node) == 1)
        layer = export_layer(node.func, visitor, 'batchnorm_layer', (4,), node._func_inst)
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'BatchNormalization',
            inputs=[arg.out_node[0], layer.out_node[0], layer.out_node[1], layer.out_node[2], layer.out_node[3]],
            outputs=[name],
            epsilon=func.eps,
            momentum=1-func.momentum
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg.set_output(onnx_node, name, value_info)
        return arg


class ExportTorchMaxPool2d:
    def __call__(self, node, func, visitor):
        assert(len(node.args) == 1)
        assert(len(node.keywords) == 0)
        arg_ty = visitor.get_type_of_node(node.args[0])
        assert(isinstance(arg_ty, TyTensor))
        arg = export_tensor_arg(node.args[0], visitor, 'maxpool_node')
        assert(len(arg.out_node) == 1)
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'MaxPool',
            inputs=[arg.out_node[0]],
            outputs=[name],
            kernel_shape=(func.kernel_size, func.kernel_size),
            pads=(func.padding,) * len(arg_ty.shape),
            strides=(func.stride, func.stride)
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg.set_output(onnx_node, name, value_info)
        return arg


class ExportTorchAvgPool2d:
    def __call__(self, node, func, visitor):
        assert(len(node.args) == 1)
        assert(len(node.keywords) == 0)
        arg_ty = visitor.get_type_of_node(node.args[0])
        assert(isinstance(arg_ty, TyTensor))
        arg = export_tensor_arg(node.args[0], visitor, 'avgpool_node')
        assert(len(arg.out_node) == 1)
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'AveragePool',
            inputs=[arg.out_node[0]],
            outputs=[name],
            kernel_shape=(func.kernel_size, func.kernel_size),
            pads=(func.padding,) * len(arg_ty.shape),
            strides=(func.stride, func.stride)
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg.set_output(onnx_node, name, value_info)
        return arg


class ExportTorchAdaptiveAvgPool2d:
    def __call__(self, node, func, visitor):
        assert(len(node.args) == 1)
        assert(len(node.keywords) == 0)
        arg = export_tensor_arg(node.args[0], visitor, 'adaptive_avg_node')
        assert(len(arg.out_node) == 1)
        name = visitor.get_or_gen_name(node)
        assert(func.output_size == (1, 1))
        onnx_node = helper.make_node(
            'GlobalAveragePool',
            inputs=[arg.out_node[0]],
            outputs=[name],
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg.set_output(onnx_node, name, value_info)
        return arg

class ExportTorchEmbedding:
    def __call__(self, node, func, visitor):
        assert(len(node.args) == 1)
        assert(len(node.keywords) == 0)
        arg = export_tensor_arg(node.args[0], visitor, 'embedding_node')
        assert(len(arg.out_node) == 1)
        layer = export_layer(node.func, visitor, 'embedding_layer', (1,), node._func_inst)
        name = visitor.get_or_gen_name(node)
        onnx_node = helper.make_node(
            'Gather',
            inputs=[layer.out_node[0], arg.out_node[0]],
            outputs=[name],
        )
        ty_call = visitor.get_type_of_node(node)
        value_info = type_to_value_info(name, ty_call, visitor)
        arg.set_output(onnx_node, name, value_info)
        return arg


def general_func_export(node, func, visitor, func_name):
    args = export_tensor_args(node, visitor, 'udf_node', False, len(node.args))
    ret_node = OnnxNodes()
    for arg in args:
        assert(len(arg.out_node) == 1)
        ret_node = ret_node + arg
    ty_call = visitor.get_type_of_node(node)
    if isinstance(ty_call, TyTuple):
        ret_names = []
        for i in range(ty_call.size()):
            ty_call_sub = ty_call[i]
            name = visitor.get_or_gen_name(node)
            ret_names.append(name)
            ret_node.def_value_infos[name] = type_to_value_info(name, ty_call_sub, visitor)
        onnx_node = helper.make_node(func_name, inputs=[arg.out_node[0] for arg in args], outputs=ret_names)
    else:
        ret_names = [visitor.get_or_gen_name(node)]
        onnx_node = helper.make_node(func_name, inputs=[arg.out_node[0] for arg in args], outputs=ret_names)
    ret_node.def_nodes.append(onnx_node)
    ret_node.out_node = ret_names
    return ret_node


pytorch_func_export = {
    torch.zeros: ExportTorchTensorOfShape(0, is_tensor=False),
    torch.ones: ExportTorchTensorOfShape(1, is_tensor=False),
    torch.full: ExportTorchTensorOfShape(None, is_tensor=False),
    # TorchScript also maps torch.empty to constant of zero
    torch.empty: ExportTorchTensorOfShape(0, is_tensor=False),
    torch.matmul: ExportTorchMatmul(is_tensor=False),
    torch.mm: ExportTorchMatmul(is_tensor=False),
    torch.sigmoid: ExportTorchElementWise('Sigmoid', is_tensor=False),
    torch.tanh: ExportTorchElementWise('Tanh', is_tensor=False),
    torch.relu: ExportTorchElementWise('Relu', is_tensor=False),
    torch.erf: ExportTorchElementWise('Erf', is_tensor=False),
    torch.sqrt: ExportTorchElementWise('Sqrt', is_tensor=False),
    torch.sum: ExportTorchSum(is_tensor=False),
    torch.max: ExportTorchMax(is_tensor=False),
    torch.transpose: ExportTorchTranspose(is_tensor=False),
    torch.split: ExportTorchSplit(is_tensor=False),
    torch.cat: ExportTorchCat(is_tensor=False),
    torch.softmax: ExportTorchSoftmax(is_tensor=False),
    torch.mean: ExportTorchMean(is_tensor=False),
    torch.flatten: ExportTorchFlatten(is_tensor=False),
    torch.tensor: ExportTorchIdentity(is_tensor=False),
    torch.reshape: ExportTorchReshape(is_tensor=False),
    torch.gt: ExportTorchCompare('Greater', is_tensor=False),
    torch.lt: ExportTorchCompare('Less', is_tensor=False),
    torch.eq: ExportTorchCompare('Equal', is_tensor=False),
    torch.all: ExportTorchAll(is_tensor=False),
    torch.zeros_like: ExportTorchConstLike(0.0, is_tensor=False),
    torch.ones_like: ExportTorchConstLike(1.0, is_tensor=False),
    torch.full_like: ExportTorchConstLike(None, is_tensor=False),
    torch.where: ExportTorchWhere(is_tensor=False),
    torch.log_softmax: ExportTorchLogSoftmax(is_tensor=False),

    torch.Tensor.size: ExportTorchShape(is_tensor=True),
    torch.Tensor.expand_as: ExportTorchExpandAs(is_tensor=True),
    torch.Tensor.view: ExportTorchView(is_tensor=True),
    torch.Tensor.reshape: ExportTorchReshape(is_tensor=True),
    torch.Tensor.permute: ExportTorchPermute(is_tensor=True),
    torch.Tensor.transpose: ExportTorchTranspose(is_tensor=True),
    torch.Tensor.item: ExportTorchIdentity(is_tensor=True),
    torch.Tensor.split: ExportTorchSplit(is_tensor=True),
    torch.Tensor.float: ExportTorchCast(type_np_to_onnx(np.float32), is_tensor=True),
    torch.Tensor.topk: ExportTorchTopK(is_tensor=True),
    torch.Tensor.argmax: ExportTorchArgmax(is_tensor=True)
}

pytorch_layer_export = {
    nn.Linear: ExportTorchLinear(),
    nn.Dropout: ExportTorchDropout(),
    nn.LayerNorm: ExportTorchLayerNorm(),
    nn.Conv2d: ExportTorchConv2d(),
    nn.BatchNorm2d: ExportTorchBatchNorm2d(),
    nn.ReLU: ExportTorchElementWise('Relu', is_tensor=False),
    nn.MaxPool2d: ExportTorchMaxPool2d(),
    nn.AvgPool2d: ExportTorchAvgPool2d(),
    nn.AdaptiveAvgPool2d: ExportTorchAdaptiveAvgPool2d(),
    nn.Embedding: ExportTorchEmbedding(),
}

builtin_func_export = {
    math.sqrt: ExportTorchElementWise('Sqrt', is_tensor=False),
    int: ExportTorchCast(type_np_to_onnx(np.int64), is_tensor=False),
}
