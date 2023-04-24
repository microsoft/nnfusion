import torch
import copy
from torch import nn
from onnx import numpy_helper, helper
import astunparse
from .utils import np_inst_to_value_info
from ast_analyzer.grad import annotations as anno
from ast_analyzer.shape_inference.types import *
import gast

__all__ = ['pytorch_layer_initializer']

# __call__:  # self, obj, str, ExportEngine -> List[TensorProto], Map[name, TypeInfoProto], List[ASTNode], List[TYPE]


class InitializerTorchParameter():
    def __call__(self, obj, name, node, visitor):
        numpy_value = obj.data.cpu().numpy()
        return [numpy_helper.from_array(numpy_value, name=name)], {name: np_inst_to_value_info(name, numpy_value, visitor)}, [copy.deepcopy(node)], [anno.getanno(node,  'type')]


class InitializerTorchLinear():
    def __call__(self, obj, name, node, visitor):
        weight_value = obj.weight.data.cpu().numpy()
        bias_value = obj.bias.data.cpu().numpy()
        weight_name = name + "_weight"
        bias_name = name + "_bias"
        return [
            numpy_helper.from_array(weight_value, name=weight_name), 
            numpy_helper.from_array(bias_value, name=bias_name)
        ], {
            weight_name: np_inst_to_value_info(weight_name, weight_value, visitor),
            name+"_bias": np_inst_to_value_info(bias_name, bias_value, visitor)
        }, [
            gast.Attribute(value=copy.deepcopy(node), attr='weight', ctx=gast.Load()),
            gast.Attribute(value=copy.deepcopy(node), attr='bias', ctx=gast.Load())
        ], [
            TyTorchTensor(weight_value.dtype, shape=weight_value.shape, ndim=len(weight_value.shape)),
            TyTorchTensor(bias_value.dtype, shape=bias_value.shape, ndim=len(bias_value.shape))
        ]


class InitializerTorchConv2d():
    def __call__(self, obj, name, node, visitor):
        weight_value = obj.weight.data.cpu().numpy()
        weight_name = name + "_weight"
        if obj.bias is None:
            return [
                numpy_helper.from_array(weight_value, name=weight_name), 
            ], {
                weight_name: np_inst_to_value_info(weight_name, weight_value, visitor),
            }, [
                gast.Attribute(value=copy.deepcopy(node), attr='weight', ctx=gast.Load()),
            ], [
                TyTorchTensor(weight_value.dtype, shape=weight_value.shape, ndim=len(weight_value.shape)),
            ]
        else:
            bias_value = obj.bias.data.cpu().numpy()
            bias_name = name + "_bias"
            return [
                numpy_helper.from_array(weight_value, name=weight_name), 
                numpy_helper.from_array(bias_value, name=bias_name), 
            ], {
                weight_name: np_inst_to_value_info(weight_name, weight_value, visitor),
                bias_name: np_inst_to_value_info(bias_name, bias_value, visitor),
            }, [
                gast.Attribute(value=copy.deepcopy(node), attr='weight', ctx=gast.Load()),
                gast.Attribute(value=copy.deepcopy(node), attr='bias', ctx=gast.Load()),
            ], [
                TyTorchTensor(weight_value.dtype, shape=weight_value.shape, ndim=len(weight_value.shape)),
                TyTorchTensor(bias_value.dtype, shape=bias_value.shape, ndim=len(bias_value.shape)),
            ]


class InitializerTorchBatchNorm2d():
    def __call__(self, obj, name, node, visitor):
        weight_value = obj.weight.data.cpu().numpy()
        bias_value = obj.bias.data.cpu().numpy()
        mean_value = obj.running_mean.data.cpu().numpy()
        var_value = obj.running_var.data.cpu().numpy()
        weight_name = name + "_weight"
        bias_name = name + "_bias"
        mean_name = name + "_mean"
        var_name = name + "_var"
        return [
            numpy_helper.from_array(weight_value, name=weight_name), 
            numpy_helper.from_array(bias_value, name=bias_name),
            numpy_helper.from_array(mean_value, name=mean_name),
            numpy_helper.from_array(var_value, name=var_name)
        ], {
            weight_name: np_inst_to_value_info(weight_name, weight_value, visitor),
            bias_name: np_inst_to_value_info(bias_name, bias_value, visitor),
            mean_name: np_inst_to_value_info(mean_name, mean_value, visitor),
            var_name: np_inst_to_value_info(var_name, var_value, visitor)
        }, [
            gast.Attribute(value=copy.deepcopy(node), attr='weight', ctx=gast.Load()),
            gast.Attribute(value=copy.deepcopy(node), attr='bias', ctx=gast.Load()),
            gast.Attribute(value=copy.deepcopy(node), attr='running_mean', ctx=gast.Load()),
            gast.Attribute(value=copy.deepcopy(node), attr='running_var', ctx=gast.Load())
        ], [
            TyTorchTensor(weight_value.dtype, shape=weight_value.shape, ndim=len(weight_value.shape)),
            TyTorchTensor(bias_value.dtype, shape=bias_value.shape, ndim=len(bias_value.shape)),
            TyTorchTensor(mean_value.dtype, shape=mean_value.shape, ndim=len(mean_value.shape)),
            TyTorchTensor(var_value.dtype, shape=var_value.shape, ndim=len(var_value.shape))
        ]


class InitializerTorchEmbedding():
    def __call__(self, obj, name, node, visitor):
        weight_value = obj.weight.data.cpu().numpy()
        weight_name = name + "_weight"
        return [
            numpy_helper.from_array(weight_value, name=weight_name), 
        ], {
            weight_name: np_inst_to_value_info(weight_name, weight_value, visitor),
        }, [
            gast.Attribute(value=copy.deepcopy(node), attr='weight', ctx=gast.Load()),
        ], [
            TyTorchTensor(weight_value.dtype, shape=weight_value.shape, ndim=len(weight_value.shape)),
        ]


pytorch_layer_initializer = {
    nn.parameter.Parameter: InitializerTorchParameter(),
    nn.Linear: InitializerTorchLinear(),
    nn.Conv2d: InitializerTorchConv2d(),
    nn.BatchNorm2d: InitializerTorchBatchNorm2d(),
    nn.Embedding: InitializerTorchEmbedding(),
}
