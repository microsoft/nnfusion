import gast
import ast
import types
import astunparse
import torch
from .to_torch_func import to_torch_func
import importlib
import copy
from ast_analyzer.shape_inference.types import *
import numpy as np
from ast_analyzer.utils.unparse import unparse_ast_list
import os
import sys
import subprocess
from ast_analyzer.utils import config
from nnfusion.session import codegen, modify_nnfusion_rt, build

def inject(inst, n_in, n_ret, n_save, is_train, pre_stmts, model_name, self_attrs, backend, func_name = "__dev_impl__"):
    to_torch_func(model_name, n_in, n_ret, n_save, is_train, self_attrs, backend, [True for _ in range(n_in)])
    torch_func = importlib.import_module(f'{model_name}.Model_{backend}')
    if backend == 'onnx':
        setattr(inst, func_name, torch_func.GenOnnxModel)
        new_body = gast.parse('print("Run Onnx Code")').body
        new_body.extend(pre_stmts)
        call_stmt = "return self.{}.apply({})".format(
            func_name,
            ", ".join(["_tensor_{}".format(i) for i in range(n_in)]))
    elif backend == 'tf':
        setattr(inst, func_name, torch_func.GenTFModel)
        new_body = gast.parse('print("Run TensorFlow Code")').body
        new_body.extend(pre_stmts)

        call_stmt = "return self.{}.apply({})".format(
            func_name,
            ", ".join(["_tensor_{}".format(i) for i in range(n_in)]))
    elif backend == 'nnfusion':
        setattr(inst, func_name, torch_func.GenNNFusionModel)
        # new_body = gast.parse('print("Run NNFusion Code")').body
        # new_body.extend(pre_stmts)
        new_body = pre_stmts

        call_stmt = "return self.{}.apply({})".format(
            func_name,
            ", ".join(["_tensor_{}".format(i) for i in range(n_in)]))
    else:
        raise NotImplementedError

    new_body.extend(gast.parse(call_stmt).body)
    new_ast = inst.__ast__
    new_ast.body[0].body = new_body
    new_ast.body[0].name = 'forward'
    node = gast.gast_to_ast(new_ast)
    node = ast.fix_missing_locations(node)

    print(f"[to inject {backend}]")
    print(astunparse.unparse(new_ast))
    # inst.__ast__ = new_ast

    code = compile(node, 'to_inject.py', 'exec')
    namespace = {
        'torch': torch,
        'TyTorchTensor': TyTorchTensor,
        'np': np
    }
    exec(code, namespace)
    return types.MethodType(namespace[new_ast.body[0].name], inst), new_ast


def build_nnfusion(workdir):
    raise NotImplementedError("shouldn't be called")



def inject_method(scope, args, rets, n_in, n_ret, n_save, is_train, model_name, self_attrs, arg_is_scalar, platform, func_name = "__dev_impl__", onnx_model=None):
    backends = {
        'onnx': 'onnx',
        'nnf_fix_flag': 'nnf_fix_flag',
        'nnf_load': 'nnf_load'
    }
    for backend in backends:
        to_torch_func(model_name, n_in, n_ret, n_save, is_train, self_attrs, backend, arg_is_scalar, platform)
    new_body = []
    call_stmt = "{} = self.{}.apply({})".format(
        ", ".join(rets),
        func_name,
        ", ".join([astunparse.unparse(arg) for arg in args]))
    new_body.extend(gast.parse(call_stmt).body)
    return new_body, (scope, func_name, model_name, onnx_model)


def inject_training(scope, node, arg_nodes, model_name):
    func_name = model_name + "_func"
    args = ", ".join([astunparse.unparse(nd).replace("\n", "") for nd in arg_nodes])
    new_body = []
    call_stmt = "return self.{}.apply({})".format(func_name, args)
    new_body.extend(gast.parse(call_stmt).body)
    node.body[0].body = new_body
    node.body[0].name = 'forward'

    print(f"[to inject training]")
    print(astunparse.unparse(node))
    ast_node = gast.gast_to_ast(node)
    ast_node = ast.fix_missing_locations(ast_node)

    code = compile(ast_node, 'to_inject.py', 'exec')
    namespace = {
        'torch': torch,
        'TyTorchTensor': TyTorchTensor,
        'np': np
    }
    exec(code, namespace)
    return types.MethodType(namespace[ast_node.body[0].name], scope), node, func_name, (scope, func_name, 'tmp.Model' + model_name + '_train')
