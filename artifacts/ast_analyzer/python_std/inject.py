import gast
import ast
import types
import astunparse
import torch
from ast_analyzer.shape_inference.types import *
import numpy as np
from ast_analyzer import grad

def inject(inst, funcAST):
    node = gast.gast_to_ast(funcAST)
    node = ast.fix_missing_locations(node)
    new_stmt = ast.parse('print("Run Compiled Code")')
    node.body[0].body = node.body[0].body[:-1] + \
        new_stmt.body + node.body[0].body[-1:]
    code = compile(node, 'inject.py', 'exec')
    namespace = {
        'torch': torch,
        'TyTorchTensor': TyTorchTensor,
        'np': np
    }
    exec(code, namespace)
    inst.forward = types.MethodType(namespace['forward'], inst)

def inject_fwdbwd(inst, funcAST):
    node = gast.gast_to_ast(funcAST)
    node = ast.fix_missing_locations(node)
    # new_stmt = ast.parse('print("Run Grad Forward Code")')
    # node.body[0].body = node.body[0].body[:-1] + \
    #     new_stmt.body + node.body[0].body[-1:]
    # new_stmt = ast.parse('print("Run Grad Backward Code")')
    # node.body[1].body = node.body[1].body[:-1] + \
    #     new_stmt.body + node.body[1].body[-1:]
    
    code = compile(node, 'inject.py', 'exec')
    namespace = {
        'torch': torch,
        'TyTorchTensor': TyTorchTensor,
        'np': np,
        'grad': grad.impl
    }
    exec(code, namespace)
    # print(namespace.keys())
    inst.grad_forward = types.MethodType(namespace[funcAST.body[0].name], inst)
    inst.grad_backward = types.MethodType(namespace[funcAST.body[1].name], inst)
