""" Optimization for Python costly pattern. """

from ast_analyzer.python_std.conversion import mangle
from ast_analyzer.python_std.analyses import Check, Placeholder
from ast_analyzer.python_std.passmanager import Transformation

from copy import deepcopy
import gast as ast

tensor_funcs = ['size', 'view', 'permute', 't',
                'float', 'item', 'masked_fill', 'fill_',
                'expand_as', 'copy_']


class ToTorchTransform(Transformation):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) \
                and node.func.attr in tensor_funcs \
                and isinstance(node.func.ctx, ast.Load):
            self.update = True
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='torch', ctx=ast.Load(
                ), annotation=None, type_comment=None), attr=node.func.attr, ctx=ast.Load()),
                args=[node.func.value] + node.args,
                keywords=[]
            )
        return node


class ToTensorTransform(Transformation):

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == 'torch' \
                and node.func.attr in tensor_funcs:
            self.update = True
            return ast.Call(
                func=ast.Attribute(
                    value=node.args[0], attr=node.func.attr, ctx=ast.Load()),
                args=node.args[1:],
                keywords=[]
            )
        return node

class CopyToAssign(Transformation):

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            call_node = node.value
            if isinstance(call_node.func, ast.Attribute) and call_node.func.attr == 'copy_':
                target = call_node.func.value
                call_node.func.ctx = ast.Store()
                assert(len(call_node.args) == 1)
                return ast.Assign(
                    targets=[target],
                    value=call_node.args[0]
                )
        return node
