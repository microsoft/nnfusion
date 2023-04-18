""" Replace constant variables in the object with their value """

from ast_analyzer.python_std.passmanager import Transformation
from ast_analyzer.python_std.conversion import to_ast
import gast as ast
import astunparse
from ast_analyzer.grad import annotations as anno
from ast_analyzer.shape_inference.types import *
from ast_analyzer.grad.grammar import STATEMENTS

class ReplaceObjConst(Transformation):
    def __init__(self):
        super(ReplaceObjConst, self).__init__()

    def visit_Attribute(self, node):
        try:
            value = eval(
                compile(ast.gast_to_ast(ast.Expression(node)), '<obj_const>', 'eval'),
                {'builtins': __import__('builtins'), 'self': self.obj}
            )
            return to_ast(value)
        except Exception as e:
            pass
        return node


class ReplaceInferedConst(Transformation):
    def visit_arguments(self, node):
        return node

    def visit_Return(self, node):
        return node

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        node = visitor(node)
        if not isinstance(node, STATEMENTS) and anno.hasanno(node, 'type') and not isinstance(getattr(node, 'ctx', None), ast.Store):
            ty = anno.getanno(node, 'type')
            new_node = fixed_num_to_ast(ty)
            if new_node is not None:
                return new_node
        return node

