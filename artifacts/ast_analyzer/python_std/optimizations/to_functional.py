""" Replace constant variables in the object with their value """

from ast_analyzer.python_std.passmanager import Transformation
from ast_analyzer.python_std.conversion import to_ast
import gast as ast
import astunparse

prefix = '_se_'

class Functional(Transformation):
    def __init__(self):
        super(Functional, self).__init__()

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            self.update = True
            name = node.attr
            self.new_params.add(name)
            return ast.Name(id = prefix + name, ctx = node.ctx, annotation=None, type_comment=None)
        return node

    def visit_FunctionDef(self, node):
        self.new_params = set()
        self.generic_visit(node)
        for n in self.new_params:
            node.args.args.append(ast.Name(id = prefix + n, ctx = ast.Param(), annotation=None, type_comment=None))
        return node
        

class OOPStyle(Transformation):
    def __init__(self):
        super(OOPStyle, self).__init__()

    def visit_Name(self, node):
        if node.id.startswith(prefix):
            self.update = True
            return ast.Attribute(
                value = ast.Name(id = 'self', ctx = node.ctx, annotation=None, type_comment=None),
                attr = node.id[len(prefix):],
                ctx = node.ctx
            )
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.args.args = [n for n in node.args.args if not isinstance(n, ast.Attribute)]
        return node

