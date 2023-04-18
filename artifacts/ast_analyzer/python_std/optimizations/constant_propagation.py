""" DeadCodeElimination remove useless code. """

from ast_analyzer.python_std.analyses import PureExpressions, UseDefChains, Ancestors, ConstantExpressions
from ast_analyzer.python_std.passmanager import Transformation
from ast_analyzer.python_std.conversion import to_ast

import gast as ast
import astunparse
from copy import deepcopy

class ConstantPropagation(Transformation):
    def __init__(self):
        super(ConstantPropagation, self).__init__(PureExpressions,
                                                  UseDefChains,
                                                  Ancestors,
                                                  ConstantExpressions)
        self.const_dict = {}

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Constant) and len(node.targets) == 1:
            self.const_dict[node.targets[0]] = node.value
        elif isinstance(node.value, ast.Tuple):
            all_const = True
            for val in node.value.elts:
                if not isinstance(node.value, ast.Constant):
                    all_const = False
                    break
            if all_const and len(node.targets) == 1:
                self.const_dict[node.targets[0]] = node.value
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        try:
            value = eval(astunparse.unparse(node), {'builtins': __import__('builtins')})
            new_node = to_ast(value)
            self.update = True
            return new_node
        except Exception as e:
            return node

    def visit_Name(self, node):
        self.generic_visit(node)
        if node not in self.use_def_chains or len(self.use_def_chains[node]) != 1:
            return node
        src = self.use_def_chains[node][0].node
        if src in self.const_dict:
            self.update = True
            return deepcopy(self.const_dict[src])
        else:
            return node