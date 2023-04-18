""" NormalizeIfElse transform early exit in if into if-else. """

from ast_analyzer.python_std.analyses import Liveness
from ast_analyzer.python_std.passmanager import Transformation

import gast as ast
import astunparse

def equal_assign(name):
    return ast.Assign(
        targets=[ast.Name(id = name, ctx = ast.Store(), annotation = None, type_comment = None)],
        value = ast.Name(id = name, ctx = ast.Load(), annotation=None, type_comment=None)
    )

class FillIfDef(Transformation):
    def __init__(self):
        super(FillIfDef, self).__init__(Liveness)

    def visit_If(self, node):
        body_defs = self.liveness[node]['body']
        orelse_defs = self.liveness[node]['orelse']
        
        for defs in body_defs - orelse_defs:
            node.orelse.append(equal_assign(defs))
            self.update = True

        for defs in orelse_defs - body_defs:
            node.body.append(equal_assign(defs))
            self.update = True
        
        return node
