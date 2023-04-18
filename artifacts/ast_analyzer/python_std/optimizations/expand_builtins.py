""" ExpandBuiltins replaces builtins by their full paths. """

from ast_analyzer.python_std.analyses import Globals, Locals
from ast_analyzer.python_std.passmanager import Transformation
from ast_analyzer.python_std.syntax import PythranSyntaxError
from ast_analyzer.python_std.tables import MODULES

import gast as ast


class ExpandBuiltins(Transformation):

    """
    Expands all builtins into full paths.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(): return list()")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(ExpandBuiltins, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        return builtins.list()
    """

    def __init__(self):
        Transformation.__init__(self, Locals, Globals)

    def visit_Name(self, node):
        s = node.id
        if(isinstance(node.ctx, ast.Load) and
           s not in self.locals[node] and
           s not in self.globals and
           s in MODULES['builtins']):
            if s == 'getattr':
                raise PythranSyntaxError("You fool! Trying a getattr?", node)
            self.update = True
            return ast.Attribute(
                ast.Name('builtins', ast.Load(), None, None),
                s,
                node.ctx)
        else:
            return node


class RemoveBuiltins(Transformation):
    def __init__(self):
        Transformation.__init__(self, Locals, Globals)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "builtins":
            return ast.Name(node.attr, node.ctx, None, None)
        return node
