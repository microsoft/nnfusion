""" Optimization for Python costly pattern. """

from ast_analyzer.python_std.analyses import Check, Placeholder
from ast_analyzer.python_std.passmanager import Transformation

from copy import deepcopy
import gast as ast
import astunparse


class Pattern(object):

    def match(self, node):
        self.check = Check(node, dict())
        return self.check.visit(self.pattern)

    def replace(self):
        sub = self.sub()
        if type(sub) is list:
            return [PlaceholderReplace(self.check.placeholders).visit(x) for x in sub]
        else:
            return PlaceholderReplace(self.check.placeholders).visit(self.sub())

    def imports(self):
        return deepcopy(getattr(self, 'extra_imports', []))

class EnumeratePattern(Pattern):
    # (i, x) = builtins.enumerate(lst)[0] => i = 0; x = lst[0]
    pattern = ast.Assign(
        targets = [ast.Tuple(elts = [Placeholder(0), Placeholder(1)], ctx = ast.Store())],
        value = ast.Subscript(
            value = ast.Call(func = ast.Name(id='enumerate', ctx=ast.Load(), annotation=None, type_comment=None),
                             args=[Placeholder(2)],
                             keywords=[]),
            slice = Placeholder(3),
            ctx = ast.Load()
        )
    )

    @staticmethod
    def sub():
        return [
            ast.Assign(targets = [Placeholder(0)], value = Placeholder(3)),
            ast.Assign(targets = [Placeholder(1)],
                   value = ast.Subscript(value = Placeholder(2), slice = ast.Index(value = Placeholder(3)), ctx = ast.Load()))
        ]

know_pattern = [x for x in globals().values() if hasattr(x, "pattern")]


class PlaceholderReplace(Transformation):

    """ Helper class to replace the placeholder once value is collected. """

    def __init__(self, placeholders):
        """ Store placeholders value collected. """
        self.placeholders = placeholders
        super(PlaceholderReplace, self).__init__()

    def visit(self, node):
        """ Replace the placeholder if it is one or continue. """
        if isinstance(node, Placeholder):
            return self.placeholders[node.id]
        else:
            return super(PlaceholderReplace, self).visit(node)


class PatternTransform(Transformation):

    """
    Replace all known pattern by pythran function call.

    Based on BaseMatcher to search correct pattern.
    """

    def __init__(self):
        """ Initialize the Basematcher to search for placeholders. """
        super(PatternTransform, self).__init__()

    def visit(self, node):
        """ Try to replace if node match the given pattern or keep going. """
        for pattern in know_pattern:
            matcher = pattern()
            if matcher.match(node):
                node = matcher.replace()
                self.update = True
        if type(node) is list:
            new_nodes = []
            for n in node:
                new_nodes.append(super(PatternTransform, self).visit(n))
            node = new_nodes
        else:
            node = super(PatternTransform, self).visit(node)
        if hasattr(node, "body") and type(node.body) is list:
            new_body = []
            for stmt in node.body:
                if type(stmt) is list:
                    new_body += stmt
                else:
                    new_body.append(stmt)
            node.body = new_body
        return node
        

class ListToNode(Transformation):
    def __init__(self):
        super().__init__()

    def visit(self, node):
        self.generic_visit(node)
        if hasattr(node, "body") and type(node.body) is list:
            new_body = []
            for stmt in node.body:
                if type(stmt) is list:
                    new_body += stmt
                else:
                    new_body.append(stmt)
            node.body = new_body
        return node
