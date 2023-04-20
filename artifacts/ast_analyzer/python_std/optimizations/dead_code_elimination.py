""" DeadCodeElimination remove useless code. """

from ast_analyzer.python_std.passmanager import Transformation

import gast as ast

class ClumsyOpenMPDependencyHandler(ast.NodeVisitor):

    def __init__(self):
        self.blacklist = set()

    def visit_OMPDirective(self, node):
        for dep in node.deps:
            if isinstance(dep, ast.Name):
                self.blacklist.add(dep.id)
        return node

# WARNING: simplified by me
class DeadCodeElimination(Transformation):
    """
    Remove useless statement like:
        - assignment to unused variables
        - remove alone pure statement
        - remove empty if

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> pm = passmanager.PassManager("test")
    >>> node = ast.parse("def foo(): a = [2, 3]; return 1")
    >>> _, node = pm.apply(DeadCodeElimination, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        pass
        return 1
    >>> node = ast.parse("def foo(): 'a simple string'; return 1")
    >>> _, node = pm.apply(DeadCodeElimination, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        pass
        return 1
    >>> node = ast.parse('''
    ... def bar(a):
    ...     return a
    ... def foo(a):
    ...    bar(a)
    ...    return 1''')
    >>> _, node = pm.apply(DeadCodeElimination, node)
    >>> print(pm.dump(backend.Python, node))
    def bar(a):
        return a
    def foo(a):
        pass
        return 1
    """
    def __init__(self):
        super(DeadCodeElimination, self).__init__()
        self.blacklist = set()

    def used_target(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.blacklist:
                return True
            chain = self.def_use_chains.chains[node]
            return bool(chain.users())
        return True

    def visit_FunctionDef(self, node):
        codh = ClumsyOpenMPDependencyHandler()
        codh.visit(node)
        self.blacklist = codh.blacklist
        return self.generic_visit(node)

    # def visit_Pass(self, node):
    #     ancestor = self.ancestors[node][-1]
    #     if getattr(ancestor, 'body', ()) == [node]:
    #         return node
    #     if getattr(ancestor, 'orelse', ()) == [node]:
    #         return node
    #     return None

    # def visit_Assign(self, node):
    #     return node
        # targets = [target for target in node.targets
        #            if self.used_target(target)]
        # if len(targets) == len(node.targets):
        #     return node
        # node.targets = targets
        # self.update = True
        # if targets:
        #     return node
        # if node.value in self.pure_expressions:
            # return ast.Pass()
        # else:
        #     return ast.Expr(value=node.value)

    def visit_Expr(self, node):
        if (not isinstance(node.value, ast.Yield)):
            self.update = True
            return ast.Pass()
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        self.generic_visit(node)

        if isinstance(node.test, ast.Constant) and node.test.value in (True, False):
            self.update = True
            if node.test.value:
                return node.body
            else:
                return node.orelse
        
        try:
            if ast.literal_eval(node.test):
                self.update = True
                return node.body
            else:
                self.update = True
                return node.orelse
        except ValueError as e:
            # not a constant expression
            pass

        # if node.test in self.constant_expressions: # Compare Node
        #     try:
        #         value = eval(astunparse.unparse(node.test), {'builtins': __import__('builtins')})
        #         if value:
        #             self.update = True
        #             return node.body
        #         else:
        #             self.update = True
        #             return node.orelse
        #     except Exception as e:
        #         print("exception", e)
        #         # not a constant expression
        #         pass

        have_body = any(not isinstance(x, ast.Pass) for x in node.body)
        have_else = any(not isinstance(x, ast.Pass) for x in node.orelse)
        # If the "body" is empty but "else content" is useful, switch branches
        # and remove else content
        if not have_body and have_else:
            test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.update = True
            return ast.If(test=test, body=node.orelse, orelse=list())
        # if neither "if" and "else" are useful, keep test if it is not pure
        elif not have_body:
            self.update = True
            if node.test in self.pure_expressions:
                return ast.Pass()
            else:
                node = ast.Expr(value=node.test)
                self.generic_visit(node)
        return node

