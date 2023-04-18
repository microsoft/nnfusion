from ast import iter_fields, AST
import gast
from ast_analyzer.grad import annotations as anno
import astunparse

# may_push is set to false if its subAST contains statements that must be executed in python
# e.g., unknown callee function

class MarkMayPush(gast.NodeVisitor):
    def __init__(self):
        super(MarkMayPush, self).__init__()
        self.call_stack = []

    def visit_arguments(self, node):
        # skip type annotations
        anno.setanno(node, 'may_push', True)
        return True

    def visit_FunctionDef(self, node):
        may_push = True
        for field, value in iter_fields(node):
            # skip type annotations
            if field == 'returns':
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        may_push = self.visit(item) and may_push
            elif isinstance(value, AST):
                may_push = self.visit(value) and may_push
        anno.setanno(node, 'may_push', may_push)
        return may_push

    def visit_Call(self, node):
        may_push = self.generic_visit(node)
        # if the function is unrecgonized
        if not hasattr(node, '_func_inst') or node._func_inst is None:
            may_push = False
        if node._func_inst == print:
            may_push = False
        if hasattr(node, 'ctx_types'):
            may_push = False
        # if it is a recursive function
        # elif node._is_recursive:
        #     may_push = False
        # 'may_push' have been set in "generic_visit"
        anno.setanno(node, 'may_push', may_push, safe=False)
        return may_push

    def visit_Attribute(self, node):
        if node.attr == 'saved_tensors':
            return False
        if node.attr == 'ones': # TODO: not hack it
            return False
        return self.generic_visit(node)
    
    def visit_Assign(self, node): # TODO: not hack it
        may_push = self.generic_visit(node)
        if isinstance(node.value, gast.Call) and isinstance(node.value.func, gast.Attribute) and node.value.func.attr == 'item':
            return False
        else:
            return may_push

    def generic_visit(self, node):
        may_push = True
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        # cannot write as "may_push and self.visit(item)", because when "may_push=False", the self.visit will be skipped
                        may_push = self.visit(item) and may_push
            elif isinstance(value, AST):
                may_push = self.visit(value) and may_push
        anno.setanno(node, 'may_push', may_push)
        return may_push


def mark_may_push(node):
    # handle recursive in device compiler
    MarkMayPush().visit(node)
    return
