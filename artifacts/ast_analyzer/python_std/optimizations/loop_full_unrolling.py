""" LoopFullUnrolling fully unrolls loops with static bounds. """

from ast_analyzer.python_std.analyses import HasBreak, HasContinue, NodeCount
from ast_analyzer.python_std.conversion import to_ast
from ast_analyzer.python_std.passmanager import FunctionAnalysis, Transformation, NodeAnalysis
from ast_analyzer.python_std.get_obj import get_obj
from ast_analyzer.utils.unparse import unparse_ast_list

from copy import deepcopy
import gast as ast
import astunparse
from ast_analyzer.grad import annotations as anno
import torch
from ast_analyzer.shape_inference.types import *


class DynamicLoopSplit(Transformation):
    UNROLL_SIZE_FOR = 8
    UNROLL_SIZE_WHILE = 2

    def visit_For(self, node):
        self.generic_visit(node)
        has_break = any(self.gather(HasBreak, n)
                        for n in node.body)
        has_cont = any(self.gather(HasContinue, n)
                       for n in node.body)

        if has_break or has_cont:
            return node
        
        dyn_loops = []
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.For):
                try:
                    _ = eval(astunparse.unparse(sub_node.iter), {'builtins': __import__('builtins'), 'self': self.obj})
                except Exception as e:
                    dyn_loops.append(sub_node)
        
        if not (len(dyn_loops) == 1 and dyn_loops[0] == node): return node
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range'):
            # only support for i in range(xxx) currently
            raise NotImplementedError
        if len(node.orelse) > 0:
            raise NotImplementedError

        outer_target = f"_outer_{node.target.id}"
        inner_target = f"_inner_{node.target.id}"
        split_target = f"_split_{node.target.id}"

        iter_visited = False
        for stmt in node.body:
            for sub_node in ast.walk(stmt):
                if isinstance(sub_node, ast.Name) and sub_node.id == node.target.id and isinstance(sub_node.ctx, ast.Load):
                    iter_visited = True
        iter_stmts_large = []
        iter_stmts_small = []
        if iter_visited:
            iter_stmts_large.append(ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store(), annotation=None, type_comment=None)], value=ast.BinOp(left=ast.BinOp(left=ast.Name(id=outer_target, ctx=ast.Load(), annotation=None, type_comment=None), op=ast.Mult(), right=ast.Constant(value=DynamicLoopSplit.UNROLL_SIZE_FOR, kind=None)), op=ast.Add(), right=ast.Name(id=inner_target, ctx=ast.Load(), annotation=None, type_comment=None))))
            iter_stmts_small.append(ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store(), annotation=None, type_comment=None)], value=ast.BinOp(left=ast.BinOp(left=ast.BinOp(left=node.iter.args[0], op=ast.FloorDiv(), right=ast.Constant(value=DynamicLoopSplit.UNROLL_SIZE_FOR, kind=None)), op=ast.Mult(), right=ast.Constant(value=DynamicLoopSplit.UNROLL_SIZE_FOR, kind=None)), op=ast.Add(), right=ast.Name(id=split_target, ctx=ast.Load(), annotation=None, type_comment=None))))

        large_node = ast.For(
            target=ast.Name(id=outer_target, ctx=ast.Store(), annotation=None, type_comment=None),
            iter=ast.Call(
                func=ast.Name(id='range', ctx=ast.Load(), annotation=None, type_comment=None),
                args=[ast.BinOp(left=node.iter.args[0], op=ast.FloorDiv(), right=ast.Constant(value=DynamicLoopSplit.UNROLL_SIZE_FOR, kind=None))],
                keywords=[]
            ),
            body=[
                ast.For(
                    target=ast.Name(id=inner_target, ctx=ast.Store(), annotation=None, type_comment=None),
                    iter=ast.Call(
                        func=ast.Name(id='range', ctx=ast.Load(), annotation=None, type_comment=None),
                        args=[ast.Constant(value=DynamicLoopSplit.UNROLL_SIZE_FOR, kind=None)],
                        keywords=[]
                    ),
                    body=deepcopy(iter_stmts_large) + deepcopy(node.body),
                    orelse=[],
                    type_comment=None
                )
            ],
            orelse=[],
            type_comment=None,
        )

        small_node = ast.For(
            target=ast.Name(id=split_target, ctx=ast.Store(), annotation=None, type_comment=None),
            iter=ast.Call(
                func=ast.Name(id='range', ctx=ast.Load(), annotation=None, type_comment=None),
                args=[ast.BinOp(left=node.iter.args[0], op=ast.Mod(), right=ast.Constant(value=DynamicLoopSplit.UNROLL_SIZE_FOR, kind=None))],
                keywords=[]
            ),
            body= deepcopy(iter_stmts_small) + deepcopy(node.body),
            orelse=[],
            type_comment=None
        )

        return [large_node, small_node]
    
    def visit_While(self, node):
        self.generic_visit(node)
        has_break = any(self.gather(HasBreak, n)
                        for n in node.body)
        has_cont = any(self.gather(HasContinue, n)
                       for n in node.body)

        if has_break or has_cont:
            return node

        if_node = ast.If(
            test = deepcopy(node.test),
            body = deepcopy(node.body),
            orelse = []
        )

        body = node.body
        for _ in range(DynamicLoopSplit.UNROLL_SIZE_WHILE - 1):
            body.append(deepcopy(if_node))

        while_node = ast.While(
            test = node.test,
            body = body,
            orelse=[]
        )

        return while_node
        

class LoopFullUnrolling(Transformation):
    '''
    Fully unroll loops with static bounds

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse('for j in [1,2,3]: i += j')
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(LoopFullUnrolling, node)
    >>> print(pm.dump(backend.Python, node))
    j = 1
    i += j
    j = 2
    i += j
    j = 3
    i += j

    >>> node = ast.parse('for j in (a,b): i += j')
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(LoopFullUnrolling, node)
    >>> print(pm.dump(backend.Python, node))
    j = a
    i += j
    j = b
    i += j

    >>> node = ast.parse('for j in {1}: i += j')
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(LoopFullUnrolling, node)
    >>> print(pm.dump(backend.Python, node))
    j = 1
    i += j

    >>> node = ast.parse('for j in builtins.enumerate("1"): j')
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(LoopFullUnrolling, node)
    >>> print(pm.dump(backend.Python, node))
    j = (0, '1')
    j
    '''

    MAX_NODE_COUNT = 65536

    def visit_For(self, node):
        # first unroll children if needed or possible
        self.generic_visit(node)

        # a break or continue in the loop prevents unrolling too
        has_break = any(self.gather(HasBreak, n)
                        for n in node.body)
        has_cont = any(self.gather(HasContinue, n)
                       for n in node.body)

        if has_break or has_cont:
            return node

        # do not unroll too much to prevent code growth
        node_count = self.gather(NodeCount, node)

        def unroll(elt, body):
            return [ast.Assign([deepcopy(node.target)], elt)] + body

        def dc(body, i, n):
            if i == n - 1:
                return body
            else:
                return deepcopy(body)

        def getrange(n):
            return getattr(getattr(n, 'func', None), 'attr', None)

        if isinstance(node.iter, (ast.Tuple, ast.List)):
            elts_count = len(node.iter.elts)
            total_count = node_count * elts_count
            issmall = total_count < LoopFullUnrolling.MAX_NODE_COUNT
            if issmall:
                self.update = True
                return sum([unroll(elt, dc(node.body, i, elts_count))
                            for i, elt in enumerate(node.iter.elts)], [])

        try:
            values = list(eval(astunparse.unparse(node.iter), {'builtins': __import__('builtins'), 'self': self.obj}))
        except Exception as e:
            return node

        values_count = len(values)
        total_count = node_count * values_count
        print("total count", total_count)
        issmall = total_count < LoopFullUnrolling.MAX_NODE_COUNT
        if issmall:
            try:
                new_node = sum([unroll(to_ast(elt),
                                       dc(node.body, i, values_count))
                                for i, elt in enumerate(values)], [])
                self.update = True
                return new_node
            except Exception:
                pass
            try:
                new_node = sum([
                    unroll(ast.Subscript(value = node.iter, slice = ast.Index(value=to_ast(i)), ctx = ast.Load()), dc(node.body, i, values_count))
                    for i, elt in enumerate(values)], [])
                self.update = True
                return new_node
            except Exception as e:
                pass
        return node


class UnrollSequential(Transformation):
    def visit_Assign(self, node):
        self.generic_visit(node)
        if not isinstance(node.value, ast.Call):
            return node
        if hasattr(node.value, '_func_inst') and isinstance(node.value._func_inst, torch.nn.Sequential):
            func = node.value.func
            args = node.value.args
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                raise NotImplementedError
            target_name =target.id
            new_nodes = []
            if len(node.value._func_inst) > 0:
                for i in range(len(node.value._func_inst)):
                    call_node = ast.Call(
                        func=ast.Subscript(value=deepcopy(func), slice=ast.Index(value=ast.Constant(value=i, kind=None)), ctx=ast.Load()),
                        args=args,
                        keywords=[]
                    )
                    call_node._func_inst = node.value._func_inst[i]
                    call_node.func_node = node.value.func_nodes[i]
                    call_node.is_udf = node.value.is_udf[i]

                    new_nodes.append(ast.Assign(
                        targets=[ast.Name(id=target_name, ctx=ast.Store(), annotation=None, type_comment=None)],
                        value=call_node))
                    args = [ast.Name(id=target_name, ctx=ast.Load(), annotation=None, type_comment=None)]
            else:
                new_nodes.append(ast.Assign(
                    targets=[ast.Name(id=target_name, ctx=ast.Store(), annotation=None, type_comment=None)],
                    value=args[0]))
            return new_nodes
        return node


'''
input:
a = (x, y, z)
a[0] ...
a[1] ...
a[2] ...

output:
a_0 = x
a_1 = y
a_2 = z
a_0 ...
a_1 ...
a_2 ...

'''

def is_const_int(node):
    try:
        x = eval(astunparse.unparse(node))
        if isinstance(x, int):
            return True
    except Exception as e:
        return False


class StaticTuple(FunctionAnalysis):
    def __init__(self):
        self.result = set()
        self.blacklist = set()
        super().__init__()
    
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Tuple) and len(node.targets) == 1 and node.targets[0].id not in self.blacklist:
            self.result.add(node.targets[0].id)
            return
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Index) and is_const_int(node.slice.value):
            return
        self.generic_visit(node)

    def visit_Name(self, node):
        self.blacklist.add(node.id)
        if node.id in self.result:
            self.result.remove(node.id)


class UnrollTuple(Transformation):
    def __init__(self):
        super().__init__(StaticTuple)
    
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Tuple) and len(node.targets) == 1 and node.targets[0].id in self.static_tuple:
            new_nodes = []
            name_prefix = node.targets[0].id
            for i, elt in enumerate(node.value.elts):
                new_nodes.append(
                    ast.Assign(
                        targets=[ast.Name(id=f"_{name_prefix}_{i}", ctx=ast.Store(), annotation=None, type_comment=None)],
                        value=elt
                    )
                )
            return new_nodes
        return self.generic_visit(node)
    
    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Index) and is_const_int(node.slice) and node.value.id in self.static_tuple:
            index_str = eval(astunparse.unparse(node.slice))
            return ast.Name(id=f"_{node.value.id}_{index_str}", ctx=node.ctx, annotation=None, type_comment=None)
        return node
