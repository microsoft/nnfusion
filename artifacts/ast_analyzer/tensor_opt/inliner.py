import astunparse
import gast
from .utils import copy_anno, is_call_stmt
import copy
import torch
from ast_analyzer.grad import annotations as anno

class InlineTransformer(gast.NodeTransformer):
    def __init__(self, mapping):
        super(InlineTransformer, self).__init__()
        self.mapping = mapping
        self.white_list = set(['torch', 'math'])
    
    def visit_Name(self, node):
        if node.id in self.mapping:
            new_node = copy.deepcopy(self.mapping[node.id])
            new_node.ctx = copy.deepcopy(node.ctx)
            return new_node
        elif node.id in self.white_list:
            return copy.deepcopy(node)
        else:
            new_node = copy.deepcopy(node)
            new_node.id = "_tmp__" + node.id
            return new_node

    
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        return node


def inline_body(node, map_list):
    # print("[inline_body]")
    # print(astunparse.unparse(node))
    assert(isinstance(node, gast.FunctionDef))
    assert(len(node.args.args) == len(map_list))
    mapping = {}
    for arg, mp in zip(node.args.args, map_list):
        mapping[arg.id] = mp

    new_node = copy.deepcopy(node)
    InlineTransformer(mapping).visit(new_node)

    # print("[result]")
    # print(astunparse.unparse(new_node))

    return new_node.body


def inline(stmt, namer):
    # print("[stmt]")
    # print(astunparse.unparse(stmt))
    # something like visit_Call
    unrolled_stmts = []
    map_list = []
    if not isinstance(stmt.value.func, gast.Name):
        if anno.hasanno(stmt.value, 'func'):
            func_inst = anno.getanno(stmt.value, 'func')
        elif hasattr(stmt.value, '_func_inst'):
            func_inst = stmt.value._func_inst
        else:
            raise ValueError("Cannot find func inst for stmt: ", astunparse.unparse(stmt))
        if isinstance(func_inst, torch.nn.Module):
            map_list.append(stmt.value.func)
        else:
            map_list.append(stmt.value.func.value)

    for arg in stmt.value.args:
        new_name = namer.next()
        arg_assign = gast.Assign(
            targets = [gast.Name(id = new_name, ctx = gast.Store(), annotation = None, type_comment = None)],
            value = arg # should I use deepcopy?
        )
        copy_anno(arg_assign.targets[0], arg, ['type', 'may_push'])
        unrolled_stmts.append(arg_assign)
        load_node = gast.Name(id = new_name, ctx = gast.Load(), annotation = None, type_comment = None)
        copy_anno(load_node, arg_assign.targets[0], ['type', 'may_push'])
        map_list.append(load_node)

    inlined_stmts = inline_body(stmt.value.func_node, map_list)
    last_stmt = inlined_stmts[-1]
    inlined_stmts = inlined_stmts[:-1]
    assert(isinstance(last_stmt, gast.Return))

    for inlined_stmt in inlined_stmts:
        if is_call_stmt(inlined_stmt) and inlined_stmt.value.is_udf:
            stmts = inline(inlined_stmt, namer)
            unrolled_stmts.extend(stmts)
        else:
            unrolled_stmts.append(inlined_stmt)

    new_assign_stmt = copy.deepcopy(stmt)
    new_assign_stmt.value = last_stmt.value
    unrolled_stmts.append(new_assign_stmt)

    return unrolled_stmts
