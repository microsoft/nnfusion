from ast_analyzer.grad import ast_utils
import astunparse
from . import annotations as anno
from . import cfg
import gast
from . import transformers
from ast import AST, iter_fields
import json

class ResolveCalls(gast.NodeVisitor):
    """Annotate Call nodes with the function being called."""

    def __init__(self, func):
        self.func = func

    def visit_FunctionDef(self, node):
        for field, value in iter_fields(node):
            if field == 'returns':
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        self.visit(item)
            elif isinstance(value, AST):
                self.visit(value)
        anno.setanno(node, 'func', self.func)

    def visit_Call(self, node):
        self.generic_visit(node)
        anno.setanno(node, 'func', node._func_inst)

    def visit_arguments(self, node):
        # ignore type annotation
        return


def mark_shape(node, type_dict):
    for n, t in type_dict.items():
        if anno.hasanno(n, 'type'):
            anno.delanno(n, 'type')
        anno.setanno(n, 'type', t)


def resolve_calls(node, func):
    """Put the function instance into annotation.
    Different from tangent because the function instance has been infered in "shape_inference"
    Args:
        node: An AST node
        func: The function whose calls are being resolved.
    """
    ResolveCalls(func).visit(node)


def get_anno_line(node):
    if not isinstance(node, gast.Expr) or not isinstance(node.value, gast.Constant):
        return None
    value = node.value.value
    if not isinstance(value, str):
        return None
    if not value.startswith('ANNO'):
        return None
    return value


class ResolveAnno(gast.NodeTransformer):
    def anno_FunctionDef(self, node):
        hint = get_anno_line(node.body[0])
        if hint is not None and hint.startswith("ANNO_FUNC:"):
            anno.setanno(node, 'hint', json.loads(hint[11:]))
            node.body = node.body[1:]

    def visit(self, node):
        if isinstance(node, gast.FunctionDef):
            self.anno_FunctionDef(node)
        for field, value in iter_fields(node):
            if isinstance(value, list):
                last_hint = None
                if field in ['body', 'orelse', 'finalbody']:
                    for item in value:
                        if isinstance(item, AST):
                            if last_hint is not None:
                                anno.setanno(item, 'hint', last_hint)
                                last_hint = None
                            last_hint = get_anno_line(item)
                            is_hint_stmt = False
                            if last_hint is not None:
                                if not last_hint.startswith('ANNO:'):
                                    last_hint = None
                                else:
                                    last_hint = json.loads(last_hint[6:])
                                    is_hint_stmt = True
                    new_body = []
                    for item in value:
                        if isinstance(item, AST):
                            hint = get_anno_line(item)
                            if hint is None or not hint.startswith('ANNO:'):
                                new_body.append(item)
                        else:
                            new_body.append(item)
                    setattr(node, field, new_body)
                for item in value:
                    if isinstance(item, AST):
                        self.visit(item)
            elif isinstance(value, AST):
                self.visit(value)
        return node


def resolve_anno(node):
    ResolveAnno().visit(node)


def unused(node, include_arg=False):
    """Find unused definitions that can be remove.

    This runs reaching definitions analysis followed by a walk over the AST to
    find all variable definitions that are not used later on.

    Args:
      node: The AST of e.g. a function body to find unused variable definitions.

    Returns:
      unused: After visiting all the nodes, this attribute contanis a set of
          definitions in the form of `(variable_name, node)` pairs which are
          unused in this AST.
    """
    cfg.backward(node, cfg.BackwardActive())
    unused = set()
    for sub in gast.walk(node):
        if isinstance(sub, gast.Assign):
            defs = ast_utils.get_updated(sub)
            active_in = anno.getanno(sub, 'bwd_active_in')
            used = False
            for d in defs:
                if d in active_in:
                    used = True
            if not used:
                unused.add(sub)
        if isinstance(sub, gast.arguments) and include_arg:
            active_in = anno.getanno(sub, 'bwd_active_in')
            for arg in sub.args:
                if arg.id not in active_in:
                    unused.add(arg)
    return unused


class ZeroFolding(transformers.TreeTransformer):
    def __init__(self):
        super(ZeroFolding, self).__init__()

    def visit_Assign(self, node):
        if isinstance(node.value, gast.BinOp):
            left = node.value.left
            right = node.value.right
            if isinstance(left, gast.Name) and left.id in anno.getanno(node, 'zero_tensor_in'):
                node.value = right
            elif isinstance(right, gast.Name) and right.id in anno.getanno(node, 'zero_tensor_in'):
                node.value = left

        return node


def zero_fold(node):
    cfg.forward(node, cfg.ZeroTensor())
    ZeroFolding().visit(node)


class GatherDefUse(gast.NodeVisitor):
    def __init__(self) -> None:
        super(GatherDefUse, self).__init__()
        self.result_def = {}
        self.result_use = {}

    def visit(self, node):
        self.generic_visit(node)
        result_def, result_use = cfg.get_def_use(node)
        for _, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        result_def.update(self.result_def[item])
                        result_use.update(self.result_use[item])
            elif isinstance(value, AST):
                result_def.update(self.result_def[value])
                result_use.update(self.result_use[value])
        self.result_def[node] = result_def
        self.result_use[node] = result_use


def block_live(node):
    cfg.backward(node, cfg.BackwardActive())
    gather_def_use = GatherDefUse()
    gather_def_use.visit(node)

    ret = {}
    for sub in gast.walk(node):
        if isinstance(sub, (gast.For, gast.If)):
            active_in = anno.getanno(sub, 'bwd_active_in')
            active_out = anno.getanno(sub, 'bwd_active_out')
            defs = gather_def_use.result_def[sub]
            blockLives = defs.intersection(active_out)
            ret[sub] = blockLives

    return ret


# class GenASTInfo(TwoNodeVisitor):
#     def __init__(self, gethered_defs):
#         self.ast_info = {}
#         self.gathered_defs = gethered_defs

#     def register_node(self, node1, node2):
#         active_out = anno.getanno(node2, 'bwd_active_out')
#         defs = self.gathered_defs[node2]
#         print("[gen_info]", node1, active_out, defs)
#         blockLives = defs.intersection(active_out)
#         self.ast_info[node1] = blockLives

#     def visit_If(self, node1, node2):
#         self.generic_visit(node1, node2)
#         self.register_node(node1, node2)
        
#     def visit_For(self, node1, node2):
#         self.generic_visit(node1, node2)
#         self.register_node(node1, node2)

def get_last_nodes(node):
    if isinstance(node, gast.While):
        return [node.test]
    if not isinstance(node, gast.If):
        return [node]
    if len(node.body) == 0:
        return get_last_nodes(node.orelse[-1])
    elif len(node.orelse) == 0:
        return get_last_nodes(node.body[-1])
    else:
        return get_last_nodes(node.body[-1]) + get_last_nodes(node.orelse[-1])


def get_arg_ret(stmts, cfg_nodes, white_list): # white_list is a set
    if len(stmts) == 0:
        return set(), set()
    assert not isinstance(stmts[-1], gast.Return)
    if isinstance(stmts[0], gast.FunctionDef):
        print("[warning] get_ret_arg for functiondef is not implemented")
        return set(), set()
    inner_cfg_nodes = set()
    for stmt in stmts:
        for sub_stmt in gast.walk(stmt):
            if sub_stmt in cfg_nodes:
                inner_cfg_nodes.add(cfg_nodes[sub_stmt])
    if isinstance(stmts[0], (gast.If, gast.While)):
        args = anno.getanno(stmts[0].test, 'bwd_active_out')
    else:
        args = anno.getanno(stmts[0], 'bwd_active_out')

    last_nodes = get_last_nodes(stmts[-1])

    rets = set()
    
    for last_node in last_nodes:
        cfg_node = cfg_nodes[last_node]
        for prev in cfg_node.next:
            if prev not in inner_cfg_nodes:
                if prev.value is not None:
                    live_out = anno.getanno(prev.value, 'bwd_active_out')
                    rets.update(live_out)
            else:
                print("[skip]", astunparse.unparse(prev.value))

    defuse = GatherDefUse()
    defs = set()
    uses = set()
    for stmt in stmts:
        defuse.visit(stmt)
        defs.update(defuse.result_def[stmt])
        uses.update(defuse.result_use[stmt])
    # print("[args]", args)
    # print("[rets]", rets)
    # print("[defs]", defs)
    # print("[uses]", uses)
    args = args.intersection(uses) - white_list
    rets = rets.intersection(defs) - white_list
    # print("[args-real]", args)
    # print("[rets-real]", rets)
    return args, rets


# def get_live_large_node(stmts, args, rets):
#     raise NotImplementedError()
#     copied_stmts = copy.deepcopy(stmts)
#     cfg.backward_block(copied_stmts, cfg.BackwardActive(), args, rets)
#     gathered_defs = {}
#     for c_stmt in copied_stmts:
#         g = GatherDefUse()
#         g.visit(c_stmt)
#         gathered_defs.update(g.result_def)
#     gen = GenASTInfo(gathered_defs)
#     for stmt, c_stmt in zip(stmts, copied_stmts):
#         gen.visit(stmt, c_stmt)
#     print("[ast_info]", gen.ast_info)
#     return gen.ast_info

def get_live_large_node(stmts, cfg_nodes, white_list):
    ast_info = {}
    for stmt in stmts:
        for sub_stmt in gast.walk(stmt):
            if isinstance(sub_stmt, gast.For):
                args, rets = get_arg_ret(sub_stmt.body, cfg_nodes, white_list)
                ast_info[sub_stmt] = rets - white_list
                # print("[ast_info]", ast_info[sub_stmt], args, rets)
            elif isinstance(sub_stmt, gast.While):
                args, rets = get_arg_ret(sub_stmt.body, cfg_nodes, white_list)
                ast_info[sub_stmt] = rets - white_list
            elif isinstance(sub_stmt, gast.If):
                args_body, rets_body = get_arg_ret(sub_stmt.body, cfg_nodes, white_list)
                args_orelse, rets_orelse = get_arg_ret(sub_stmt.orelse, cfg_nodes, white_list)
                rets = rets_body.union(rets_orelse) - white_list
                ast_info[sub_stmt] = rets
                # print("[ast_info]", ast_info[sub_stmt], args_body, args_orelse, rets)

    return ast_info
