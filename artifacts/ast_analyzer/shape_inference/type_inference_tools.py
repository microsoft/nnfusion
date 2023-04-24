import ast
import gast
import inspect
import numpy as np
import sys
import typing

from ast_analyzer.shape_inference import types
from ast_analyzer.shape_inference.type_inference import InferenceEngine
from ast_analyzer.shape_inference.utils import node_description, is_expr, clip_head
from ast_analyzer.shape_inference.types import *
from ast_analyzer.grad import annotations as anno


class IDAssignor(gast.NodeVisitor):
    def __init__(self):
        self.counter = 0
        self.node2id = {}

    def visit(self, node):
        self.node2id[node] = self.counter
        self.counter += 1
        return super().visit(node)

    def run(self, node, subroutine_node):
        self.visit(node)

        for ns in subroutine_node.values():
            for n in ns:
                self.visit(n)

        return self.node2id


def generate_node2id(tree, subroutine_node):
    a = IDAssignor()
    node2id = a.run(tree, subroutine_node)
    return node2id


def generate_id2node(node2id):
    id2node = {}
    for n, i in node2id.items():
        id2node[i] = n

    return id2node


def generate_node2type(tree, args, is_debug=False, module=None, type_hints={}, func_inst=None):
    reset_state()
    tc = InferenceEngine(is_debug=is_debug, module=module, func_inst=func_inst)
    if isinstance(tree, gast.Module):
        func_body = tree.body[0]  # XXX: only checks first function
    elif isinstance(tree, gast.FunctionDef):
        func_body = tree
    else:
        raise NotImplementedError
    func_body._func_inst = func_inst
    try:
        node2type = tc.infer_function_value_args(
            func_body, args, type_hints=type_hints)
        return node2type, tc.subroutine_node
    except Exception as e:
        tc.dump_tyenv()
        raise e


def generate_id2type(node2type, node2id):
    id2type = {}
    for n, t in node2type.items():
        if n not in node2id.keys():
            continue  # user-defined modules in nn.Sequential
        id2type[node2id[n]] = t

    return id2type


def generate_assertion(type_table_name, id2type, id2node, ofile=None):
    for i, t in sorted(id2type.items()):
        node = id2node[i]
        if not is_expr(node):
            if not isinstance(node, gast.FunctionDef):
                continue
            output = "        # === function {} ===".format(node.name)
        else:
            comment = "\t# " + node_description(node)
            output = "        self.assertEqual(str({}[{}]), \"{}\"){}".format(
                type_table_name, i, t, comment)
        if ofile is None:
            print(output)
        else:
            ofile.write(output + '\n')


def print_inference_results(node2type):
    for n, t in node2type.items():
        # import astunparse
        # print("===============")
        # print(i, type(id2node[i]))
        # print(astunparse.unparse(node))
        print("{} : \x1b[36m{}\x1b[39m".format(node_description(n), t))
        # print("===============")


# For testing
def generate_id2type_from_forward(model, args, is_debug=False):
    code = clip_head(inspect.getsource(model.forward))
    tree = gast.ast_to_gast(ast.parse(code))
    module = sys.modules[model.forward.__module__]
    node2type, subroutine_node = generate_node2type(
        tree, (model,) + args, is_debug=is_debug, module=module,
        type_hints=typing.get_type_hints(model.forward),
        func_inst=model.forward)
    node2id = generate_node2id(tree, subroutine_node)
    id2type = generate_id2type(node2type, node2id)
    return id2type


def type_inference_model(model, forward_args, is_debug=False):
    if model.__ast__ is not None:
        node = model.__ast__
    else:
        code = clip_head(inspect.getsource(model.forward))
        node = gast.ast_to_gast(ast.parse(code))
    # node = Canonicalizer().visit(node)
    module = sys.modules[model.forward.__module__]
    node2type, subroutine_node = generate_node2type(
        node, (model,) + forward_args, is_debug=is_debug, module=module,
        type_hints=typing.get_type_hints(model.forward),
        func_inst=model.forward)
    return node2type


def type_inference_func(loss_func, forward_args, is_debug=False):
    if loss_func.__ast__ is not None:
        node = loss_func.__ast__
    else:
        code = clip_head(inspect.getsource(loss_func))
        node = gast.ast_to_gast(ast.parse(code))
    # node = Canonicalizer().visit(node)
    module = sys.modules[loss_func.__module__]
    node2type, subroutine_node = generate_node2type(
        node, forward_args, is_debug=is_debug, module=module,
        type_hints=typing.get_type_hints(loss_func),
        func_inst=loss_func)
    return node2type


def type_inference_fwdbwd(fwdbwd, forward_args, model, is_debug=False):
    node_fwd = fwdbwd.body[0]
    module = sys.modules[model.forward.__module__]
    node2type_fwd, subroutine_node_fwd = generate_node2type(
        node_fwd, forward_args, is_debug=is_debug, module=module, type_hints=typing.get_type_hints(model.forward))

    dummy_result = generate_dummy_value(node2type_fwd[node_fwd].retty)

    type_hints = {} # get type_hints from retty

    node_bwd = fwdbwd.body[1]
    if node_bwd.args.args[0].id == 'self':
        attrs = tuple(getattr(model, anno.getanno(node, 'attr_name'))
                      for node in node_bwd.args.args if anno.hasanno(node, 'attr_name'))
        # the shape of d_out is the same as out
        dummy_input_bwd = (forward_args[0],) + dummy_result + attrs
        for i, ty in enumerate(node2type_fwd[node_fwd].retty):
            type_hints[node_bwd.args.args[i+1].id] = ty
    else:
        dummy_input_bwd = dummy_result  # the shape of d_out is the same as out
        for i, ty in enumerate(node2type_fwd[node_fwd].retty):
            type_hints[node_bwd.args.args[i].id] = ty
    
    node2type_bwd, subroutine_node_bwd = generate_node2type(
        node_bwd, dummy_input_bwd, is_debug=is_debug, module=module, type_hints=type_hints)

    return node2type_fwd, node2type_bwd


def fetch_ctx_type(node):
    autograd_node = node.body[0].body[0].value.func_node
    for sub_node in gast.walk(autograd_node):
        if isinstance(sub_node, gast.Call) and hasattr(sub_node, 'ctx_types'):
            return getattr(sub_node, 'ctx_types')
    raise ValueError("no ctx node")


def type_inference_fwdbwd_function(model, forward_args, node, func_name, is_debug=False):
    model.__ast__ = node
    module = sys.modules['__main__']
    for sub_node in gast.walk(node):
        if isinstance(sub_node, gast.Attribute) and sub_node.attr == 'apply':
            sub_node.attr = 'forward'
    node2type_fwd, subroutine_node_fwd = generate_node2type(
        node, forward_args, is_debug=is_debug, module=module, type_hints=typing.get_type_hints(model.forward))

    func_inst = getattr(model, func_name)
    node_fwd = node.body[0].body[0].value.func_node
    ctx_types = fetch_ctx_type(node)
    bwd_code = clip_head(inspect.getsource(func_inst.backward))
    node_bwd = gast.ast_to_gast(ast.parse(bwd_code)).body[0]

    for sub_node in gast.walk(node_bwd):
        if isinstance(sub_node, gast.Attribute) and sub_node.attr == 'saved_tensors':
            sub_node.type_anno = TyTuple(ctx_types[1:])

    dummy_result = generate_dummy_value(node2type_fwd[node_fwd].retty)
    if not isinstance(dummy_result, tuple):
        dummy_result = (dummy_result, )
    assert(len(node_bwd.args.args) == 2)
    type_hints = {node_bwd.args.args[1].id: node2type_fwd[node_fwd].retty}
    node2type_bwd, subroutine_node_bwd = generate_node2type(
        node_bwd, (ctx_types[0],) + dummy_result, is_debug=is_debug, module=module, type_hints=type_hints)

    return node_fwd, node_bwd, node2type_fwd, node2type_bwd, func_inst


def reset_state():
    np.random.seed(42)
    types.var_counter = 0
