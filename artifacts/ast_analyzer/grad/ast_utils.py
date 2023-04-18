"""Utilities to manipulate the AST and its annotations."""
from __future__ import absolute_import
import copy

import gast

from . import annotations as anno
from . import quoting
from . import template

from ast_analyzer.shape_inference.shape_elem import unwrap_shape
from ast_analyzer.shape_inference.types import *


def get_name(node):
    """Get the name of a variable.

    Args:
        node: A `Name`, `Subscript` or `Attribute` node.

    Returns:
        The name of the variable e.g. `'x'` for `x`, `x.i` and `x[i]`.
    """
    if isinstance(node, gast.Name):
        return node.id
    elif isinstance(node, (gast.Subscript, gast.Attribute)):
        return get_name(node.value)
    else:
        raise TypeError


def _get_target(node):
    if isinstance(node, (gast.Name, gast.Subscript, gast.Attribute)):
        return set([get_name(node)])
    elif isinstance(node, (gast.Tuple, gast.List)):
        return set.union(*(_get_target(target)
                           for target in node.elts))
    elif isinstance(node, (gast.Constant,)):
        return set()
    else:
        print(gast.dump(node))
        raise ValueError


def get_updated(node):
    """Return the variable names created or mutated by this statement.

    This function considers assign statements, augmented assign statements, and
    the targets of for loops, as well as function arguments.

    For example, `x[0] = 2` will return `x`, `x, y = 3, 4` will return `x` and
    `y`, `for i in range(x)` will return `i`, etc.

    Args:
        node: An AST node

    Returns:
        A set of variable names (strings) of all the variables created or mutated.
    """
    if isinstance(node, gast.Assign):
        return set.union(*(_get_target(target)
                           for target in node.targets))
    elif isinstance(node, (gast.For, gast.AugAssign)):
        return _get_target(node.target)
    elif isinstance(node, gast.arguments):
        targets = set(arg.id for arg in node.args + node.kwonlyargs)
        if node.vararg:
            targets.add(node.vararg.id)
        if node.kwarg:
            targets.add(node.kwarg.id)
        return targets
    else:
        return set()


def copy_node(node):
    """Copy a node but keep its annotations intact."""
    if not isinstance(node, gast.AST):
        return [copy_node(n) for n in node]

    # The shape inference result cannot be deepcopied

    grad_anno = getattr(node, anno.ANNOTATION_FIELD, anno.Annotation())
    if hasattr(node, anno.ANNOTATION_FIELD):
        delattr(node, anno.ANNOTATION_FIELD)

    new_node = copy.deepcopy(node)

    setattr(node, anno.ANNOTATION_FIELD, grad_anno)
    setattr(new_node, anno.ANNOTATION_FIELD, copy.copy(grad_anno))
    return new_node


class ArgAppend(gast.NodeTransformer):
    """Append arguments to a function definition."""

    def __init__(self, node_list):
        self.visited = False
        self.node_list = node_list

    def visit_FunctionDef(self, node):
        if not self.visited:
            node.args.args.extend(self.node_list)
            self.visited = True
        return node


def append_args(node, node_list):
    if not isinstance(node_list, list):
        raise TypeError('Please pass in a list')
    if all([isinstance(n, str) for n in node_list]):
        node_list = [quoting.quote(n) for n in node_list]
    return ArgAppend(node_list).visit(node)


def is_insert_grad_of_statement(node):
    """Check whether a context manager calls `insert_grad_of`.

    Args:
        node: The context manager node.

    Returns:
        Whether or not this node contains `insert_grad_of` calls.

    Raises:
        ValueError: If the `insert_grad_of` calls are mixed with other calls.
    """
    tangent_calls = [anno.getanno(item.context_expr, 'func', None)
                     is utils.insert_grad_of for item in node.items]
    if all(tangent_calls):
        return True
    elif any(tangent_calls):
        raise ValueError
    else:
        return False


def is_attr_of(node, inst):
    return isinstance(node, gast.Attribute) and isinstance(node.value, gast.Name) and node.value.id == inst


class LoopLevel:
    def __init__(self):
        self.levels_fwd = []
        self.levels_bwd = []
        self.bounds = []
        self.is_fixed = True

    def depth(self):
        return len(self.bounds)

    def get_forward(self, d):
        return self.levels_fwd[d]

    def get_backward(self, d):
        return self.levels_bwd[d]

    def add_level(self, target_fwd_node, target_bwd_node, bound_node):
        self.levels_fwd.append(target_fwd_node)
        self.levels_bwd.append(target_bwd_node)
        self.bounds.append(bound_node)

    def del_level(self):
        self.levels_fwd.pop()
        self.levels_bwd.pop()
        self.bounds.pop()

    def tensor_of_type(self, ty, init="empty", device=None):
        if isinstance(ty, TyNum):
            shape = self.bounds
            if ty.is_int():
                ty_torch = quoting.quote('torch.int64')
            else:
                raise NotImplementedError
        elif isinstance(ty, TyTensor):
            if not ty.is_fixed_shape():
                raise NotImplementedError
            else:
                ts_shape = list(unwrap_shape(ty.shape))
                ts_shape = [quoting.quote(str(s)) for s in ts_shape]
                shape = self.bounds + ts_shape
            ty_torch = quoting.quote(np_dtype_to_torch_string(ty.dtype))
        else:
            print(ty)
            raise NotImplementedError

        if len(shape) == 0:
            return template.replace(
                # use 'repr' to format cuda to "cuda" (with quotes)
                "torch.{}((), dtype=ty, device={})".format(init, repr(device)),
                ty=ty_torch
            )
        elif len(shape) == 1:
            return template.replace(
                "torch.{}(d1, dtype=ty, device={})".format(init, repr(device)),
                d1=shape[0],
                ty=ty_torch
            )
        elif len(shape) == 2:
            return template.replace(
                "torch.{}(d1, d2, dtype=ty, device={})".format(
                    init, repr(device)),
                d1=shape[0],
                d2=shape[1],
                ty=ty_torch
            )
        elif len(shape) == 3:
            return template.replace(
                "torch.{}(d1, d2, d3, dtype=ty, device={})".format(
                    init, repr(device)),
                d1=shape[0],
                d2=shape[1],
                d3=shape[2],
                ty=ty_torch
            )
        elif len(shape) == 4:
            return template.replace(
                "torch.{}(d1, d2, d3, d4, dtype=ty, device={})".format(
                    init, repr(device)),
                d1=shape[0],
                d2=shape[1],
                d3=shape[2],
                d4=shape[3],
                ty=ty_torch
            )
        else:
            raise NotImplementedError


def tensor_of_type(ty, init="empty", device=None):
    return LoopLevel().tensor_of_type(ty, init, device=device)


def generate_zero_ast(var, ty, device):
    ty = ty.deref()

    if isinstance(ty, TyNum):
        if ty.is_int():
            return gast.Constant(value=0, kind=None)
        elif ty.is_float():
            return gast.Constant(value=0.0, kind=None)

    if isinstance(ty, TyTensor):
        if ty.is_fixed_shape():
            return tensor_of_type(ty, "zeros", device=device).value
        if var is not None:
            return template.replace("torch.zeros_like(param, device=dev)", param=var, dev=device).value

    if isinstance(ty, TyTuple):
        if ty.is_fixed_len:
            elts = [generate_zero_ast(None, t, device) for t in ty.get_tys()]
            return gast.List(elts=elts, ctx=gast.Load())  # TODO: use tuple

    raise ValueError("generate_zero: type not understood: " +
                     str(ty) + "(" + str(type(ty)) + ")")
