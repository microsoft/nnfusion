import gast

from . import quoting
from . import annotations as anno


def is_constant_num(node):
    return isinstance(node, gast.Constant) and isinstance(node.value, (int, float, complex))


def is_constant_str(node):
    return isinstance(node, gast.Constant) and isinstance(node.value, str)


def is_basic_node(node):
    return isinstance(node, (gast.Constant, gast.Name))