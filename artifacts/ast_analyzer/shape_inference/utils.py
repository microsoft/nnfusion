import ast
import gast
import numbers
import os
import inspect


def print_warning(msg):
    print("\x1b[33m[WARNING] " + msg + "\x1b[39m")

# ============================== Display utils =================================


def intercalate(strings, sep):
    if strings == []:
        return ""
    return "".join([s + sep for s in strings[:-1]]) + strings[-1]


def expr_to_str(node):
    if isinstance(node, gast.BoolOp):
        return intercalate([expr_to_str(e) for e in node.values],
                           " " + boolop_to_str(node.op) + " ")
    if isinstance(node, gast.BinOp):
        return "{} {} {}".format(expr_to_str(node.left),
                                 operator_to_str(node.op), expr_to_str(node.right))
    if isinstance(node, gast.UnaryOp):
        return "{}{}".format(unaryop_to_str(node.op), expr_to_str(node.operand))
    if isinstance(node, gast.Call):
        args = [expr_to_str(arg) for arg in node.args] + \
            ["{}={}".format(kwarg.arg, expr_to_str(kwarg.value))
             for kwarg in node.keywords]
        return "{}({})".format(expr_to_str(node.func), intercalate(args, ", "))
    if isinstance(node, gast.Constant) and isinstance(node.value, numbers.Number):
        return str(node.value)
    if isinstance(node, gast.Constant) and isinstance(node.value, str):
        if len(node.value) < 20:
            return "\'" + node.value + "\'"
        return "\"...\""  # sometimes it is too long
    if isinstance(node, gast.Constant):  # assume is NameConstant
        return str(node.value)
    if isinstance(node, gast.Attribute):
        return "{}.{}".format(expr_to_str(node.value), node.attr)
    if isinstance(node, gast.Subscript):
        return "{}[{}]".format(expr_to_str(node.value), slice_to_str(node.slice))
    if isinstance(node, gast.Name):
        return node.id
    if isinstance(node, gast.List):
        return "[" + intercalate([expr_to_str(e) for e in node.elts], ", ") + "]"
    if isinstance(node, gast.Tuple):
        return "(" + intercalate([expr_to_str(e) for e in node.elts], ", ") + ")"
    return ""


def boolop_to_str(node):
    if isinstance(node, gast.And):
        return "and"
    if isinstance(node, gast.Or):
        return "or"


def operator_to_str(node):
    if isinstance(node, gast.Add):
        return "+"
    if isinstance(node, gast.Sub):
        return "-"
    if isinstance(node, gast.Mult):
        return "*"
    if isinstance(node, gast.Div):
        return "/"
    if isinstance(node, gast.FloorDiv):
        return "//"


def unaryop_to_str(node):
    if isinstance(node, gast.Invert):
        return "!"  # 合ってる?
    if isinstance(node, gast.Not):
        return "not"
    if isinstance(node, gast.UAdd):
        return "+"
    if isinstance(node, gast.USub):
        return "-"


def slice_to_str(node):
    if isinstance(node, gast.Slice):
        ret = ""
        if node.lower:
            ret += expr_to_str(node.lower)
        ret += ":"
        if node.upper:
            ret += expr_to_str(node.upper)
        ret += ":"
        if node.step:
            ret += expr_to_str(node.step)
        return ret
    if isinstance(node, gast.ExtSlice):
        return intercalate([slice_to_str(s) for s in node.dims], ", ")
    if isinstance(node, gast.Index):
        return expr_to_str(node.value)


def is_expr(node):
    expr_types = (gast.BoolOp, gast.BinOp, gast.UnaryOp, gast.Lambda, gast.IfExp, gast.Dict, gast.Set, gast.ListComp, gast.SetComp, gast.DictComp, gast.GeneratorExp, gast.Await, gast.Yield,
                  gast.YieldFrom, gast.Compare, gast.Call, gast.Repr, gast.Constant, gast.FormattedValue, gast.JoinedStr, gast.Attribute, gast.Subscript, gast.Starred, gast.Name, gast.List, gast.Tuple)
    return isinstance(node, expr_types)


def node_description(node):
    type_name = type(node).__name__
    lineno = " (line {})".format(
        node.lineno) if hasattr(node, 'lineno') else ""
    if isinstance(node, gast.FunctionDef):
        return "{} {}{}".format(type_name, node.name, lineno)
    if is_expr(node):
        return "{} {}{}".format(type_name, expr_to_str(node), lineno)
    return type_name

# ==============================================================================


def add_dict(dest, src):
    for k, v in src.items():
        dest[k] = v


def find(seq, pred):
    for elt in seq:
        if pred(elt):
            return elt


def set_attr_if_None(obj1, obj2, attr_name):
    if hasattr(obj1, attr_name) and getattr(obj1, attr_name) is None:
        setattr(obj1, attr_name, getattr(obj2, attr_name))
        return
    if hasattr(obj2, attr_name) and getattr(obj2, attr_name) is None:
        setattr(obj2, attr_name, getattr(obj1, attr_name))
        return


def all_same(l):
    return all([e == l[0] for e in l])


def foldl(fn, acc, l):
    for x in l:
        acc = fn(acc, x)
    return acc


def clip_head(s: 'str'):
    splitted = s.split('\n')

    # remove comments
    comment_count = 0
    indent_targets = []
    for sp in splitted:
        if '"""' in sp or "'''" in sp:
            comment_count += sp.count('"""') + sp.count("'''")
        else:
            if comment_count % 2 == 0:
                indent_targets.append(sp)

    hs = os.path.commonprefix(list(filter(lambda x: x != '', indent_targets)))
    # print('hs',list(map(ord,hs)))
    ls = len(hs)
    strs = map(lambda x: x[ls:], splitted)
    return '\n'.join(strs)


def get_ast(func):
    if hasattr(func, '__ast__'):
        return func.__ast__
    code = clip_head(inspect.getsource(func))
    node = gast.ast_to_gast(ast.parse(code))
    return node
