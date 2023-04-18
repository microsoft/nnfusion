"""Transform AST into something similar to A-normal form.

This significantly simplifies certain procedures later on. The ANF
transformations guarantee the following:

All nested expressions on the right hand side of assignments are expanded and
reduced to the following:

    y = x
    y = f(x1, ..., xn)
    z = x + y
    y = -x
    y.i = x
    y = x.i
    y[i] = x
    y = x[i]
    z = x, y

Note that we do not allow tuple unpacking, because statements like `x[i], y =
f(x)` are difficult to process in this case. Hence, unpacking is made explicit.

The value of the return statement is reduced to either a single variable, or a
tuple of variables (nested tuples are expanded).

"""
from __future__ import absolute_import
import gast
import astunparse

from . import annotations as anno
from . import grammar
from . import naming
from . import quoting
from . import transformers
from . import cfg
from . import ast_utils


class ANF(transformers.TreeTransformer):
    """Transform a tree to an ANF-like form."""

    def __init__(self):
        super(ANF, self).__init__()
        # Whether the current statement in question must be trivialized
        self.trivializing = False
        # The original line that is transformed, which is kept as an annotation
        self.src = ''
        # TODO: remove attribute from attr2name when the attribute is modified
        self.attr2name = {}

    def mark(self, node):
        if not anno.hasanno(node, 'pre_anf') and self.src:
            anno.setanno(node, 'pre_anf', self.src)

    def trivialize(self, node):
        if isinstance(node, (gast.Name, type(None), gast.Constant)):
            return node
        is_attr_load = False
        if isinstance(node, gast.Attribute) and isinstance(node.value, gast.Name) and node.value.id == 'self' and isinstance(node.ctx, gast.Load):
            is_attr_load = True
            if node.attr in self.attr2name:
                load_node = gast.Name(annotation=None, id=self.attr2name[node.attr],
                                      ctx=gast.Load(), type_comment=None)
                if anno.hasanno(node, 'type'):
                    anno.setanno(load_node, 'type', anno.getanno(node, 'type'))
                return load_node

        name = self.namer.name(node)
        stmt = gast.Assign(
            targets=[gast.Name(annotation=None, id=name,
                               ctx=gast.Store(), type_comment=None)],
            value=None)
        self.mark(stmt)
        self.prepend(stmt)
        stmt.value = self.visit(node)
        load_node = gast.Name(annotation=None, id=name,
                              ctx=gast.Load(), type_comment=None)
        if is_attr_load:
            self.attr2name[node.attr] = name
            anno.setanno(stmt, 'attr_name', node.attr)
        if anno.hasanno(stmt.value, 'type'):
            anno.setanno(stmt.targets[0], 'type',
                         anno.getanno(stmt.value, 'type'))
            anno.setanno(load_node, 'type', anno.getanno(stmt.value, 'type'))
        return load_node

    def visit_Call(self, node):
        if self.trivializing:
            for i, arg in enumerate(node.args):
                node.args[i] = self.trivialize(arg)
            for keyword in node.keywords:
                keyword.value = self.trivialize(keyword.value)
        return node

    def visit_FunctionDef(self, node):
        self.namer = naming.Namer.build(node)
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if self.trivializing:
            node.left = self.trivialize(node.left)
            node.right = self.trivialize(node.right)
        return node

    def visit_UnaryOp(self, node):
        if self.trivializing:
            node.operand = self.trivialize(node.operand)
        return node

    def visit_Return(self, node):
        self.trivializing = True
        self.namer.target = node
        node.value = self.trivialize(node.value)
        self.trivializing = False
        self.namer.target = None
        return node

    def trivialize_slice(self, node):
        if isinstance(node, gast.Slice):
            name = self.namer.name(node)
            target = gast.Name(id=name, ctx=gast.Store(),
                               annotation=None, type_comment=None)
            stmt = gast.Assign(targets=[target], value=None)
            self.prepend(stmt)
            stmt.value = gast.Call(
                func=gast.Name(id='slice', ctx=gast.Load(),
                               annotation=None, type_comment=None),
                args=[
                    self.trivialize(arg) if arg else
                    gast.Name(id='None', ctx=gast.Load(),
                              annotation=None, type_comment=None)
                    for arg in [node.lower, node.upper,
                                node.step]],
                keywords=[])
            return gast.Name(id=name, ctx=gast.Load(), annotation=None, type_comment=None)
        elif isinstance(node, gast.ExtSlice):
            name = self.namer.name(node)
            target = gast.Name(id=name, ctx=gast.Store(),
                               annotation=None, type_comment=None)
            stmt = gast.Assign(targets=[target], value=None)
            self.prepend(stmt)
            dim_names = [self.trivialize_slice(s).id for s in node.dims]
            stmt.value = gast.Tuple(elts=[
                gast.Name(id=n, ctx=gast.Load(),
                          annotation=None, type_comment=None)
                for n in dim_names], ctx=gast.Load())
            return gast.Name(id=name, ctx=gast.Load(), annotation=None, type_comment=None)
        elif isinstance(node, gast.Index):
            return self.trivialize(node.value)
        else:
            raise ValueError(node)

    def visit_Subscript(self, node):
        if self.trivializing:
            node.value = self.trivialize(node.value)
            node.slice = gast.Index(value=self.trivialize_slice(node.slice))
        return node

    def visit_Tuple(self, node):
        if self.trivializing:
            node.elts = [self.trivialize(elt) for elt in node.elts]
        return node

    def visit_List(self, node):
        if self.trivializing:
            node.elts = [self.trivialize(elt) for elt in node.elts]
        return node

    def visit_AugAssign(self, node):
        self.src = quoting.unquote(node)
        self.trivializing = True
        self.namer.target = node.target
        right = self.trivialize(node.value)
        target = self.trivialize(node.target)
        left = gast.Name(id=target.id, ctx=gast.Load(),
                         annotation=None, type_comment=None)
        node = gast.Assign(targets=[target],
                           value=gast.BinOp(
            left=left, op=node.op, right=right))
        self.mark(node)
        node = self.generic_visit(node)
        self.namer.target = None
        self.trivializing = False
        return node

    def visit_Assign(self, node):
        self.src = quoting.unquote(node)
        self.mark(node)
        self.trivializing = True
        self.namer.target = node.targets[0]
        if isinstance(node.targets[0], (gast.Subscript, gast.Attribute)):
            node.value = self.trivialize(node.value)
            node.targets[0] = self.visit(node.targets[0])
        elif isinstance(node.targets[0], gast.Tuple):
            # TODO: a hack for a, b, c, d = torch.split
            node.value = self.visit(node.value)
            # name = self.namer.name(node.targets[0])
            # target = gast.Name(id=name, ctx=gast.Store(),
            #                    annotation=None, type_comment=None)
            # for i, elt in enumerate(node.targets[0].elts):
            #     stmt = gast.Assign(
            #         targets=[elt],
            #         value=gast.Subscript(
            #             value=gast.Name(id=name, ctx=gast.Load(),
            #                             annotation=None, type_comment=None),
            #             slice=gast.Index(value=gast.Constant(value=i, kind=None)),
            #             ctx=gast.Load()))
            #     self.mark(stmt)
            #     self.append(stmt)
            # node.targets[0] = target
        elif not isinstance(node.targets[0], gast.Name):
            raise ValueError('Cannot Assign to %s' % type(node.target))
        node = self.generic_visit(node)
        self.namer.target = None
        self.trivializing = False
        return node


class InvariantMotion(transformers.TreeTransformer):
    def __init__(self):
        super(InvariantMotion, self).__init__()
        self.in_loop = False

    def visit_For(self, node):
        self.in_loop = True
        ret = self.generic_visit(node)
        self.in_loop = False
        return ret        

    def visit_Assign(self, node):
        if not self.in_loop:
            return node
        invariants = anno.getanno(node, 'invariant_out')
        upds = ast_utils.get_updated(node)
        for id_ in upds:
            if id_ not in invariants:
                return node
        self.insert_top_last(node)
        return []


def anf(node):
    """Turn an AST into ANF-like form."""
    ANF().visit(node)
    cfg.forward(node, cfg.ReachingDefinitions())
    cfg.forward(node, cfg.Invariant())
    InvariantMotion().visit(node)
    return node


def invariant(node):
    cfg.forward(node, cfg.ReachingDefinitions())
    cfg.forward(node, cfg.Invariant())
    InvariantMotion().visit(node)
    return node
