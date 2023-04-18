from . import cfg
from . import transformers
from . import ast_utils
from . import annotations as anno
from . import quoting
from . import template

import astunparse
import gast

from ast_analyzer.shape_inference.types import *
from ast_analyzer.shape_inference.shape_elem import unwrap_shape


class CleanUndef(transformers.TreeTransformer):
    def __init__(self, device):
        super(CleanUndef, self).__init__()
        self.depth = 0
        self.device = device

    def visit_Assign(self, node):
        if anno.hasanno(node, 'store_stmt'):
            store, restore, arg, ret = anno.getanno(node, 'related_nodes')
            assert(len(store.targets) == 1)
            assert(isinstance(store.value, gast.Name))
            name = store.value.id
            if name not in anno.getanno(node, 'defined_in'):
                if self.depth == 0:
                    self.remove(store)
                    self.remove(restore)
                    self.remove(arg)
                    self.remove(ret)
                else:
                    to_insert = quoting.quote('{} = None'.format(name))
                    if anno.hasanno(node.value, 'type'):
                        ty = anno.getanno(node.value, 'type')
                        if isinstance(ty, TyTensor):
                            to_insert = template.replace(
                                "tmp = empty",
                                tmp=gast.Name(
                                    id=name, annotation=None, ctx=gast.Store, type_comment=None),
                                empty=ast_utils.tensor_of_type(ty, device=self.device).value
                            )
                    self.insert_top(to_insert)

        return node

    def visit_For(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        return node


class CleanGrad(gast.NodeTransformer):
    """Replace `dx = dx + partial` with `dx = partial` if `dx` undefined."""

    def visit_Assign(self, node):
        if anno.hasanno(node, 'add_grad'):
            defs = frozenset(id_ for id_, node in anno.getanno(
                node, 'definitions_in'))
            if ast_utils.get_name(node.targets[0]) not in defs:
                node.value = node.value.right.args[0]
        return node


class FixGrad(transformers.TreeTransformer):
    """Explicitly initialize gradient to zero if needed."""

    def __init__(self, device):
        super(FixGrad, self).__init__()
        self.added = set()
        self.device = device

    def _init(self, node):
        gradname = ast_utils.get_name(node)
        if anno.hasanno(node, 'adjoint_var'):
            var = anno.getanno(node, 'adjoint_var')
        else:
            var = anno.getanno(node, 'temp_adjoint_var')

        if anno.hasanno(var, 'type'):
            ty = anno.getanno(var, 'type')
            value = ast_utils.generate_zero_ast(var, ty, self.device)
        else:
            raise NotImplementedError

        return gast.Assign(
            targets=[
                gast.Name(id=gradname, ctx=gast.Store(), annotation=None, type_comment=None)],
            value=value)

    def prepend_uninitialized_grads(self, node):
        if anno.hasanno(node, 'defined_in'):
            uses = (succ for succ in gast.walk(node) if
                    isinstance(succ, gast.Name) and
                    isinstance(succ.ctx, gast.Load))
            for use in uses:
                if ((anno.hasanno(use, 'adjoint_var') or
                     anno.hasanno(use, 'temp_adjoint_var')) and
                    use.id not in anno.getanno(node, 'defined_in') and
                        use.id not in self.added):
                    self.added.add(use.id)
                    self.insert_top(self._init(use))
        return node

    def visit_Assign(self, node):
        node = self.prepend_uninitialized_grads(node)
        return node

    def visit_AugAssign(self, node):
        node = self.prepend_uninitialized_grads(node)
        return node

    def visit_Expr(self, node):
        node = self.prepend_uninitialized_grads(node)
        return node

    def visit_Return(self, node):
        node = self.prepend_uninitialized_grads(node)
        return node


class SplitToIndex(transformers.TreeTransformer):
    def __init__(self):
        super().__init__()
        self.last_assign = None
    
    def visit_Assign(self, node): # reshape + split => gather TODO: clean impl.
        if isinstance(node.value, gast.Call) and isinstance(node.value.func, gast.Attribute) and node.value.func.attr == 'split' and \
            isinstance(self.last_assign.value, gast.Call) and isinstance(self.last_assign.value.func, gast.Attribute) and self.last_assign.value.func.attr == 'reshape':
            targets = node.targets[0].elts
            inp = node.value.args[0]
            if isinstance(self.last_assign.targets[0], gast.Name) and isinstance(inp, gast.Name) and inp.id == self.last_assign.targets[0].id:
                real_inp = self.last_assign.value.args[0]
                for i, target in enumerate(targets):
                    assign_code = f"{astunparse.unparse(target).strip()}={astunparse.unparse(real_inp).strip()}[{i}]"
                    # print(assign_code)
                    new_assign_node = gast.parse(assign_code).body[0]
                    self.append(new_assign_node)
                self.remove(self.last_assign)
                self.remove(node)
        self.last_assign = node
        return node


class RemoveGrad(transformers.TreeTransformer):
    def __init__(self, name):
        super().__init__()
        self.avoid = name
        self.to_remove_assign = False
    
    def visit_Name(self, node):
        if node.id == self.avoid:
            self.to_remove_assign = True
        return node
    
    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if self.to_remove_assign:
            self.to_remove_assign = False
            # print("remove", astunparse.unparse(node))
            # return gast.Name(id='Pass', ctx=gast.Load(), annotation=None, type_comment=None)
            self.remove(node)
        return node
    
    def visit_Return(self, node):
        if isinstance(node.value, gast.Tuple):
            elts = []
            for elt in node.value.elts:
                if isinstance(elt, gast.Name) and elt.id == self.avoid:
                    elts.append(gast.Constant(value=None, kind=None))
                    self.to_remove_assign = False
                else:
                    elts.append(elt)
            node.value.elts = elts
        return node


# class ZeroToSub(transformers.TreeTransformer):
#     def __init__(self):
#         super().__init__()
#         self.in_loop = 0

#     def visit_For(self, node):
#         self.in_loop += 1
#         self.generic_visit(node)
#         self.in_loop -= 1
#         return node
    
#     def visit_Assign(self, node):
#         if self.in_loop > 0 and isinstance(node.value, gast.Call) and isinstance(node.value.func, gast.Attribute) and node.value.func.attr == 'zeros':
#             print(astunparse.unparse(node))
#         return node

def simplify(node, device):
    pri_cfg = cfg.CFG.build_cfg(node.body[0])
    defined = cfg.Defined()  # must reach
    defined.visit(pri_cfg.entry)
    reaching = cfg.ReachingDefinitions()  # may reach
    reaching.visit(pri_cfg.entry)

    cfg.forward(node.body[1], cfg.Defined())
    cfg.forward(node.body[1], cfg.ReachingDefinitions())

    CleanUndef(device).visit(node)
    CleanGrad().visit(node)
    FixGrad(device).visit(node)

    RemoveGrad("binputs").visit(node) # TODO: not hardcode