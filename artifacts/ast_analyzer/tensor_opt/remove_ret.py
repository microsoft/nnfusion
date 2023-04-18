from ast_analyzer.grad import annotations as anno
import gast
import astunparse
from ast import iter_fields, AST
import copy
from ast_analyzer.shape_inference.types import *
import typing


class FindUnresolveCall(gast.NodeVisitor):
    def __init__(self):
        self.has_unresolve = False

    def visit_Call(self, node):
        if not hasattr(node, '_func_inst') or node._func_inst is None:
            self.has_unresolve = True
            print("WA at", astunparse.unparse(node))
        self.generic_visit(node)

    def visit_arguments(self, node):
        # skip type annotations
        pass

    def visit_FunctionDef(self, node):
        if not hasattr(node, '_func_inst') or node._func_inst is None:
            self.has_unresolve = True
        for field, value in iter_fields(node):
            # skip type annotations
            if field == 'returns':
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        self.visit(item)
            elif isinstance(value, AST):
                self.visit(value)


class ReplaceFuncName(gast.NodeTransformer):
    def __init__(self, func_inst, new_name):
        self.func_inst = func_inst
        self.new_name = new_name

    def visit_Call(self, node):
        assert(hasattr(node, '_func_inst'))
        if node._func_inst == self.func_inst:
            if isinstance(node.func, gast.Attribute):
                node.func.attr = copy.deepcopy(self.new_name)
            elif isinstance(node.func, gast.Name):
                node.func.id = copy.deepcopy(self.new_name)
            else:
                raise NotImplemented
        self.generic_visit(node)
        return node

    def visit_arguments(self, node):
        # skip type annotations
        return node

    def visit_FunctionDef(self, node):
        assert(hasattr(node, '_func_inst'))
        if node._func_inst == self.func_inst:
            node.name = self.new_name
        for field, old_value in iter_fields(node):
            if field == 'returns':
                continue
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class RemoveRet(gast.NodeTransformer):
    pass


def is_fixed_shape_tree(ty):
    if isinstance(ty, TyTensor):
        return ty.is_fixed_shape()
    if isinstance(ty, TyNum):
        return True
    if isinstance(ty, TyTuple):
        if ty.size() is not None:
            for t in ty.get_tys():
                if not is_fixed_shape_tree(t):
                    return False
            return True
        else:
            return False
    return False


def shape_tree_to_list(ty):
    if isinstance(ty, TyTensor):
        return [ty]
    if isinstance(ty, TyNum):
        return [TyTensor(TensorKind.torch_tensor, ty.kind, ())]
    if isinstance(ty, TyTuple):
        if ty.size() is not None:
            ret = []
            for t in ty.get_tys():
                ret += shape_tree_to_list(t)
            return ret

    assert(False)  # unreachable


def to_ret_stmt(ty, cur, new_idx):
    if isinstance(ty, TyTensor):
        tensor = f"buf_{cur}[{new_idx}]"
        cur += 1
        return tensor, cur
    if isinstance(ty, TyNum):
        tensor = f"buf_{cur}[{new_idx}].item()"
        cur += 1
        return tensor, cur
    if isinstance(ty, TyTuple):
        if ty.size() is not None:
            ret = ""
            for t in ty.get_tys():
                code, cur = to_ret_stmt(t, cur, new_idx)
                if ret != "":
                    ret += ", "
                ret += code
            return ret, cur


def wrap_recursion(node, func_inst):
    outer_func = node.body[0].name
    inner_func = 'inner_' + outer_func
    ReplaceFuncName(func_inst, inner_func).visit(node)
    new_outer = copy.copy(node.body[0])
    new_outer.name = outer_func
    params = [x.id for x in node.body[0].args.args]
    if params[0] == 'self':
        new_body = f"return self.{inner_func}({', '.join(params[1:])})"
    else:
        new_body = f"return {inner_func}({', '.join(params)})"
    new_body_ast = gast.parse(new_body).body[0]
    new_outer.body = [new_body_ast]
    node.body.insert(0, new_outer)
    return True


def modify_outer(func_node, plain_types, return_type, new_idx, batch_size):
    buffer_body = []
    for i, ty in enumerate(plain_types):
        assert(isinstance(ty, TyTensor))
        shape_strs = [str(x) for x in ty.unwrapped_shape()]
        code = f"buf_{i} = torch.empty(({batch_size}, {', '.join(shape_strs)}), dtype={np_dtype_to_torch_string(ty.dtype)}, device='cuda')"
        buf_ast = gast.parse(code)
        buffer_body.append(buf_ast.body[0])

    func_call_ast = func_node.body[0].value
    assert(isinstance(func_call_ast, gast.Call))
    for i in range(len(plain_types)):
        func_call_ast.args.append(
            gast.Name(id=f'buf_{i}', ctx=gast.Load(), annotation=None, type_comment=None))

    func_node.body = buffer_body + \
        [gast.Expr(value=func_call_ast)]

    ret_code, _ = to_ret_stmt(return_type, 0, new_idx)
    ret_expr = gast.parse(ret_code)
    ret_ast = gast.Return(value=ret_expr.body[0].value)
    func_node.body.append(ret_ast)
    print(astunparse.unparse(func_node))


def modify_inner():
    pass


def remove_ret(node, func_inst):
    if not node.body[0]._is_recursive:
        print("not a recursive function")
        return False
    unresolve_pass = FindUnresolveCall()
    unresolve_pass.visit(node)
    if unresolve_pass.has_unresolve:
        print("cannot optimize this recursive function")
        return False
    if not hasattr(node.body[0], '_grad_anno') or not anno.hasanno(node.body[0], 'hint'):
        return False
    hints = anno.getanno(node.body[0], 'hint')
    if 'index_by' not in hints:
        return False
    if not wrap_recursion(node, func_inst):
        return False

    print(astunparse.unparse(node))

    new_idx, batch_size = hints['index_by']

    hints = typing.get_type_hints(func_inst)
    if not 'return' in hints:
        return
    return_type = hints['return']
    if not is_fixed_shape_tree(return_type):
        return False

    plain_types = tuple(shape_tree_to_list(return_type))

    modify_outer(node.body[0], plain_types, return_type, new_idx, batch_size)
    modify_inner()

    print("TODO")


# 先保证所有 call 都找到了 instance，再按 instance 判断要调用的函数
