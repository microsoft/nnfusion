import ast
import collections
# import inspect
import gast
import numbers
import sys
import types
import typing

from ast_analyzer.shape_inference.annotation import *
from ast_analyzer.shape_inference.types import *
from ast_analyzer.shape_inference.shape_elem import *
from ast_analyzer.shape_inference import utils

from ast_analyzer.shape_inference.ext.numpy_functions import *
from ast_analyzer.shape_inference.ext.pytorch_functions import *
from ast_analyzer.shape_inference.std.builtin_functions import *
from ast_analyzer.shape_inference.std.builtin_ops import *
from ast_analyzer.shape_inference.std.list_functions import *
from ast_analyzer.shape_inference.std.math_functions import *

import numpy as np
import logging

import torch
import torch.nn as nn

import astunparse
import inspect
import copy
# ==============================================================================


def debug(sth):
    frame = inspect.currentframe().f_back
    print("[{} {}] {}".format(frame.f_code.co_name, frame.f_lineno, sth))


def copy_tyenv(tyenv):
    new_tyenv = {}
    for name, ty in tyenv.items():
        new_tyenv[name] = copy_ty(ty)
    return new_tyenv


def copy_InferenceEngine(tc):
    new_tc = InferenceEngine(
        tyenv=tc.tyenv, attribute_tyenv=tc.attribute_tyenv,
        is_debug=tc.is_debug, module=tc.module, called_func=copy.copy(tc.called_func))
    return new_tc


def is_isNone(node):
    # Returns x if the 'node' is of the form 'x is None' for some x
    if isinstance(node, gast.Compare) and \
            isinstance(node.left, (gast.Name, gast.Attribute)) and \
            isinstance(node.ops[0], gast.Is) and \
            isinstance(node.comparators[0], gast.Constant) and \
            node.comparators[0].value is None:
        return node.left
    return None


def is_isNotNone(node):
    # Returns x if the 'node' is of the form 'x is None' for some x
    if isinstance(node, gast.Compare) and \
            isinstance(node.left, (gast.Name, gast.Attribute)) and \
            isinstance(node.ops[0], gast.IsNot) and \
            isinstance(node.comparators[0], gast.Constant) and \
            node.comparators[0].value is None:
        return node.left
    return None


def handle_inference_error(exception, func, node):
    print("exception:", exception)
    exit(1)
    if hasattr(func, '__name__'):
        name = func.__name__
    elif hasattr(func, '__class__'):
        name = func.__class__.__name__
    else:
        name = str(func)
    utils.print_warning(str(exception))
    utils.print_warning("Failed to infer type of " + name +
                        ". Falling back to TyVar...")
    # raise Exception
    return TyVar(lineno=getattr(node, 'lineno', None))


def call_function(table, func, node, ty_args, ty_kwargs):
    inference_logic = table[func]
    # try:
    ty_ret = inference_logic(ty_args, ty_kwargs)
    # except Exception as e:
    #     ty_ret = handle_inference_error(e, func, node)
    return ty_ret


def call_callable(table, obj, node, ty_args, ty_kwargs):
    inference_logic = table[type(obj)]
    # try:
    ty_ret = inference_logic(obj, ty_args, ty_kwargs)
    # except Exception as e:
    #     ty_ret = handle_inference_error(e, obj, node)
    return ty_ret

# TODO(momohatt): Deprecate this function.


def call_builtin_function(func, node, ty_args):
    # try:
    dummy_args = [generate_dummy_value(t) for t in ty_args]
    ty_ret = type_of_value(func(*dummy_args))
    # except Exception as e:
    #     ty_ret = handle_inference_error(e, func, node)
    return ty_ret


def call_binop(op, node, tyl, tyr):
    # try:
    return ty_ops(op, tyl, tyr)
    # except Exception as e:
    #     return handle_inference_error(e, op.__class__.__name__, node)


func_to_ignore = [logging.info]

# ==============================================================================


class InferenceEngine():
    def __init__(self, tyenv=None, attribute_tyenv=None, is_debug=False, module=None, func_inst=None, called_func=set()):
        # Type environments for local objects
        # string -> TyObj
        self.tyenv = {} if tyenv is None else copy_tyenv(tyenv)

        # Type environments for model attributes
        # (object, str) -> TyObj
        self.attribute_tyenv = {} if attribute_tyenv is None \
            else copy_tyenv(attribute_tyenv)

        # Annotation to input AST
        # Node -> TyObj
        self.nodetype = {}

        self.is_debug = is_debug
        self.module = module

        # Map from user-defined function call points to a list of inlined function ASTs
        # The length of the list is usually 1, but becomes >= 2 for the case
        # where multiple user-defined functions are called at once with
        # nn.Sequence
        # Node (Call) -> Node (FunctionDef)
        self.subroutine_node = collections.OrderedDict()
        self.called_func = called_func
        if func_inst is not None:
            called_func.add(func_inst)
        self.is_recursive = False
        self.func_inst = func_inst

    def dump_tyenv(self):
        print("=== tyenv ===")
        for name, ty in self.tyenv.items():
            print("{} : \x1b[35m{}\x1b[39m".format(name, ty))
        for (obj, name), ty in self.attribute_tyenv.items():
            # XXX: remove attributes inherited from libraries
            if name[0] == '_':
                continue
            print("{}.{} : \x1b[35m{}\x1b[39m".format(type(obj), name, ty))
        print()

    def dump_nodetype(self):
        for node, ty in self.nodetype.items():
            print("{} : \x1b[36m{}\x1b[39m".format(
                utils.node_description(node), ty))
        print()

    def dump_one_node(self, node):
        if node not in self.nodetype.keys():
            return
        print("{} : \x1b[36m{}\x1b[39m".format(
            utils.node_description(node), self.nodetype[node]))

    def generate_fresh_TyVar(self, node):
        assert isinstance(node, gast.Name)
        t = TyVar()
        self.nodetype[node] = t
        self.tyenv[node.id] = t
        return t

    def split_optional(self, tc1, tc2, x):
        # tc1: type inference engine for 'then' branch
        # tc2: type inference engine for 'else' branch
        if isinstance(x, gast.Name):
            if isinstance(self.tyenv[x.id], TyOptional):
                tc1.tyenv[x.id] = TyNone()
                tc2.tyenv[x.id] = self.tyenv[x.id].ty
            elif isinstance(self.tyenv[x.id], TyNone):
                tc2.tyenv[x.id] = TyVar()
            else:
                tc1.tyenv[x.id] = TyNone()
        if isinstance(x, gast.Attribute):
            obj = self.infer_expr(x.value).instance
            if isinstance(self.attribute_tyenv[(obj, x.attr)], TyOptional):
                tc1.attribute_tyenv[(obj, x.attr)] = TyNone()
                tc2.attribute_tyenv[(obj, x.attr)] = \
                    self.attribute_tyenv[(obj, x.attr)].ty
            elif isinstance(self.attribute_tyenv[(obj, x.attr)], TyNone):
                tc2.attribute_tyenv[(obj, x.attr)] = TyVar()
            else:
                tc1.attribute_tyenv[(obj, x.attr)] = TyNone()

    def infer_function_value_args(self, node, args, type_hints={}):
        # args: example inputs
        ty_args = [type_of_value(arg) for arg in args]
        return self.infer_function(node, ty_args, type_hints)

    def infer_function(self, node, ty_args, type_hints={}):
        # TODO(momohatt): varargs
        assert isinstance(node, gast.FunctionDef)
        if node.args.vararg is None:
            assert len(ty_args) >= len(node.args.args) - len(node.args.defaults) and \
                len(ty_args) <= len(node.args.args) + len(node.args.defaults), \
                "Wrong number of arguments: expected {}, default {}, got {}".format(
                len(node.args.args), len(node.args.defaults), len(ty_args))

        if self.is_debug:
            print("\x1b[33m==================== function {} ====================\x1b[39m".format(
                node.name))

        for i, arg_node in enumerate(node.args.args):
            if i < len(ty_args):
                self.tyenv[arg_node.id] = ty_args[i]
            else:
                # Use default value
                n = len(node.args.args) - i

                try:
                    value = eval(utils.expr_to_str(node.args.defaults[-n]))
                except Exception:
                    print("Default arguments must be constants")
                    raise Exception

                self.tyenv[arg_node.id] = type_of_value(value)

        # apply type hints
        subst = match_types([self.tyenv[n] for n in type_hints.keys() if n != 'return'],
                            type_hints.values())

        for n, t in type_hints.items():
            self.tyenv[n] = apply_subst(subst, type_hints[n])

        for i, arg_node in enumerate(node.args.args):
            if not isinstance(ty_args[i], TyUserDefinedClass):
                continue
            inst = ty_args[i].instance
            if isinstance(self.tyenv[arg_node.id], TyTree):
                env_ty = self.tyenv[arg_node.id]
                for attr, ty in env_ty.type_dict.items():
                    self.attribute_tyenv[(inst, attr)] = ty
            else:  # a general user defined class
                if i < len(ty_args):
                    for attr, val in inst.__dict__.items():
                        self.attribute_tyenv[(inst, attr)] = \
                            type_of_value(val)

        for arg in node.args.args:
            self.nodetype[arg] = self.tyenv[arg.id]

        self.infer_stmt(node)
        node._is_recursive = self.is_recursive
        return self.nodetype

    def infer_block(self, tc, stmts):  # use in if (without else), for, while
        for stmt in stmts:
            ty_ret = tc.infer_stmt(stmt)

        # unify the intersection of 2 tyenvs and update local tyenv
        for name, ty in tc.tyenv.items():
            if name in self.tyenv.keys():
                self.tyenv[name] = join(ty, self.tyenv[name])
            else:
                self.tyenv[name] = ty

        for (obj, name), ty in tc.attribute_tyenv.items():
            if (obj, name) in self.attribute_tyenv.keys():
                self.attribute_tyenv[(obj, name)] = \
                    join(ty, self.attribute_tyenv[(obj, name)])
            else:
                self.attribute_tyenv[(obj, name)] = ty

        unify(ty_ret, TyNone())
        return TyNone()

    def infer_2blocks(self, tc1, tc2, stmts1, stmts2):
        for stmt in stmts1:
            ty_ret1 = tc1.infer_stmt(stmt)
        for stmt in stmts2:
            ty_ret2 = tc2.infer_stmt(stmt)

        # unify the intersection of 2 tyenvs and update local tyenv
        for name, ty in tc1.tyenv.items():
            if name in tc2.tyenv.keys():
                self.tyenv[name] = join(ty, tc2.tyenv[name])
            else:
                self.tyenv[name] = ty
        for name, ty in tc2.tyenv.items():
            if name in tc1.tyenv.keys():
                continue
            self.tyenv[name] = ty

        for (obj, name), ty in tc1.attribute_tyenv.items():
            if (obj, name) in tc2.attribute_tyenv.keys():
                self.attribute_tyenv[(obj, name)] = \
                    join(ty, tc2.attribute_tyenv[(obj, name)])
            else:
                self.attribute_tyenv[(obj, name)] = ty
        for (obj, name), ty in tc2.attribute_tyenv.items():
            if (obj, name) in tc1.attribute_tyenv.keys():
                continue
            self.attribute_tyenv[(obj, name)] = ty

        return join(ty_ret1, ty_ret2)

    def infer_function_instance(self, node, func, ty_args, ty_kwargs):
        node.is_udf = False
        if func in numpy_func_ty.keys():
            return call_function(numpy_func_ty, func, node, ty_args, ty_kwargs)

        if func in pytorch_func_ty.keys():
            return call_function(pytorch_func_ty, func, node, ty_args, ty_kwargs)

        if type(func) in nn.__dict__.values():
            # torch.nn
            if isinstance(func, nn.Sequential):
                x_type, = ty_args
                if not hasattr(node, 'func_nodes'):
                    node.func_nodes = [None for _ in func.children()]
                is_udf = []
                for idx, module in enumerate(func.children()):
                    if node.func_nodes[idx] is not None:
                        node.func_node = node.func_nodes[idx]
                    else:
                        if hasattr(node, 'func_node'):
                            delattr(node, 'func_node')
                    x_type = self.infer_function_instance(
                        node, module, [x_type], {})
                    func_node = getattr(node, 'func_node', None)
                    if func_node is not None:
                        node.func_nodes[idx] = func_node
                    is_udf.append(getattr(node, 'is_udf', False))
                node.is_udf = is_udf
                node._func_inst = func
                return x_type

            return call_callable(pytorch_callable_ty, func, node, ty_args, ty_kwargs)

        if func in list_func_ty.keys():
            return call_function(list_func_ty, func, node, ty_args, ty_kwargs)

        if func in math_func_ty.keys():
            return call_function(math_func_ty, func, node, ty_args, ty_kwargs)

        if func in __builtins__.values():
            # builtin functions
            if func in builtin_func_ty.keys():
                return call_function(builtin_func_ty, func, node, ty_args, {})
            return call_builtin_function(func, node, ty_args)

        node.is_udf = True
        # user defined functions/methods/callables, need to inline
        return self.infer_user_defined_function(func, ty_args, node)

    def infer_user_defined_function(self, func, ty_args, node):
        old_func_node = None
        old_func_inst = None
        if hasattr(node, 'func_node'):
            old_func_node = node.func_node
            # print("use old inst")
            old_func_inst = old_func_node._func_inst
    
        func_node = None
        if isinstance(func, (types.FunctionType, types.MethodType)):
            func_body = func

            if isinstance(node.func, gast.Attribute):
                ty_self = self.nodetype[node.func.value]
                ty_args = [ty_self] + ty_args

        else:
            # defined with __call__
            if isinstance(func, nn.Module):
                func_body = func.forward
                # node._func_inst = func.forward
                if hasattr(func, '__ast__'):
                    func_node = func.__ast__.body[0]
            else:
                func_body = func.__call__

            ty_self = type_of_value(func)
            ty_args = [ty_self] + ty_args

        if func_body in self.called_func:
            self.is_recursive = True
            node._is_recursive = True
            hints = typing.get_type_hints(func_body)
            if 'return' in hints:
                return hints['return']
            else:
                return TyVar()

        if old_func_inst is not None and func_body == old_func_inst:
            func_node = node.func_node

        if func_node is None:
            func_node_module = utils.get_ast(func_body)
            func_node = func_node_module.body[0]

        if node not in self.subroutine_node.keys():
            self.subroutine_node[node] = [func_node]
        else:
            self.subroutine_node[node].append(func_node)
        new_called_func = copy.copy(self.called_func)
        new_called_func.add(func_body)

        tc = InferenceEngine(is_debug=self.is_debug,
                             module=sys.modules[func.__module__],
                             called_func=new_called_func,
                             func_inst=func_body)
        func_node._func_inst = func_body
        node.func_node = func_node
        # print("------infer callee------")
        if func_node.name == "save_for_backward":
            node.ctx_types = ty_args
        tc.infer_function(func_node, ty_args,
                          type_hints=typing.get_type_hints(func_body))
        # print("------infer callee end------")
        # copy nodetype and subroutine_node from subroutine
        utils.add_dict(self.nodetype, tc.nodetype)
        utils.add_dict(self.subroutine_node, tc.subroutine_node)
        return tc.nodetype[func_node].retty

    # ================================ mod =====================================

    def infer_mod(self, node):
        if isinstance(node, gast.Module):
            self.infer_stmt(node.body[0])
            return

    # ================================ stmt ====================================

    def infer_stmt(self, node):
        if self.is_debug:
            debug(gast.dump(node))

        if isinstance(node, gast.FunctionDef):
            self.nodetype[node] = self.infer_FunctionDef(node)
        elif isinstance(node, gast.Return):
            # Return(expr? value)
            if node.value is None:
                self.nodetype[node] = TyNone()
            else:
                self.nodetype[node] = self.infer_expr(node.value)
        elif isinstance(node, gast.Delete):
            # TODO(momohatt): erase from tyenv, etc.
            # TODO(momohatt): support deletion of element from list
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.Assign):
            self.infer_Assign(node)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.AugAssign):
            self.infer_AugAssign(node)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.For):
            self.infer_For(node)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.While):
            # While(expr test, stmt* body, stmt* orelse)
            self.infer_While(node)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.If):
            self.nodetype[node] = self.infer_If(node)
        elif isinstance(node, gast.Raise):
            self.nodetype[node] = TyVar()
        elif isinstance(node, gast.Try):
            # TODO(momohatt): What is 'finalbody' ?
            ty_ret = self.infer_2blocks(self, self, node.body, node.orelse)
            self.nodetype[node] = ty_ret
        elif isinstance(node, gast.Assert):
            self.nodetype[node] = TyNone()
        elif isinstance(node, (gast.Import, gast.ImportFrom)):
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.Expr):
            # Expr(expr value)
            self.infer_expr(node.value)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.Pass):
            self.nodetype[node] = TyNone()

        assert node in self.nodetype.keys(), "infer_stmt fail" + type(node).__name__
        return self.nodetype[node]

    def infer_FunctionDef(self, node):
        # FunctionDef(identifier name, arguments args, stmt* body,
        #             expr* decorator_list, expr? returns)
        ty_args = [self.tyenv[arg.id] for arg in node.args.args]
        ty = None

        for stmt in node.body:
            ty = self.infer_stmt(stmt)

        assert ty is not None
        return TyArrow(ty_args, ty)

    def infer_Assign(self, node):
        # Assign(expr* targets, expr value)
        assert len(node.targets) == 1  # cannot think of cases where >= 2
        target = node.targets[0]
        ty_val = self.infer_expr(node.value)

        if isinstance(target, gast.Name):
            self.tyenv[target.id] = ty_val
            self.nodetype[target] = ty_val
            return

        if isinstance(target, gast.Attribute):
            self.infer_expr(target.value)
            ty_obj = self.nodetype[target.value]
            assert isinstance(ty_obj, TyUserDefinedClass)
            self.attribute_tyenv[(ty_obj.instance, target.attr)] = ty_val
            self.nodetype[target] = ty_val
            return

        if isinstance(target, (gast.Tuple, gast.List)):
            assert isinstance(target, gast.Tuple)
            ty_target = TyTuple([self.generate_fresh_TyVar(e)
                                for e in target.elts])
            self.nodetype[target] = ty_target
            unify(ty_target, ty_val)
            for (var, ty) in zip(target.elts, ty_val.deref().get_tys()):
                self.tyenv[var.id] = ty
                self.nodetype[var] = ty
            return

        if isinstance(target, gast.Subscript):
            # Subscript(expr value, slice slice, expr_context ctx)
            ty_target = self.infer_expr(target.value).deref()
            if isinstance(target.slice, gast.Index):
                ty_index = self.infer_expr(target.slice.value).deref()
    
                if isinstance(ty_target, TyList):
                    assert is_subtype(ty_index, TyInt())
                    unify(ty_target, TyList(ty_val))
                    return

                if isinstance(ty_target, TyDict):
                    unify(ty_target, TyDict(ty_index, ty_val))
                    return
            elif isinstance(target.slice, gast.ExtSlice):
                self.infer_slice(target.slice)
                if isinstance(ty_target, TyTensor):
                    slice_shape = self.infer_Subscript_shape(ty_target.shape, target.slice)
                    unify(ty_val, TyTensor(ty_target.kind, ty_target.dtype, slice_shape))
                    return
                raise NotImplementedError
            else: raise NotImplementedError

    def infer_AugAssign(self, node):
        # AugAssign(expr target, operator op, expr value)
        tyr = self.infer_expr(node.value)
        tyl = self.infer_expr(node.target)
        ty_val = call_binop(node.op, node, tyl, tyr)
        if isinstance(tyl, TyList):
            unify(ty_val, tyl)

        if isinstance(node.target, gast.Name):
            self.tyenv[node.target.id] = ty_val

        if isinstance(node.target, gast.Attribute):
            ty_obj = self.nodetype[node.target.value]
            assert isinstance(ty_obj, TyUserDefinedClass)
            self.attribute_tyenv[(ty_obj.instance, node.target.attr)] = ty_val

        if isinstance(node.target, gast.Subscript):
            assert isinstance(node.target.slice, gast.Index)
            ty_target = self.infer_expr(node.target.value).deref()
            ty_index = self.infer_expr(node.target.slice.value).deref()

            if isinstance(ty_target, TyList):
                unify(ty_index, TyInt())
                unify(ty_target, TyList(ty_val))

            if isinstance(ty_target, TyDict):
                unify(ty_target, TyDict(ty_index, ty_val))

        self.nodetype[node.target] = ty_val

    def infer_For(self, node):
        # For(expr target, expr iter, stmt* body, stmt* orelse)
        assert isinstance(node.target, (gast.Name, gast.Tuple))

        ty_iteration = self.infer_expr(node.iter)
        ty_i = self.infer_expr(node.target)
        if isinstance(ty_iteration, TyTensor):
            unify(ty_i, TyTensor(ty_iteration.kind, ty_iteration.dtype,
                                 ty_iteration.shape[1:]))
        elif isinstance(ty_iteration, TyList):
            unify(ty_iteration, TyList(ty_i))
        else:
            unify(ty_iteration, TyTuple(ty_i))

        for _ in range(2):
            tc = copy_InferenceEngine(self)
            self.infer_block(tc, node.body)
            self.is_recursive = self.is_recursive or tc.is_recursive

        utils.add_dict(self.nodetype, tc.nodetype)
        utils.add_dict(self.subroutine_node, tc.subroutine_node)
    
    def infer_While(self, node):
        if not isinstance(node.test, gast.Name): raise NotImplementedError # node.test may be updated by body, cannot simply use infer_expr(node.test)
        for _ in range(2):
            ty_cond = self.infer_expr(node.test)
            unify(ty_cond, TyBool())
            tc = copy_InferenceEngine(self)
            self.infer_block(tc, node.body)
            self.is_recursive = self.is_recursive or tc.is_recursive
            self.nodetype[node.test] = join(self.nodetype[node.test], tc.tyenv[node.test.id])
        
        utils.add_dict(self.nodetype, tc.nodetype)
        utils.add_dict(self.subroutine_node, tc.subroutine_node)

    def infer_If(self, node):
        # If(expr test, stmt* body, stmt* orelse)
        # XXX: type of node.test can be anything
        self.infer_expr(node.test)

        x = is_isNone(node.test)

        if node.orelse == []:
            tc = copy_InferenceEngine(self)

            if x is not None:
                self.split_optional(tc, self, x)

            ty_ret = self.infer_block(tc, node.body)
            self.is_recursive = self.is_recursive or tc.is_recursive
            utils.add_dict(self.nodetype, tc.nodetype)
            utils.add_dict(self.subroutine_node, tc.subroutine_node)
            return ty_ret

        tc1 = copy_InferenceEngine(self)
        tc2 = copy_InferenceEngine(self)

        if x is not None:
            self.split_optional(tc1, tc2, x)

        ty_ret = self.infer_2blocks(tc1, tc2, node.body, node.orelse)
        self.is_recursive = self.is_recursive or tc1.is_recursive
        self.is_recursive = self.is_recursive or tc2.is_recursive
        utils.add_dict(self.nodetype, tc1.nodetype)
        utils.add_dict(self.nodetype, tc2.nodetype)
        utils.add_dict(self.subroutine_node, tc1.subroutine_node)
        utils.add_dict(self.subroutine_node, tc2.subroutine_node)
        return ty_ret


    def infer_Compare(self, node):
        # Compare(expr left, cmpop* ops, expr* comparators)
        self.infer_expr(node.left)
        for comparator in node.comparators:
            self.infer_expr(comparator)
        
        x = is_isNone(node)
        y = is_isNotNone(node)

        if x is None and y is None:
            return TyBool()

        if isinstance(self.nodetype[node.left], (TyVar, TyOptional)):
            return TyBool()

        if x is not None:
            if isinstance(self.nodetype[node.left], TyNone):
                return TyBool(True)
            else:
                return TyBool(False)
        
        if y is not None:
            if isinstance(self.nodetype[node.left], TyNone):
                return TyBool(False)
            else:
                return TyBool(True)

        raise ValueError() # unreachable!

    # ================================= expr ===================================

    def infer_expr(self, node):
        if node in self.nodetype.keys():
            return self.nodetype[node]

        if self.is_debug:
            pass
            # debug(gast.dump(node))
            # self.dump_tyenv()

        if hasattr(node, 'type_anno'):
            self.nodetype[node] = node.type_anno
        elif isinstance(node, gast.BoolOp):
            self.nodetype[node] = self.infer_BoolOp(node)
        elif isinstance(node, gast.BinOp):
            self.nodetype[node] = self.infer_BinOp(node)
        elif isinstance(node, gast.UnaryOp):
            self.nodetype[node] = self.infer_UnaryOp(node)
        elif isinstance(node, gast.Dict):
            self.nodetype[node] = self.infer_Dict(node)
        elif isinstance(node, gast.ListComp):
            self.nodetype[node] = self.infer_ListComp(node)
        elif isinstance(node, gast.Compare):
            self.nodetype[node] = self.infer_Compare(node)
        elif isinstance(node, gast.Call):
            self.nodetype[node] = self.infer_Call(node)
        elif isinstance(node, gast.Constant):
            # Constant(constant value)
            self.nodetype[node] = type_of_value(node.value)
        elif isinstance(node, gast.Attribute):
            self.nodetype[node] = self.infer_Attribute(node)
        elif isinstance(node, gast.Subscript):
            self.nodetype[node] = self.infer_Subscript(node)
        elif isinstance(node, gast.Name):
            self.nodetype[node] = self.infer_Name(node)
        elif isinstance(node, gast.List):
            # List(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = TyList(joins(elts_ty))
        elif isinstance(node, gast.Tuple):
            # Tuple(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = TyTuple(elts_ty)

        assert node in self.nodetype.keys() and \
            self.nodetype[node] is not None, "infer expr fail " + \
            type(node).__name__ + "\n" + astunparse.unparse(node)
        if self.is_debug:
            self.dump_one_node(node)
        return self.nodetype[node]

    def infer_BoolOp(self, node):
        # BoolOp(boolop op, expr* values)
        ty_vals = [self.infer_expr(val) for val in node.values]
        for ty in ty_vals:
            unify(ty, TyBool())
        return TyBool()

    def infer_BinOp(self, node):
        # BinOp(expr left, operator op, expr right)
        tyl = self.infer_expr(node.left).deref()
        tyr = self.infer_expr(node.right).deref()
        return call_binop(node.op, node, tyl, tyr)

    def infer_UnaryOp(self, node):
        # UnaryOp(unaryop op, expr operand)
        if isinstance(node.op, gast.Invert):
            ty_expr = self.infer_expr(node.operand)
            if isinstance(ty_expr, TyNum) and ty_expr.value is not None:
                raise NotImplementedError
            return ty_expr
        elif isinstance(node.op, gast.Not):
            ty_expr = self.infer_expr(node.operand)
            assert isinstance(ty_expr, TyNum) and ty_expr.is_bool()
            return TyBool()
        elif isinstance(node.op, gast.UAdd):
            pass
        elif isinstance(node.op, gast.USub):
            ty_expr = self.infer_expr(node.operand)
            if isinstance(ty_expr, TyNum) and ty_expr.value is not None:
                return type_of_value(- ty_expr.value)
            return ty_expr

    def infer_Dict(self, node):
        # Dict(expr* keys, expr* values)
        if node.keys == []:
            return TyDict(TyVar(), TyVar())
        ty_keys = [self.infer_expr(key) for key in node.keys]
        ty_vals = [self.infer_expr(val) for val in node.values]
        return TyDict(joins(ty_keys), joins(ty_vals))

    def infer_ListComp(self, node):
        # ListComp(expr elt, comprehension* generators)

        # cannot think of cases where len > 2
        assert len(node.generators) == 1
        gen = node.generators[0]
        # TODO: handle cases where len(gen.ifs) > 0
        assert len(gen.ifs) == 0

        tc = copy_InferenceEngine(self)
        ty_iteration = tc.infer_expr(gen.iter)
        self.is_recursive = self.is_recursive or tc.is_recursive
        ty_i = tc.generate_fresh_TyVar(gen.target)
        if isinstance(ty_iteration, TyTensor):
            unify(ty_i, TyTensor(ty_iteration.kind, ty_iteration.dtype,
                                 ty_iteration.shape[1:]))
        elif isinstance(ty_iteration, TyList):
            unify(ty_iteration, TyList(ty_i))
        else:
            unify(ty_iteration, TyTuple(ty_i))
        tc.infer_expr(node.elt)

        utils.add_dict(self.nodetype, tc.nodetype)
        utils.add_dict(self.subroutine_node, tc.subroutine_node)

        self.nodetype[node] = TyList(tc.nodetype[node.elt])
        return self.nodetype[node]

    def get_function_instance(self, node):
        if isinstance(node, gast.Attribute):
            if isinstance(node.value, gast.Name) and \
                    hasattr(self.module, node.value.id) and \
                    isinstance(getattr(self.module, node.value.id), types.ModuleType):
                # function of imported libraries (eg. np, F, L)
                module = getattr(self.module, node.value.id)
                return getattr(module, node.attr), None

            ty_obj = self.infer_expr(node.value).deref()

            if isinstance(ty_obj, TyList):
                return getattr(list, node.attr, None), ty_obj

            if isinstance(ty_obj, TyTensor):
                if ty_obj.is_ndarray():
                    return getattr(np.ndarray, node.attr, None), ty_obj
                if ty_obj.is_torch_tensor():
                    return getattr(torch.Tensor, node.attr, None), ty_obj

            if isinstance(ty_obj, TyUserDefinedClass):
                # if there is no such attribute, just return None (undefined)
                return getattr(ty_obj.instance, node.attr, None), None

            return None, None

        if isinstance(node, gast.Name):
            if node.id in self.tyenv.keys():
                ty = self.tyenv[node.id].deref()
                if isinstance(ty, TyUserDefinedClass):
                    return ty.instance, None

            if node.id in __builtins__.keys():
                return __builtins__[node.id], None

            if hasattr(self.module, node.id):
                return getattr(self.module, node.id), None

        if isinstance(node, gast.Subscript):
            subst = self.infer_Subscript(node)
            if isinstance(subst, TyUserDefinedClass):
                return subst.instance, None
            raise NotImplementedError

        assert False, "get function instance fail" + gast.dump(node)

    def infer_Call(self, node):
        # print("infer_call", astunparse.unparse(node))
        # Call(expr func, expr* args, keyword* keywords)
        func, ty_obj = self.get_function_instance(node.func)

        if func in func_to_ignore:
            return TyNone()

        # XXX: no need to deref() argument type later on
        ty_args = [self.infer_expr(arg).deref() for arg in node.args]
        ty_kwargs = {kwarg.arg: self.infer_expr(kwarg.value)
                     for kwarg in node.keywords}
        ty_ret = TyVar()

        if ty_obj is not None:
            ty_args_ = [ty_obj] + ty_args
        else:
            ty_args_ = ty_args

        if func is None:
            return TyVar()

        node._func_inst = func
        node._ty_obj = ty_obj
        node._is_recursive = False
        
        if isinstance(func, nn.Module):
            self.nodetype[node.func] = TyUserDefinedClass(type(func).__name__, func)
        ty_ret = self.infer_function_instance(node, func, ty_args_, ty_kwargs)
        return ty_ret.deref()

    def infer_Attribute(self, node):
        # print("infer_attr", astunparse.unparse(node))
        # Attribute(expr value, identifier attr, expr_context ctx)

        if isinstance(node.value, gast.Name) and \
                hasattr(self.module, node.value.id) and \
                isinstance(getattr(self.module, node.value.id), types.ModuleType):
            # function of imported libraries (eg. np, F, L)
            module = getattr(self.module, node.value.id)
            attr = getattr(module, node.attr)
            return type_of_value(attr)

        ty_obj = self.infer_expr(node.value).deref()

        if isinstance(ty_obj, TyTensor):
            if ty_obj.is_ndarray():
                logic = numpy_attr_ty[node.attr]
            elif ty_obj.is_chainer_variable():
                logic = chainer_attr_ty[node.attr]
            else:
                logic = pytorch_attr_ty[node.attr]
            return logic(ty_obj)

        if isinstance(ty_obj, TyTree):
            if node.attr in ty_obj.type_dict:
                return ty_obj.type_dict[node.attr]
            else:
                # raise notimplemented for debug
                raise NotImplementedError(ty_obj.type_dict, node.attr)
                return TyVar()

        if isinstance(ty_obj, TyUserDefinedClass):
            # x: value of existing instance
            x = getattr(ty_obj.instance, node.attr)

            if (ty_obj.instance, node.attr) in self.attribute_tyenv.keys():
                return self.attribute_tyenv[(ty_obj.instance, node.attr)]

            return type_of_value(x)

        if isinstance(ty_obj, TyDType):
            return type_of_value(getattr(ty_obj.t, node.attr))

        if isinstance(ty_obj, TyNone):
            return TyVar()

    def infer_Subscript(self, node):
        # Subscript(expr value, slice slice, expr_context ctx)
        ty_obj = self.infer_expr(node.value)

        if isinstance(ty_obj, TyList):
            self.infer_slice(node.slice)
            if isinstance(node.slice, gast.Index):
                return ty_obj.ty
            if isinstance(node.slice, gast.Slice):
                return ty_obj
            assert False, "ExtSlice for lists is not supported"

        if isinstance(ty_obj, TyTuple):
            self.infer_slice(node.slice)

            if ty_obj.is_fixed_len and isinstance(node.slice, gast.Index):
                t = self.infer_expr(node.slice.value)
                if isinstance(t, TyNum) and t.value is not None:
                    return ty_obj.get_tys()[t.value]

            if ty_obj.is_fixed_len and isinstance(node.slice, gast.Slice) and \
                    self.is_const_slice(node.slice):
                slice_ = self.extract_slice(node.slice)
                return TyTuple(ty_obj.get_tys()[slice_])

            if isinstance(node.slice, gast.Index):
                return ty_obj.get()
            if isinstance(node.slice, gast.Slice):
                return TyTuple(ty_obj.get())
            assert False, "ExtSlice for tuples is not supported"

        if isinstance(ty_obj, TyDict):
            self.infer_slice(node.slice, ty_obj.keyty)
            assert isinstance(node.slice, gast.Index)
            return ty_obj.valty

        if isinstance(ty_obj, TyTensor):
            self.infer_slice(node.slice)
            ret_shape = self.infer_Subscript_shape(ty_obj.shape, node.slice)
            return TyTensor(ty_obj.kind, ty_obj.dtype, ret_shape)

        if isinstance(ty_obj, TyUserDefinedClass):
            if isinstance(ty_obj.instance, nn.Sequential) and \
                    isinstance(node.slice, gast.Index):
                t = self.infer_expr(node.slice.value)
                if isinstance(t, TyNum) and t.value is not None:
                    return type_of_value(ty_obj.instance[t.value])

    def infer_Subscript_shape(self, shape, node_slice):
        if isinstance(node_slice, gast.Index):
            return shape[1:]
        if isinstance(node_slice, gast.Slice):
            if not self.is_const_slice(node_slice):
                return (None,) + shape[1:]
            if shape[0].value is None and (node_slice.upper is None or
                                           extract_value_from_ty(self.nodetype[node_slice.upper]) < 0):
                return (None,) + shape[1:]
            slice_ = self.extract_slice(node_slice)
            shape_0 = ShapeElem(len(((0,) * shape[0].value)[slice_]))
            return (shape_0,) + shape[1:]
        if isinstance(node_slice, gast.ExtSlice):
            ret_shape = ()
            for i in range(len(node_slice.dims)):
                ret_shape += self.infer_Subscript_shape(shape[i:i+1],
                                                        node_slice.dims[i])
            ret_shape += shape[len(node_slice.dims):]
            return ret_shape

    def infer_Name(self, node):
        # Name(identifier id, expr_context ctx, expr? annotation)
        if node.id in self.tyenv.keys():
            return self.tyenv[node.id]
        if node.id in __builtins__.keys():
            value = __builtins__[node.id]
            return type_of_value(value)
        if hasattr(self.module, node.id):
            x = getattr(self.module, node.id)
            return type_of_value(x)

        # XXX: print comes here
        ty_var = TyVar()
        self.tyenv[node.id] = ty_var
        return ty_var

    # ================================= slice ==================================

    def infer_slice(self, node, ty_key_expected=TyInt()):
        if isinstance(node, gast.Slice):
            # Slice(expr? lower, expr? upper, expr? step)
            if node.lower:
                ty_lower = self.infer_expr(node.lower)
                unify(ty_lower, ty_key_expected)
            if node.upper:
                ty_upper = self.infer_expr(node.upper)
                unify(ty_upper, ty_key_expected)
            if node.step:
                ty_step = self.infer_expr(node.step)
                unify(ty_step, ty_key_expected)
            return

        if isinstance(node, gast.Index):
            # Index(expr value)
            ty_val = self.infer_expr(node.value)
            unify(ty_val, ty_key_expected)
            return

        if isinstance(node, gast.ExtSlice):
            # ExtSlice(slice* dims)
            for s in node.dims:
                self.infer_slice(s, ty_key_expected)

    def is_const_slice(self, node_slice):
        def is_constnum(t): return isinstance(t, TyNum) and t.value is not None

        if node_slice.lower and not is_constnum(self.infer_expr(node_slice.lower)):
            return False
        if node_slice.upper and not is_constnum(self.infer_expr(node_slice.upper)):
            return False
        if node_slice.step and not is_constnum(self.infer_expr(node_slice.step)):
            return False
        return True

    def extract_slice(self, node_slice) -> slice:
        lower, upper, step = None, None, None
        if node_slice.lower:
            lower = self.infer_expr(node_slice.lower).value
        if node_slice.upper:
            upper = self.infer_expr(node_slice.upper).value
        if node_slice.step:
            step = self.infer_expr(node_slice.step).value
        return slice(lower, upper, step)


if __name__ == '__main__':
    from copy import deepcopy
    import ast
    import gast
    import importlib
    import sys
    import traceback

    try:
        from astmonkey import transformers, visitors
        IMPORT_ASTMONKEY = True
    except ImportError:
        IMPORT_ASTMONKEY = False

    def dump_ast(mod, name):
        if IMPORT_ASTMONKEY:
            mod = deepcopy(mod)
            mod = transformers.ParentChildNodeTransformer().visit(deepcopy(mod))
            visitor = visitors.GraphNodeVisitor()
            visitor.visit(mod)
            visitor.graph.write_png(name + '.png')
            print(
                "\033[1;32;40mAST visualization saved as \033[94m%s.png\033[0m" % name)
        else:
            print("\033[93mInstall astmonkey for visualization.\033[0m")

    if len(sys.argv) == 3:
        module = importlib.import_module(sys.argv[1])
        func = getattr(module, sys.argv[2])
        code = utils.clip_head(inspect.getsource(func))
    else:
        module = None
        code = open(sys.argv[1]).read()
    orig_ast = gast.ast_to_gast(ast.parse(code))
    dump_ast(orig_ast, 'original')

    tc = InferenceEngine(is_debug=True, module=module)
    try:
        nodetype = tc.infer(orig_ast)
    except UnifyError as e:
        print(traceback.format_exc(), end="")
