from ast_analyzer.grad import annotations as anno
from ast_analyzer.grad import cfg
import ast
import gast
import astunparse
import copy
from .fake_device_compiler import DeviceCompilerWrapper
from ast_analyzer.python_std.optimizations import *
import torch
from ast_analyzer.shape_inference.types import *
import numpy as np
import types
from ast_analyzer.grad.annotate import mark_shape, resolve_calls, resolve_anno, get_arg_ret
from ast_analyzer.tensor_opt.mark_in_py import mark_may_push
from .utils import Namer, is_call_stmt
from .inliner import inline
from ast_analyzer.utils.misc import white_list
from ast_analyzer.to_onnx.to_torch_func import to_torch_autograd
import importlib

ENABLE_CONTROL_FLOW=True
SEARCH_ALL_SUBAST=False

class MergeChance:
    def __init__(self, external_functions = set()):
        self.external_functions = external_functions

    def join(a, b):
        assert(isinstance(a, MergeChance))
        assert(isinstance(b, MergeChance))
        return MergeChance(external_functions = set.union(a.external_functions, b.external_functions))


CONTROL_FLOW_BLOCK = (gast.For, gast.AsyncFor, gast.While, gast.If, gast.Try)


def get_scope(inst):
    if hasattr(inst, "__self__"):
        if isinstance(inst.__self__, torch.nn.Sequential):
            raise NotImplementedError
        return inst.__self__
    elif isinstance(inst, torch.nn.Module):
        return inst
    else:
        raise NotImplementedError


# an old node can be a new node's child, but a new node cannot be an old node's child
class ButtomUpFeed:
    def __init__(self, model_name, type_dict, scope, cfg_nodes, simple_mode, platform):
        super(ButtomUpFeed, self).__init__()
        self.model_name = model_name
        self.type_dict = type_dict
        self.subgraph_id = 0
        self.called_func = set()
        self.scope = scope
        self.cfg_nodes = cfg_nodes
        self.simple_mode = simple_mode
        self.func2file = dict()
        self.platform = platform

    def gen_new_graph_name(self):
        name = f"{self.model_name}_{self.subgraph_id}"
        self.subgraph_id += 1
        return name


    def try_inline(self, stmt, namer):
        if is_call_stmt(stmt) and stmt.value.is_udf and stmt.value._func_inst not in self.called_func and hasattr(stmt.value, 'func_node'):
            inlined_stmts = inline(stmt, namer)
            return inlined_stmts
        elif isinstance(stmt, gast.For):
            new_body = [] # warning: breaks the father-child constrain
            for s in stmt.body:
                new_body.extend(self.try_inline(s, namer))
            stmt.body = new_body
            return [stmt]
        elif isinstance(stmt, gast.If):
            new_body = [] # warning: breaks the father-child constrain
            for s in stmt.body:
                new_body.extend(self.try_inline(s, namer))
            stmt.body = new_body

            new_orelse = [] # warning: breaks the father-child constrain
            for s in stmt.orelse:
                new_orelse.extend(self.try_inline(s, namer))
            stmt.orelse = new_orelse
            return [stmt]
        else:
            return [stmt]


    def split_body(self, stmts): # -> can_merge, may_merge
        can_merge = True
        may_merge = True
        body_stmts = []
        body_stmts_original = []
        gen_stmts = []
        to_import_list = []
        chance = MergeChance()

        def step_before(body_stmts, body_stmts_original, gen_stmts, to_import_list, next_stmt = None):
            if len(body_stmts_original) > 0 and isinstance(body_stmts_original[-1], gast.Return):
                body_stmts_original = body_stmts_original[:-1]
            if len(body_stmts_original) > 0:
                dev_compiler = DeviceCompilerWrapper(self.scope, self.gen_new_graph_name(), self.type_dict, self.platform)
                args, rets = get_arg_ret(body_stmts_original, self.cfg_nodes, white_list)
                gen, to_import = dev_compiler.run(body_stmts, args, rets, self.cfg_nodes, self.simple_mode)
                if to_import is not None:
                    to_import_list.append(to_import)
                if self.simple_mode:
                    self.func2file[dev_compiler.func_name] = dev_compiler.file_name
                gen_stmts.extend(gen)
                body_stmts.clear()
                body_stmts_original.clear()
                if next_stmt is not None:
                    gen_stmts.append(next_stmt)
                real_stmts = 0
                for stmt in gen:
                    if not isinstance(stmt, gast.Return):
                        real_stmts += 1
                can_merge = (real_stmts == 1)
                return can_merge
            else:
                if len(body_stmts) > 0 and isinstance(body_stmts[-1], gast.Return):
                    gen_stmts.append(body_stmts[-1])
                if next_stmt is not None:
                    gen_stmts.append(next_stmt)
                return True
        
        namer = Namer()
        for stmt in stmts:
            if not anno.getanno(stmt, 'may_push'):
                step_before(body_stmts, body_stmts_original, gen_stmts, to_import_list, stmt)
                can_merge = may_merge = False
            elif is_call_stmt(stmt):
                if stmt.value.is_udf:
                    func_inst = stmt.value._func_inst
                    assert(func_inst is not None)
                    if func_inst in self.called_func:
                        can_merge_stmts = step_before(body_stmts, body_stmts_original, gen_stmts, to_import_list, stmt)
                        can_merge = False
                        if not can_merge_stmts:
                            may_merge = False
                        chance.external_functions.add(func_inst)
                        # keep may_merge unchanged
                    else:
                        # print("[user defined function]")
                        # print(astunparse.unparse(stmt.value.func_node))
                        # print("func_inst", stmt.value._func_inst)
                        if SEARCH_ALL_SUBAST:
                            can_merge_node, may_merge_node, new_node, chance, to_import_sub = buttom_up_feed(stmt.value.func_node, self.gen_new_graph_name(), self.type_dict, get_scope(stmt.value._func_inst), stmt.value._func_inst, self.platform)
                            if not may_merge_node:
                                may_merge = False
                            if not can_merge_node:
                                can_merge_stmts = step_before(body_stmts, body_stmts_original, gen_stmts, to_import_list, stmt)
                                can_merge = False
                                if not can_merge_stmts:
                                    may_merge = False
                            if can_merge_node:
                                inlined_stmts = inline(stmt, namer)
                                body_stmts.extend(inlined_stmts)
                                body_stmts_original.append(stmt)
                            else:
                                to_import_list.extend(to_import_sub)
                        else:
                            inlined_stmts = self.try_inline(stmt, namer)
                            body_stmts.extend(inlined_stmts)
                            body_stmts_original.append(stmt)
                else:
                    body_stmts.append(stmt)
                    body_stmts_original.append(stmt)
            elif isinstance(stmt, CONTROL_FLOW_BLOCK):
                if SEARCH_ALL_SUBAST:
                    can_merge_node, may_merge_node, new_node, chance, to_import_sub = self.split(stmt)
                    if not may_merge_node:
                        may_merge = False
                    if not can_merge_node: # or worse_try?
                        can_merge_stmts = step_before(body_stmts, body_stmts_original, gen_stmts, to_import_list, new_node)
                        to_import_list.extend(to_import_sub)
                        can_merge = False
                        if not can_merge_stmts:
                            may_merge = False
                    else:
                        body_stmts.append(stmt)
                        body_stmts_original.append(stmt)
                else:
                    body_stmts.extend(self.try_inline(stmt, namer))
                    body_stmts_original.append(stmt)
            else:
                body_stmts.append(stmt)
                body_stmts_original.append(stmt)
        can_merge_stmts = step_before(body_stmts, body_stmts_original, gen_stmts, to_import_list, None)
        if can_merge: # didn't try step_before
            can_merge = may_merge = can_merge_stmts
        else:
            can_merge = False
            may_merge = may_merge and can_merge_stmts
        # print("[can_merge] {} [may_merge] {} [can_merge_stmts] {}".format(can_merge, may_merge, can_merge_stmts))
        # print("to_import:", [x[2] for x in to_import_list])
        # print("++++++++++++++++++++++++++++++++++++++++++")
        # print(unparse_ast_list(stmts))
        # print("------------------------------------------")
        # print(unparse_ast_list(gen_stmts))
        # print("++++++++++++++++++++++++++++++++++++++++++")
        return can_merge, may_merge, gen_stmts, chance, to_import_list
    
    # only be called when may_merge = true
    def split_node(self, stmt):
        if not anno.getanno(stmt, 'may_push'):
            return False, False, stmt, MergeChance()
        func2name = {}
        if isinstance(stmt, gast.FunctionDef):
            func2name[stmt._func_inst] = f'func_{stmt.name}'
            body = stmt.body
            check_model = False
            wrap_recursion = True # TODO: a functiondef may not be recursive
        else:
            check_model = True
            wrap_recursion = False
            body = [stmt]

        namer = Namer()
        body_inlined = []
        for st in body:
            body_inlined.extend(self.try_inline(st, namer))

        dev_compiler = DeviceCompilerWrapper(self.scope, self.gen_new_graph_name(), self.type_dict, self.platform)
        if isinstance(body[-1], gast.Return):
            args, rets = get_arg_ret(body[:-1], self.cfg_nodes, white_list)
        else:
            args, rets = get_arg_ret(body, self.cfg_nodes, white_list)
        if isinstance(stmt, gast.FunctionDef):
            args = [arg.id for arg in stmt.args.args]
        gen, to_import = dev_compiler.run(body_inlined, args, rets, self.cfg_nodes, self.simple_mode, func2name=func2name, check_model=check_model, wrap_recursion=wrap_recursion)
        if self.simple_mode:
            self.func2file[dev_compiler.func_name] = dev_compiler.file_name
        if gen is None:
            print("split node: None")
            return False, False, [stmt], MergeChance(), []
        else:
            if not isinstance(stmt, gast.FunctionDef):
                assert(len(gen) == 1)
            return True, True, gen, MergeChance(), [to_import]

    def split_FunctionDef(self, node):
        func_inst = node._func_inst
        assert(func_inst not in self.called_func)
        self.called_func.add(func_inst)
        can_merge, may_merge, body_stmts, chance, to_import_list = self.split_body(node.body)
        # TODO: precise worse_try
        worse_try = False

        # worse_try case 1: top definition of recursive function
        assert(func_inst in self.called_func)
        self.called_func.remove(func_inst)
        if func_inst in chance.external_functions and ENABLE_CONTROL_FLOW:
            chance.external_functions.remove(func_inst)
            if len(chance.external_functions) == 0:
                print("worse try because no external function")
                worse_try = True
        # case 1 end

        # print(f"may merge {may_merge}, worse_try {worse_try}")

        if may_merge and worse_try:
            can_merge, may_merge, func_stmts, try_chance, to_import_sub = self.split_node(node)
            if can_merge:
                body_stmts = func_stmts
                chance = try_chance
                to_import_list = to_import_sub

        new_node = copy.deepcopy(node)
        new_node.body = body_stmts
        print("[after functiondef]", astunparse.unparse(new_node))
        return can_merge, may_merge, new_node, chance, to_import_list

    def split_If(self, node):
        can_merge_body, may_merge_body, body_stmts, body_chance, to_import_body = self.split_body(node.body)
        can_merge_orelse, may_merge_orelse, orelse_stmts, orelse_chance, to_import_orelse = self.split_body(node.orelse)
        may_merge_if = may_merge_body and may_merge_orelse
        if_chance = MergeChance.join(body_chance, orelse_chance)

        if can_merge_body and can_merge_orelse and ENABLE_CONTROL_FLOW:
            can_merge_if, may_merge_if, merged_stmts, body_chance, to_import_sub = self.split_node(node)
            if can_merge_if:
                assert(len(merged_stmts) == 1)
                merged_stmt = merged_stmts[0]
                return can_merge_if, may_merge_if, merged_stmt, body_chance, to_import_sub
        else:
            can_merge_if = False

        assert(can_merge_if == False)
        new_node = copy.deepcopy(node)
        new_node.body = body_stmts
        new_node.orelse = orelse_stmts
        print("[if node]")
        print(astunparse.unparse(node))
        # print(f"if node: body {can_merge_body} {may_merge_body} orelse {can_merge_orelse} {may_merge_orelse} node {can_merge_if} {may_merge_if}")
        return can_merge_if, may_merge_if, new_node, body_chance, to_import_body + to_import_orelse

    def split_For(self, node):
        can_merge_body, may_merge_body, body_stmts, body_chance, to_import_body = self.split_body(node.body)
        if can_merge_body and ENABLE_CONTROL_FLOW:
            can_merge_for, may_merge_for, merged_stmts, body_chance, to_import_sub = self.split_node(node)
            if can_merge_for:
                assert(len(merged_stmts) == 1)
                merged_stmt = merged_stmts[0]
                return can_merge_for, may_merge_for, merged_stmt, body_chance, to_import_sub
        else:
            can_merge_for = False
            may_merge_for = may_merge_body

        assert(can_merge_for == False)
        new_node = copy.deepcopy(node)
        new_node.body = body_stmts
        print("[for node]")
        print(astunparse.unparse(new_node))
        return can_merge_for, may_merge_for, new_node, body_chance, to_import_body

    def split_While(self, node):
        can_merge_body, may_merge_body, body_stmts, body_chance, to_import_body = self.split_body(node.body)
        if can_merge_body and ENABLE_CONTROL_FLOW:
            can_merge_while, may_merge_while, merged_stmts, body_chance, to_import_sub = self.split_node(node)
            if can_merge_while:
                assert(len(merged_stmts) == 1)
                merged_stmt = merged_stmts[0]
                return can_merge_while, may_merge_while, merged_stmt, body_chance, to_import_sub
        else:
            can_merge_while = False
            may_merge_while = may_merge_body

        assert(can_merge_while == False)
        new_node = copy.deepcopy(node)
        new_node.body = body_stmts
        return can_merge_while, may_merge_while, new_node, body_chance, to_import_body

    def split_Module(self, node):
        can_merge = True
        may_merge = True
        chance = MergeChance()
        new_node = copy.deepcopy(node)
        new_node.body.clear()
        to_import_list = []
        for body in node.body:
            can_merge_func, may_merge_func, merged_stmt, body_chance, to_import_sub = self.split(body)
            new_node.body.append(merged_stmt)
            can_merge = can_merge and can_merge_func
            may_merge = may_merge and may_merge_func
            chance = MergeChance.join(chance, body_chance)
            to_import_list.extend(to_import_sub)
        return can_merge, may_merge, new_node, chance, to_import_list

    def split(self, node):
        method = 'split_' + node.__class__.__name__
        if not hasattr(self, method):
            raise ValueError('Unsupported node type: %s' % node.__class__.__name__)
        visitor = getattr(self, method)
        return visitor(node)
    
    def run(self, node):
        if isinstance(node, gast.Module):
            pass
        elif isinstance(node, gast.FunctionDef):
            node = gast.Module(body=[node], type_ignores=[])
        else:
            raise ValueError('ButtomUpFeed:', type(node))
        return self.split(node)


def buttom_up_feed(node, model_name, type_dict, scope, func_inst, platform):
    # print("[buttom_up_feed]")
    # print(astunparse.unparse(node))
    # process annotation
    resolve_calls(node, func_inst) # TODO: get the real function
    mark_shape(node, type_dict)
    resolve_anno(node)

    # buttom_up
    mark_may_push(node)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    cfg_nodes = cfg.backward(node, cfg.BackwardActive())
    can_merge, may_merge, merged_node, chance, to_import_list = ButtomUpFeed(model_name, type_dict, scope, cfg_nodes, False, platform).run(node)
    merged_node_ast = gast.gast_to_ast(merged_node)
    merged_node_ast= ast.fix_missing_locations(merged_node_ast)

    code = compile(merged_node_ast, f'{model_name + "_tmp"}.py', 'exec')
    namespace = {
        'torch': torch,
        'TyTorchTensor': TyTorchTensor,
        'TyTuple': TyTuple,
        'TyInt': TyInt,
        'TyBool': TyBool,
        'np': np
    }
    exec(code, namespace)
    func_name = merged_node.body[0].name
    black_box = types.MethodType(namespace[func_name], scope)
    setattr(scope, func_name, black_box)
    return can_merge, may_merge, merged_node, chance, to_import_list


def buttom_up_feed_simple(node, model_name, type_dict, scope, func_inst, platform):
    # print(astunparse.unparse(node))

    # process annotation
    resolve_calls(node, func_inst) # TODO: get the real function
    mark_shape(node, type_dict)
    resolve_anno(node)

    # buttom_up
    mark_may_push(node)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    cfg_nodes = cfg.backward(node, cfg.BackwardActive())
    buf = ButtomUpFeed(model_name, type_dict, scope, cfg_nodes, True, platform)
    _, _, merged_node, _, to_import_list = buf.run(node)
    for todo in to_import_list:
        assert(todo is None)
    return merged_node, buf.func2file


def buttom_up_feed_train(node_fwd, node_bwd, model_name, type_dict_fwd, type_dict_bwd, inst, fwd_ret_cnt, attrs_order, func_name, model_scope, platform):
    merged_node_fwd, func2file_fwd = buttom_up_feed_simple(node_fwd, model_name + '_fwd', type_dict_fwd, inst, inst.forward, platform)
    merged_node_bwd, func2file_bwd = buttom_up_feed_simple(node_bwd, model_name + '_bwd', type_dict_bwd, inst, inst.backward, platform)

    file_name = to_torch_autograd(model_name, func2file_fwd, func2file_bwd, merged_node_fwd, merged_node_bwd)
    file_name = file_name[:-3].replace("/", '.')
    torch_func = importlib.import_module(file_name)
    setattr(model_scope, func_name, torch_func.GenTrainingModel)
