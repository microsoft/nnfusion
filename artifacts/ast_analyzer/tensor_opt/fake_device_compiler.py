import gast
import astunparse
import copy
from ast_analyzer.utils.unparse import unparse_ast_list
from ast_analyzer.grad import annotations as anno
from ast_analyzer.grad.cfg import get_def_use, BackwardActive, backward_block
from ast_analyzer.grad.annotate import GatherDefUse
from ast_analyzer.to_onnx.exporter import export_to_onnx_subast, export_to_onnx_subast_simple

class FetchType(gast.NodeVisitor):
    to_ignore = ['torch', 'self', 'math', 'range', 'print', 'int']

    def __init__(self, to_fetch):
        self.to_fetch = set(to_fetch)
        for x in FetchType.to_ignore:
            self.to_fetch.discard(x)
        self.to_fetch_all = copy.deepcopy(self.to_fetch)
        self.result = {}
        self.in_target = 0

    def visit_Assign(self, node):
        self.visit(node.value)
        # self.in_target += 1
        for target in node.targets:
            self.visit(target)
        # self.in_target -= 1

    def visit_Name(self, node):
        if node.id in self.to_fetch:
            # assert(self.in_target == 0)
            if node.id in self.to_fetch_all:
                self.result[node.id] = anno.getanno(node, 'type')
            self.to_fetch.remove(node.id)

class DeviceCompilerWrapper():
    def __init__(self, scope, graph_name, type_dict, platform):
        super(DeviceCompilerWrapper, self).__init__()
        self.scope = scope
        self.graph_name = graph_name
        self.type_dict = type_dict
        self.platform = platform

    def run(self, stmts, args, rets, cfg_nodes, simple_mode, func2name = {}, check_model=True, wrap_recursion=False): # list[ast node]
        print("[unparse_ast_list]")
        print(unparse_ast_list(stmts))
        # print("run", astunparse.unparse(node))
        # print("-----------------------------------")
        if len(stmts) > 0 and isinstance(stmts[-1], gast.Return):
            ret_stmt = stmts[-1]
            stmts = stmts[:-1]
        else:
            ret_stmt = None
        fetch_type = FetchType(args)
        for stmt in stmts:
            fetch_type.visit(stmt)
        assert(len(fetch_type.to_fetch) == 0)  # find the type of all args
        arg_with_type = []
        for name in args:
            if name in fetch_type.result:
                arg_with_type.append((name, fetch_type.result[name]))
        print("[arg_with_type]", [x for x, _ in arg_with_type])
        if simple_mode:
            self.func_name = 'func_' + self.graph_name
            self.file_name, new_ast = export_to_onnx_subast_simple(self.scope, self.type_dict, stmts, self.graph_name,
                            arg_with_type, rets, 'self' in args, self.func_name , cfg_nodes, platform)
            to_import = None
        else:
            new_ast, to_import = export_to_onnx_subast(self.scope, self.type_dict, stmts, self.graph_name,
                            arg_with_type, rets, 'self' in args, '__' + self.graph_name, cfg_nodes, func2name, check_model, wrap_recursion, self.platform)
        if new_ast is not None and ret_stmt is not None:
            new_ast.append(copy.deepcopy(ret_stmt))
        return new_ast, to_import
    
    def live_outs(self, node):
        if isinstance(node, gast.If):
            # "if" nodes do not have annotation "bwd_active_out"
            return anno.getanno(node.test, 'bwd_active_out')
        else:
            return anno.getanno(node, 'bwd_active_out')

    def live_ins(self, node):
        if isinstance(node, gast.If):
            return self.live_ins(node.body[-1])
        else:
            return anno.getanno(node, 'bwd_active_in')
