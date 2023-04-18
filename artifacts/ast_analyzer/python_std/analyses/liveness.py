from ast_analyzer.python_std.passmanager import NodeAnalysis
from .use_def_chain import DefUseChains
from .child_nodes import ChildNodes
import gast as ast

def is_live(d, exclude_dfns, node2id):
    for user in d.users():
        user_dfn = node2id[user.node]
        if user_dfn < exclude_dfns[0] or user_dfn >= exclude_dfns[1]:
            return True
    return False


class Liveness(NodeAnalysis):

    def __init__(self):
        self.result = {}
        super(Liveness, self).__init__(DefUseChains, ChildNodes)

    def visit_For(self, node):
        self.generic_visit(node)
        node_info = self.child_nodes.info[node]
        body_defs = set()
        for n in self.child_nodes.id2node[node_info['body'][0]: node_info['body'][1]]:
            if n in self.defs:
                if isinstance(n, ast.Name):
                    n_dfn = self.child_nodes.node2id[n]
                    if is_live(self.def_use_chains.chains[n], (n_dfn, node_info['body'][1]), self.child_nodes.node2id):
                        body_defs.add(n.id)
                else:
                    raise NotImplementedError

        orelse_defs = set()
        for n in self.child_nodes.id2node[node_info['orelse'][0]: node_info['orelse'][1]]:
            if n in self.defs:
                if isinstance(n, ast.Name):
                    n_dfn = self.child_nodes.node2id[n]
                    if is_live(self.def_use_chains.chains[n], (n_dfn, node_info['orelse'][1]), self.child_nodes.node2id):
                        body_defs.add(n.id)
                else:
                    raise NotImplementedError
        self.result[node] = {
            'body': body_defs,
            'orelse': orelse_defs
        }


    def visit_If(self, node):
        self.generic_visit(node)
        node_info = self.child_nodes.info[node]
        body_defs = set()
        for n in self.child_nodes.id2node[node_info['body'][0]: node_info['body'][1]]:
            if n in self.defs:
                if isinstance(n, ast.Name):
                    n_dfn = self.child_nodes.node2id[n]
                    if is_live(self.def_use_chains.chains[n], (n_dfn, node_info['body'][1]), self.child_nodes.node2id):
                        body_defs.add(n.id)
                else:
                    raise NotImplementedError

        orelse_defs = set()
        for n in self.child_nodes.id2node[node_info['orelse'][0]: node_info['orelse'][1]]:
            if n in self.defs:
                if isinstance(n, ast.Name):
                    n_dfn = self.child_nodes.node2id[n]
                    if is_live(self.def_use_chains.chains[n], (n_dfn, node_info['orelse'][1]), self.child_nodes.node2id):
                        orelse_defs.add(n.id)
                else:
                    raise NotImplementedError
        self.result[node] = {
            'body': body_defs,
            'orelse': orelse_defs
        }


    def visit_Module(self, node):
        self.defs = set()
        for n, defs in self.def_use_chains.locals.items():
            for d in defs:
                self.defs.add(d.node)
        self.generic_visit(node)
        # print("liveness", self.result)