from ast_analyzer.python_std.passmanager import NodeAnalysis
import gast as ast

class ChildNodesResult:
    def __init__(self):
        self.id2node = []
        self.node2id = {}
        self.info = {}

    def next_dfn(self):
        return len(self.id2node)

class ChildNodes(NodeAnalysis):
    def __init__(self):
        self.result = ChildNodesResult()
        super(ChildNodes, self).__init__()

    def record(self, node):
        dfn = self.result.next_dfn()
        self.result.id2node.append(node)
        self.result.node2id[node] = dfn

    def visit_If(self, node):
        body_dfn_left = self.result.next_dfn()
        for stmt in node.body:
            self.visit(stmt)
        body_dfn_right = self.result.next_dfn()
        
        orelse_dfn_left = self.result.next_dfn()
        for stmt in node.orelse:
            self.visit(stmt)
        orelse_dfn_right = self.result.next_dfn()

        self.visit(node.test)

        self.record(node)

        self.result.info[node] = {
            'body': (body_dfn_left, body_dfn_right),
            'orelse': (orelse_dfn_left, orelse_dfn_right),
            'node': (body_dfn_left, self.result.next_dfn())
        }

    def visit_For(self, node):
        body_dfn_left = self.result.next_dfn()
        for stmt in node.body:
            self.visit(stmt)
        body_dfn_right = self.result.next_dfn()
        
        orelse_dfn_left = self.result.next_dfn()
        for stmt in node.orelse:
            self.visit(stmt)
        orelse_dfn_right = self.result.next_dfn()

        self.visit(node.target)
        self.visit(node.iter)

        self.record(node)

        self.result.info[node] = {
            'body': (body_dfn_left, body_dfn_right),
            'orelse': (orelse_dfn_left, orelse_dfn_right),
            'node': (body_dfn_left, self.result.next_dfn())
        }

    def visit_Assign(self, node):
        # value -> target, for correct liveness analysis
        self.visit(node.value)
        for t in node.targets:
            self.visit(t)
        self.record(node)

    def visit(self, node):
        if isinstance(node, ast.If):
            return self.visit_If(node)
        elif isinstance(node, ast.For):
            return self.visit_For(node)
        elif isinstance(node, ast.Assign):
            return self.visit_Assign(node)
        else:
            self.generic_visit(node)
            self.record(node)