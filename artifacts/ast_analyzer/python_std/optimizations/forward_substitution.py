"""
Replace variable that can be lazy evaluated and used only once by their full
computation code.
"""

from ast_analyzer.python_std.analyses import LazynessAnalysis, UseDefChains, DefUseChains
from ast_analyzer.python_std.analyses import Literals, Ancestors, Identifiers, CFG, IsAssigned
from ast_analyzer.python_std.passmanager import Transformation

from collections import defaultdict
import gast as ast
import networkx as nx

import astunparse

try:
    from math import isfinite
except ImportError:
    from math import isinf, isnan
    def isfinite(x):
        return not isinf(x) and not isnan(x)


class Remover(ast.NodeTransformer):

    def __init__(self, nodes):
        self.nodes = nodes

    def visit_Assign(self, node):
        if node in self.nodes:
            to_prune = self.nodes[node]
            node.targets = [tgt for tgt in node.targets if tgt not in to_prune]
            if node.targets:
                return node
            else:
                return ast.Pass()
        return node


class ForwardSubstitution(Transformation):

    """
    Replace variable that can be computed later.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> pm = passmanager.PassManager("test")

    >>> node = ast.parse("def foo(): a = [2, 3]; builtins.print(a)")
    >>> _, node = pm.apply(ForwardSubstitution, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        pass
        builtins.print([2, 3])

    >>> node = ast.parse("def foo(): a = 2; builtins.print(a + a)")
    >>> _, node = pm.apply(ForwardSubstitution, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        a = 2
        builtins.print((2 + 2))

    >>> node = ast.parse("def foo():\\n a=b=2\\n while a: a -= 1\\n return b")
    >>> _, node = pm.apply(ForwardSubstitution, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        a = 2
        while a:
            a -= 1
        return 2
    """

    def __init__(self):
        """ Satisfy dependencies on others analyses. """
        super(ForwardSubstitution, self).__init__(LazynessAnalysis,
                                                  UseDefChains,
                                                  DefUseChains,
                                                  Ancestors,
                                                  CFG,
                                                  Literals)
        self.to_remove = None

    def visit_FunctionDef(self, node):
        self.to_remove = defaultdict(list)
        self.locals = self.def_use_chains.locals[node]

        # prune some assignment as a second phase, as an assignment could be
        # forward-substituted several times (in the case of constants)
        self.generic_visit(node)
        Remover(self.to_remove).visit(node)
        return node

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            return node

        # OpenMP metdata are not handled by beniget, which is fine in our case
        if node not in self.use_def_chains:
            return node
        defuses = self.use_def_chains[node]
        if node.id == 'i':
            return node

        if len(defuses) != 1:
            return node

        defuse = defuses[0]

        dnode = defuse.node
        if not isinstance(dnode, ast.Name):
            return node

        # multiple definition, which one should we forward?
        # zc: why we need this?
        # if sum(1 for d in self.locals if d.name() == dnode.id) > 1:
        #     return node

        if dnode.id not in self.lazyness_analysis:
            return node

        # either a constant or a value
        fwd = (dnode in self.literals and
               isfinite(self.lazyness_analysis[dnode.id]))
        fwd |= self.lazyness_analysis[dnode.id] == 1

        if not fwd:
            return node

        parent = self.ancestors[dnode][-1]
        if isinstance(parent, ast.Assign):
            value = parent.value
            if dnode in self.literals:
                self.update = True
                if len(defuse.users()) == 1:
                    self.to_remove[parent].append(dnode)
                    return value
                else:
                    # FIXME: deepcopy here creates an unknown node
                    # for alias computations
                    return value
            elif len(parent.targets) == 1:
                ids = self.gather(Identifiers, value)
                node_stmt = next(reversed([s for s in self.ancestors[node]
                                 if isinstance(s, ast.stmt)]))

                bfs_pred = list(dict(nx.bfs_predecessors(self.cfg, parent)).keys())
                bfs_pred.append(parent)
                sg = self.cfg.subgraph(bfs_pred)
                sg = sg.reverse()
                bfs_pred = list(dict(nx.bfs_predecessors(sg, node_stmt)).keys())
                bfs_pred.append(node_stmt)
                sg = sg.subgraph(bfs_pred).reverse()

                all_paths = nx.all_simple_paths(sg, parent, node_stmt)
                for path in all_paths:
                    for stmt in path[1:-1]:
                        assigned_ids = {n.id
                                        for n in self.gather(IsAssigned, stmt)}
                        if not ids.isdisjoint(assigned_ids):
                            break
                    else:
                        continue
                    break
                else:
                    self.update = True
                    self.to_remove[parent].append(dnode)
                    return value

        return node
