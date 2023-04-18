"""The analyses submodule contains all the analyses passes offered in Pythran.

This file is just for convenience and turns the import from

import analyses.foo.Foo

into

import analyses.Foo
"""

from .aliases import Aliases, StrictAliases
from .ancestors import Ancestors, AncestorsWithBody
from .argument_effects import ArgumentEffects
from .ast_matcher import ASTMatcher, AST_any, AST_or, Placeholder, Check
from .global_declarations import GlobalDeclarations
from .global_effects import GlobalEffects
from .globals_analysis import Globals
from .has_return import HasReturn, HasBreak, HasContinue
from .locals_analysis import Locals
from .node_count import NodeCount
from .pure_expressions import PureExpressions
from .use_def_chain import DefUseChains, UseDefChains
from .constant_expressions import ConstantExpressions
from .cfg import CFG
from .identifiers import Identifiers
from .is_assigned import IsAssigned
from .lazyness_analysis import LazynessAnalysis
from .literals import Literals
from .child_nodes import ChildNodes
from .liveness import Liveness