"""The analyses submodule contains all the analyses passes offered in Pythran.

This file is just for convenience and turns the import from

import analyses.foo.Foo

into

import analyses.Foo
"""

from .ast_matcher import ASTMatcher, AST_any, AST_or, Placeholder, Check
from .has_return import HasReturn, HasBreak, HasContinue
from .node_count import NodeCount