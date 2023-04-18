"""
The optimisations submodule contains all the optimisations offered in Pythran.

This file is just for convenience and turns the import from

import optimisations.xxxxx.xxxxx

into

import optimisations.xxxxx
"""

from .loop_full_unrolling import LoopFullUnrolling, UnrollSequential, UnrollTuple, DynamicLoopSplit
from .pattern_transform import PatternTransform, ListToNode
from .replace_obj_const import ReplaceObjConst, ReplaceInferedConst
from .dead_code_elimination import DeadCodeElimination
from .expand_builtins import ExpandBuiltins, RemoveBuiltins
from .to_functional import Functional, OOPStyle
from .tensor_to_torch import ToTorchTransform, ToTensorTransform, CopyToAssign
from .forward_substitution import ForwardSubstitution
from .constant_propagation import ConstantPropagation
from .constant_folding import ConstantFolding, PartialConstantFolding
from .normalize_ifelse import NormalizeIfElse
from .fill_if_def import FillIfDef
from .fill_shape import fill_shape
from .advance_dce import advance_dce
from .zero_tensor_folding import zero_fold