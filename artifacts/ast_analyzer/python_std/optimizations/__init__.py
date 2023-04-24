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
from .tensor_to_torch import ToTorchTransform, ToTensorTransform, CopyToAssign