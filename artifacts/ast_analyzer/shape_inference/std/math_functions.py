from   ast_analyzer.shape_inference.types  import *
import math

__all__ = [ 'math_func_ty' ]

def ty_sqrt(ty_args, ty_kwargs):
    assert(len(ty_args) == 1)
    assert(len(ty_kwargs) == 0)
    arg = ty_args[0]
    assert(isinstance(arg, TyNum))
    if arg.value is not None:
        return TyFloat(math.sqrt(arg.value))
    else:
        return TyFloat()


math_func_ty = {
    math.sqrt  : ty_sqrt,
}
