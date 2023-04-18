from   ast_analyzer.shape_inference.types  import *

__all__ = [ 'list_func_ty' ]

def ty_append(ty_args, ty_kwargs):
    ty_list, ty_elem = ty_args
    unify(ty_list, TyList(ty_elem))
    return TyNone()


def ty_reverse(ty_args, ty_kwargs):
    ty_list, = ty_args
    return TyNone()


list_func_ty = {
        list.append  : ty_append,
        list.reverse : ty_reverse,
        }
