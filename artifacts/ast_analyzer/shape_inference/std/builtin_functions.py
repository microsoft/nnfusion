from  ast_analyzer.shape_inference.types  import *

__all__ = [ 'builtin_func_ty' ]


def ty_len(ty_args, ty_kwargs):
    x_type, = ty_args
    if isinstance(x_type, TyList):
        return TyInt()
    if isinstance(x_type, TyTuple):
        return TyInt(x_type.size())
    if isinstance(x_type, TyTensor):
        return TyInt(x_type.shape[0].value)
    if isinstance(x_type, TyUserDefinedClass):
        assert hasattr(x_type.instance, '__len__')
        return TyInt(len(x_type.instance))
    assert False


def ty_range(ty_args, ty_kwargs):
    return TyList(TyInt())


def ty_int(ty_args, ty_kwargs):
    print("call ty_int")
    x = ty_args[0]
    if not isinstance(x, TyNum):
        raise NotImplementedError
    if x.value is not None:
        return TyInt(int(x.value))
    else:
        return TyInt()


builtin_func_ty = {
        len   : ty_len,
        range: ty_range,
        int:   ty_int,
        }
