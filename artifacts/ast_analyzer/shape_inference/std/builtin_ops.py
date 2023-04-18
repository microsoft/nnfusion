import gast

from   ast_analyzer.shape_inference.types      import *
from   ast_analyzer.shape_inference.ext.common import ty_TensorArith

__all__ = [ 'ty_ops' ]


def ty_ops(op, tyl, tyr):
    if isinstance(tyl, TyList) and isinstance(tyr, TyNum):
        assert tyr.is_int()
        assert isinstance(op, gast.Mult)
        return copy_ty(tyl)

    if isinstance(tyl, TyTuple) and isinstance(tyr, TyNum):
        assert tyr.is_int()
        assert isinstance(op, gast.Mult)
        if tyr.value is None:
            return TyTuple(tyl.get())
        return TyTuple([tyl.get() for _ in range(tyr.value)])

    if isinstance(tyl, TyList) and isinstance(tyr, TyList):
        assert isinstance(op, gast.Add)
        return TyList(join(tyl.ty, tyr.ty))

    if isinstance(tyl, TyTuple) and isinstance(tyr, TyTuple):
        assert isinstance(op, gast.Add)
        if tyl.is_fixed_len and tyr.is_fixed_len:
            return TyTuple(tyl.get_tys() + tyr.get_tys())
        return TyTuple(join(tyl.get(), tyr.get()))

    if isinstance(tyl, TyString) and (tyr, TyString):
        if tyl.value is not None and tyr.value is not None:
            return TyString(tyl.value + tyr.value)
        return TyString()

    semantics = {
            gast.Add : (lambda x, y: x + y),
            gast.Sub : (lambda x, y: x - y),
            gast.Mult : (lambda x, y: x * y),
            gast.Div : (lambda x, y: x / y),
            gast.FloorDiv : (lambda x, y: x // y),
            gast.Pow: (lambda x, y: x ** y),
            gast.Mod: (lambda x, y: x % y),
            gast.BitOr: (lambda x, y: x | y),
            gast.BitAnd: (lambda x, y: x & y),
            }
    func = semantics[type(op)]

    if isinstance(tyl, TyTensor):
        return ty_TensorArith(tyl.kind, func)([tyl, tyr], {})
    if isinstance(tyr, TyTensor):
        return ty_TensorArith(tyr.kind, func)([tyl, tyr], {})

    assert isinstance(tyl, TyNum) and isinstance(tyr, TyNum)

    vall, valr = generate_dummy_value(tyl), generate_dummy_value(tyr)
    ty_ret = type_of_value(func(vall, valr))
    if tyl.value is None or tyr.value is None:
        ty_ret.value = None
    return ty_ret
