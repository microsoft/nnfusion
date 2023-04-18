from   chainer.utils import type_check

from   ast_analyzer.shape_inference.shape_elem  import *
from   ast_analyzer.shape_inference.types       import *
from   ast_analyzer.shape_inference             import utils


__all__ = [ 'match_types', 'MatchFail', 'apply_subst' ]


class MatchFail(Exception):
    def __init__(self, ty1, ty2):
        self.msg = "MatchFail: couldn't match {} and {}".format(ty1, ty2)


def match_types(tys1, tys2):
    subst = {}
    for t1, t2 in zip(tys1, tys2):
        t1 = apply_subst(subst, t1)
        t2 = apply_subst(subst, t2)
        utils.add_dict(subst, match_type(t1, t2))
    return subst


def match_type(ty1, ty2):
    assert not isinstance(ty1, (TyVar, TyArrow))
    assert not isinstance(ty2, (TyVar, TyArrow))

    if isinstance(ty1, TyNone) and isinstance(ty2, TyNone):
        return {}

    if isinstance(ty1, TyNum) and isinstance(ty2, TyNum):
        ty1.kind = ty2.kind = max(ty1.kind, ty2.kind)
        ty1.coerce_value()
        ty2.coerce_value()
        return {}

    if isinstance(ty1, TyString) and isinstance(ty2, TyString):
        return {}

    if isinstance(ty1, TyList) and isinstance(ty2, TyList):
        return match_types(ty1.ty, ty2.ty)

    if isinstance(ty1, TyTuple) and isinstance(ty2, TyTuple):
        assert ty1.is_fixed_len and ty2.is_fixed_len
        if len(ty1.get_tys()) == len(ty2.get_tys()):
            return match_types(ty1.get_tys(), ty2.get_tys())

    if isinstance(ty1, TyDict) and isinstance(ty2, TyDict):
        return match_types([ty1.keyty, ty1.valty], [ty2.keyty, ty2.valty])

    if isinstance(ty1, TyTensor) and isinstance(ty2, TyTensor):
        if ty1.dtype == ty2.dtype and ty1.ndim == ty2.ndim:
            try:
                return match_shape(ty1.shape, ty2.shape)
            except Exception:
                raise MatchFail(ty1, ty2)

    if isinstance(ty1, TyDType) and isinstance(ty2, TyDType):
        if ty1.t == ty2.t:
            return {}

    if isinstance(ty1, TyTree) and \
            isinstance(ty2, TyUserDefinedClass):
        return match_types(
            [ty1.type_dict[attr] for attr in ty1.type_dict.keys()],
            [type_of_value(getattr(ty2.instance, attr)) for attr in ty1.type_dict.keys()]
        )

    if isinstance(ty1, TyUserDefinedClass) and \
            isinstance(ty2, TyTree):
        return match_types(
            [type_of_value(getattr(ty1.instance, attr)) for attr in ty2.type_dict.keys()],
            [ty2.type_dict[attr] for attr in ty2.type_dict.keys()]
        )

    if isinstance(ty1, TyTree) and isinstance(ty2, TyTree):
        if ty1.type_dict.keys() != ty2.type_dict.keys():
            return MatchFail(ty1, ty2)
        keys = list(ty1.type_dict.keys())
        return match_types(
            [ty1.type_dict[k] for k in keys],
            [ty2.type_dict[k] for k in keys]
        )
    
    if isinstance(ty1, TyUserDefinedClass) and \
            isinstance(ty2, TyUserDefinedClass):
        if ty1.name == ty2.name:
            return {}

    raise MatchFail(ty1, ty2)


def match_shape(shape1, shape2):
    subst = {}
    for e1, e2 in zip(shape1, shape2):
        e2 = apply_subst_shapeElem(subst, e2)
        if isinstance(e2.value, str):
            subst[e2.value] = None
        elif e2.is_null():
            subst[e2.value] = e2.value # None
        else:
            assert e1.value == e2.value
    return subst


def apply_subst(subst, ty):
    if isinstance(ty, TyList):
        return TyList(apply_subst(subst, ty.ty))

    if isinstance(ty, TyTuple):
        if ty.is_fixed_len:
            return TyTuple([apply_subst(subst, t) for t in ty.get_tys()])
        else:
            return TyTuple(apply_subst(subst, ty.get_ty()))

    if isinstance(ty, TyDict):
        return TyDict(apply_subst(subst, ty.keyty),
                apply_subst(subst, ty.valty))

    if isinstance(ty, TyTensor):
        return TyTensor(ty.kind, ty.dtype, apply_subst_shape(subst, ty.shape))

    return ty


def apply_subst_shapeElem(subst, e):
    if e.value in subst.keys():
        return ShapeElem(subst[e.value], expr=type_check.Variable(None, e.value))
    return e


def apply_subst_shape(subst, shape):
    return [apply_subst_shapeElem(subst, e) for e in shape]
