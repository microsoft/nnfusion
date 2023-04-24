from copy import deepcopy
from enum import Enum, IntEnum

import chainer
import numpy as np

import torch

import gast as ast

from ast_analyzer.shape_inference import utils
from ast_analyzer.shape_inference.shape_elem import *

__all__ = ['TyObj', 'TyNone', 'TyNum', 'TyBool', 'TyInt', 'TyFloat', 'TyString', 'TyArrow', 'TyList', 'TyTuple', 'TyDict', 'TyUserDefinedClass', 'TyDType', 'TyVar', 'TyOptional', 'TyTensor', 'TensorKind', 'TyNdarray', 'TyChainerVariable', 'TyTorchTensor', 'TyTree', 'torch_dtype_to_np_dtype', 'type_of_value', 'extract_value_from_ty', 'lacks_value', 'generate_dummy_value', 'tyobj_to_dtype', 'dtype_to_tyobj', "np_dtype_to_torch_string", 'copy_ty', 'unify', 'UnifyError', 'join', 'JoinError', 'joins', 'is_subtype', 'is_sametype', 'fixed_num_to_ast'
           ]


class TyObj():  # base type, meaning 'unknown'
    def __str__(self):
        assert False, "Not implemented"

    def __repr__(self):
        return self.__str__()

    # dereference internal type
    def deref(self):
        return self

# --------------------------- python primivite types ---------------------------


class TyNone(TyObj):
    def __str__(self):
        return "NoneType"


class NumKind(IntEnum):
    BOOL = 0
    INT = 1
    FLOAT = 2

    def __str__(self):
        if self.value == 0:
            return "bool"
        if self.value == 1:
            return "int"
        if self.value == 2:
            return "float"


class TyNum(TyObj):
    def __init__(self, kind, value=None):
        super().__init__()
        self.kind = kind
        self.value = value

    def __str__(self):
        if self.value is None:
            value_str = ""
        else:
            value_str = "(" + str(self.value) + ")"
        return str(NumKind(self.kind)) + value_str

    def coerce_value(self):
        if self.value is None:
            return
        self.value = eval(str(NumKind(self.kind)))(self.value)

    def is_int(self):
        return self.kind <= NumKind.INT

    def is_float(self):
        return self.kind == NumKind.FLOAT
    
    def is_bool(self):
        return self.kind == NumKind.BOOL


def TyBool(value=None):
    return TyNum(0, value=value)  # bool or int or float


def TyInt(value=None):
    return TyNum(1, value=value)  # int or float


def TyFloat(value=None):
    return TyNum(2, value=value)  # float


class TyString(TyObj):
    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def __str__(self):
        return "string"


class TyArrow(TyObj):
    def __init__(self, argty, retty):
        super().__init__()
        self.argty = argty  # Arguments are uncurried
        self.retty = retty

    def __str__(self):
        if self.argty == []:
            return "(no argument) -> {}".format(self.retty)
        return "arg: " + ' | '.join([str(t) for t in self.argty]) + "ret: " + str(self.retty)
        # return "".join([str(t) + " -> " for t in self.argty]) + str(self.retty)

    def deref(self):
        self.argty = [t.deref() for t in self.argty]
        self.retty = self.retty.deref()
        return self


class TyList(TyObj):
    def __init__(self, ty):
        super().__init__()
        self.ty = ty

    def __str__(self):
        return "{} list".format(self.ty)

    def deref(self):
        self.ty = self.ty.deref()
        return self

    def get(self):
        return self.ty


class TyTuple(TyObj):
    def __init__(self, ty):
        super().__init__()
        self.is_fixed_len = isinstance(ty, list)
        self._ty = ty

    def __str__(self):
        if self.is_fixed_len:
            if len(self._ty) == 1:
                return "(" + str(self._ty[0]) + ",)"
            return "(" + utils.intercalate([str(t) for t in self._ty], ", ") + ")"
        return str(self._ty) + " tuple"

    def __getitem__(self, i):
        assert self.is_fixed_len
        return self._ty[i]

    def size(self):
        if self.is_fixed_len:
            return len(self._ty)
        return None

    def deref(self):
        if self.is_fixed_len is not None:
            if self.is_fixed_len:
                self._ty = [t.deref() for t in self._ty]
            else:
                self._ty = self._ty.deref()
        return self

    def get(self):
        # get one type as a representative
        if self.is_fixed_len:
            ty = joins(self._ty)
            return ty.deref()
        return self.get_ty()

    def get_ty(self):
        assert not self.is_fixed_len
        return self._ty

    def get_tys(self):
        assert self.is_fixed_len
        return self._ty


class TyDict(TyObj):
    def __init__(self, keyty, valty):
        super().__init__()
        self.keyty = keyty
        self.valty = valty

    def __str__(self):
        return "{" + str(self.keyty) + " : " + str(self.valty) + "}"

    def deref(self):
        self.keyty = self.keyty.deref()
        self.valty = self.valty.deref()
        return self


class TyUserDefinedClass(TyObj):
    def __init__(self, name, instance):
        super().__init__()
        self.name = name
        # XXX: we will assume that an instance already exists
        self.instance = instance

    def __str__(self):
        return "class " + self.name


class TyOptional(TyObj):
    def __init__(self, ty):
        super().__init__()
        self.ty = ty

    def __str__(self):
        return "optional(" + str(self.ty) + ")"

# --------------------- numpy ndarray / chainer variable -----------------------


class TensorKind(Enum):
    ndarray = 0
    chainer_variable = 1
    torch_tensor = 2


class TyDType(TyObj):
    def __init__(self, t):
        super().__init__()
        self.t = np.dtype(t)

    def __str__(self):
        return "dtype({})".format(str(self.t))


class TyTensor(TyObj):
    def __init__(self, kind, dtype, shape):  # we do not allow heterogeneous type ndarray
        super().__init__()
        if isinstance(dtype, torch.dtype):
            self.dtype = torch_dtype_to_np_dtype(dtype)
        else:
            self.dtype = np.dtype(dtype)
        self.kind = kind
        self.ndim = len(shape)
        self.shape = wrap_shape(shape)  # Tuple[ShapeElem]

    def __str__(self):
        if self.kind == TensorKind.ndarray:
            return "ndarray({}, {})".format(self.dtype, self.shape)
        if self.kind == TensorKind.chainer_variable:
            return "Variable({}, {})".format(self.dtype, self.shape)
        if self.kind == TensorKind.torch_tensor:
            return "torch.Tensor({}, {})".format(self.dtype, self.shape)

    def is_ndarray(self):
        return self.kind == TensorKind.ndarray

    def is_chainer_variable(self):
        return self.kind == TensorKind.chainer_variable

    def is_torch_tensor(self):
        return self.kind == TensorKind.torch_tensor

    def is_fixed_shape(self):
        if self.shape is None:
            return False
        for x in self.shape:
            if not (x.has_value() and isinstance(x.get_value(), int)):
                return False
        return True

    def is_int(self):
        return self.dtype in (np.bool, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)

    def is_scalar(self):
        return self.is_fixed_shape and self.ndim == 0

    def unwrapped_shape(self):
        return unwrap_shape(self.shape)


class TyTree(TyObj):
    def __init__(self, type_dict):
        self.type_dict = type_dict

    def __str__(self):
        return "Tree({})".format(self.type_dict)


def TyNdarray(dtype, shape=None, ndim=None):
    # ndim and shape cannot be None at the same time
    if shape is None:
        shape = (None,) * ndim
    return TyTensor(TensorKind.ndarray, dtype, shape)


def TyChainerVariable(dtype, shape=None, ndim=None):
    if shape is None:
        shape = (None,) * ndim
    return TyTensor(TensorKind.chainer_variable, dtype, shape)


def TyTorchTensor(dtype, shape=None, ndim=None):
    if shape is None:
        shape = (None,) * ndim
    return TyTensor(TensorKind.torch_tensor, dtype, shape)


def torch_dtype_to_np_dtype(dtype):
    dtype_dict = {
        torch.bool: np.dtype(np.bool),
        torch.uint8: np.dtype(np.uint8),
        torch.int8: np.dtype(np.int8),
        torch.int16: np.dtype(np.int16),
        torch.short: np.dtype(np.int16),
        torch.int32: np.dtype(np.int32),
        torch.int: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.long: np.dtype(np.int64),
        torch.float16: np.dtype(np.float16),
        torch.half: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float: np.dtype(np.float32),
        torch.float64: np.dtype(np.float64),
        torch.double: np.dtype(np.float64),
    }
    return dtype_dict[dtype]


def np_dtype_to_torch_string(dtype):
    dtype_dict = {
        np.bool: "torch.bool",
        np.uint8: "torch.uint8",
        np.int8: "torch.int8",
        np.int16: "torch.short",
        np.int32: "torch.int",
        np.int64: "torch.long",
        np.float16: "torch.half",
        np.float32: "torch.float",
        np.float64: "torch.double"
    }
    return dtype_dict[dtype.type]


# ---------------------- InferenceEngine internal types ------------------------

var_counter = 0


class TyVar(TyObj):
    def __init__(self, lineno=None):
        global var_counter
        super().__init__()
        self.i = var_counter
        var_counter += 1
        self.ty = None
        self.is_set = False
        self.lineno = lineno

    def __str__(self):
        if self.ty is not None:
            return str(self.ty)
        if self.lineno is not None:
            return "a{} (from line {})".format(self.i, self.lineno)
        return "a{}".format(self.i)

    def set(self, ty):
        assert self.is_set == False
        self.is_set = True
        self.ty = ty

    def deref(self):
        if self.is_set:
            return self.ty.deref()
        return self


# ------------------------------------------------------------------------------

def type_of_value(value):
    if value is None:
        return TyNone()
    if isinstance(value, bool):
        return TyBool(value=value)
    if isinstance(value, int):
        return TyInt(value=value)
    if isinstance(value, float):
        return TyFloat(value=value)
    if isinstance(value, str):
        return TyString(value=value)
    if isinstance(value, list):
        return TyList(joins([type_of_value(v) for v in value]))
    if isinstance(value, range):
        return TyList(joins([type_of_value(v) for v in value]))
    if isinstance(value, enumerate):
        return TyList(joins([type_of_value(v) for v in value]))
    if isinstance(value, zip):
        return TyList(joins([type_of_value(v) for v in value]))
    if isinstance(value, tuple) or isinstance(value, torch.nn.ModuleList):
        return TyTuple([type_of_value(v) for v in value])
    if isinstance(value, dict):
        if len(value) == 0:
            return TyDict(TyVar(), TyVar())
        return TyDict(type_of_value(list(value.keys())[0]),
                      type_of_value(list(value.items())[0]))
    if isinstance(value, np.ndarray):
        return TyNdarray(value.dtype, shape=wrap_shape(value.shape))
    if isinstance(value, chainer.Variable):
        return TyChainerVariable(value.dtype, shape=wrap_shape(value.shape))
    if isinstance(value, torch.Tensor):
        return TyTorchTensor(value.dtype, shape=wrap_shape(value.shape))
    if isinstance(value, np.dtype):
        return TyDType(value)
    if isinstance(value, type) and value in np.typeDict.values():
        # XXX: np.typeDict.values() is a list of all dtypes
        return TyDType(value)
    if isinstance(value, torch.dtype):
        return TyDType(torch_dtype_to_np_dtype(value))
    if isinstance(value, ShapeElem):
        if isinstance(value.value, int):
            return TyInt(value.value)
        return TyInt()

    return TyUserDefinedClass(type(value).__name__, value)


def lacks_value(ty) -> bool:
    ty = ty.deref()

    if isinstance(ty, TyNone):
        return False
    if isinstance(ty, TyNum):
        return ty.value is None
    if isinstance(ty, TyString):
        return ty.value is None
    if isinstance(ty, TyList):
        return True
    if isinstance(ty, TyTuple):
        if not ty.is_fixed_len:
            return True
        return any([lacks_value(t) for t in ty.get_tys()])
    if isinstance(ty, TyDict):
        return True
    if isinstance(ty, TyTensor):
        return ty.shape is None or any([not i.has_value() for i in ty.shape])
    if isinstance(ty, TyDType):
        return ty.t is None


def generate_dummy_value(ty) -> object:
    # creates dummy value

    ty = ty.deref()

    if isinstance(ty, TyNone):
        return None
    if isinstance(ty, TyNum):
        if ty.value is not None:
            return ty.value
        # XXX: use 1 to avoid division by zero
        return eval(str(NumKind(ty.kind)))(1)
    if isinstance(ty, TyString):
        if ty.value is not None:
            return ty.value
        return ""
    if isinstance(ty, TyList):
        return [generate_dummy_value(ty.ty)]
    if isinstance(ty, TyTuple):
        if ty.is_fixed_len:
            return tuple([generate_dummy_value(t) for t in ty.get_tys()])
        return tuple([generate_dummy_value(ty.get_ty())])
    if isinstance(ty, TyDict):
        return {generate_dummy_value(ty.keyty): generate_dummy_value(ty.valty)}
    if isinstance(ty, TyTensor):
        ret = np.zeros(dtype=ty.dtype, shape=unwrap_shape(ty.shape))
        if ty.is_ndarray():
            return ret
        if ty.is_chainer_variable():
            return chainer.Variable(ret)
        if ty.is_torch_tensor():
            return torch.as_tensor(ret)
    if isinstance(ty, TyDType):
        return ty.t
    if isinstance(ty, TyUserDefinedClass):
        # We don't need to copy the instance because it won't be overwritten
        return ty.instance

    assert False, "generate_dummy_value: type not understood: " + \
        str(ty) + "(" + str(type(ty)) + ")"


def extract_value_from_ty(ty):
    # returns None where it doesn't have value
    ty = ty.deref()

    if isinstance(ty, TyNone):
        return None
    if isinstance(ty, TyNum):
        if ty.value is not None:
            return ty.value
        return None
    if isinstance(ty, TyString):
        if ty.value is not None:
            return ty.value
        return None
    if isinstance(ty, TyTuple):
        if not ty.is_fixed_len:
            return None
        return tuple([extract_value_from_ty(t) for t in ty.get_tys()])
    if isinstance(ty, TyDict):
        return None
    if isinstance(ty, TyTensor):
        return None
    if isinstance(ty, TyDType):
        return ty.t

    assert False, "extract_value_from_ty: type not understood: " + str(ty)


def copy_ty(ty):
    if isinstance(ty, (TyNone, TyNum, TyString)):
        return deepcopy(ty)
    if isinstance(ty, TyArrow):
        return TyArrow([copy_ty(t) for t in ty.argty], copy_ty(ty.retty))
    if isinstance(ty, TyList):
        return TyList(copy_ty(ty.ty))
    if isinstance(ty, TyTuple):
        if ty.is_fixed_len:
            return TyTuple([copy_ty(t) for t in ty.get_tys()])
        return TyTuple(copy_ty(ty.get_ty()))
    if isinstance(ty, TyDict):
        return TyDict(copy_ty(ty.keyty), copy_ty(ty.valty))
    if isinstance(ty, TyTree):
        return TyTree(
            dict((attr, copy_ty(attr_ty))
                 for attr, attr_ty in ty.type_dict.items())
        )
    if isinstance(ty, TyUserDefinedClass):
        return TyUserDefinedClass(ty.name, ty.instance)
    if isinstance(ty, TyDType):
        return TyDType(ty.t)
    if isinstance(ty, TyTensor):
        return TyTensor(ty.kind, ty.dtype, ty.shape)
    if isinstance(ty, TyVar):
        if ty.ty is not None:
            return ty.deref()
        return ty
    if isinstance(ty, TyOptional):
        return TyOptional(ty.ty)
    assert False, "copy_ty: {}".format(ty)


def tyobj_to_dtype(ty):
    assert isinstance(ty, TyNum), "tyobj_to_dtype: Unknown dtype"
    return np.dtype(str(NumKind(ty.kind)))


def dtype_to_tyobj(dtype):
    if dtype.kind == 'b':
        return TyBool()
    if dtype.kind in 'iu':
        return TyInt()
    if dtype.kind == 'f':
        return TyFloat()
    assert False


# ==============================================================================

class UnifyError(Exception):
    def __init__(self, ty1, ty2):
        self.msg = "UnifyError: {} and {} are not unifiable".format(ty1, ty2)


def occur(var, ty):
    if isinstance(ty, TyVar):
        if var is ty:
            return True
        return occur(var, ty.ty)
    if isinstance(ty, TyArrow):
        return any([occur(var, t) for t in ty.argty]) or occur(var, ty.retty)
    if isinstance(ty, TyList):
        return occur(var, ty.ty)
    if isinstance(ty, TyTuple):
        if ty.is_fixed_len:
            return any([occur(var, t) for t in ty.get_tys()])
        return occur(var, ty.get_ty())
    if isinstance(ty, TyDict):
        return occur(var, ty.keyty) or occur(var, ty.valty)
    return False


# Solves constraint over type variables and the element of TyList,
# which needs a special care since it is an invariant type
def unify(ty1, ty2):
    ty1 = ty1.deref()
    ty2 = ty2.deref()

    if isinstance(ty1, TyNone) and isinstance(ty2, TyNone):
        return

    if isinstance(ty1, TyOptional) and isinstance(ty2, TyOptional):
        unify(ty1.ty, ty2.ty)
        return

    if isinstance(ty1, TyVar):
        if isinstance(ty2, TyVar) and ty1 is ty2:
            return ty1
        if occur(ty1, ty2):
            raise UnifyError(ty1, ty2)
        ty1.set(ty2)
        return ty2

    if isinstance(ty2, TyVar):
        if occur(ty2, ty1):
            raise UnifyError(ty1, ty2)
        ty2.set(ty1)
        return ty1

    if (isinstance(ty1, TyNum) and isinstance(ty2, TyNum) and
            ty1.kind == ty2.kind):
        return

    if isinstance(ty1, TyString) and isinstance(ty2, TyString):
        return

    if isinstance(ty1, TyList) and isinstance(ty2, TyList):
        unify(ty1.ty, ty2.ty)
        return

    if isinstance(ty1, TyTuple) and isinstance(ty2, TyTuple):
        if ty1.is_fixed_len and ty2.is_fixed_len and \
                len(ty1.get_tys()) == len(ty2.get_tys()):
            for t1, t2 in zip(ty1.get_tys(), ty2.get_tys()):
                unify(t1, t2)
            return
        unify(ty1.get(), ty2.get())
        return

    if isinstance(ty1, TyDict) and isinstance(ty2, TyDict):
        unify(ty1.keyty, ty2.keyty)
        unify(ty1.valty, ty2.valty)
        return

    if isinstance(ty1, TyTensor) and isinstance(ty2, TyTensor):
        utils.set_attr_if_None(ty1, ty2, 'kind')

        if ty1.dtype == ty2.dtype and ty1.ndim == ty2.ndim:
            return

    if isinstance(ty1, TyTensor) and isinstance(ty2, TyNum):
        if ty1.ndim == 0:
            return

    if isinstance(ty1, TyNum) and isinstance(ty2, TyTensor):
        if ty2.ndim == 0:
            return

    if isinstance(ty1, TyDType) and isinstance(ty2, TyDType):
        assert ty1.t == ty2.t
        return

    if isinstance(ty1, TyUserDefinedClass) and \
            isinstance(ty2, TyUserDefinedClass):
        if ty1.name == ty2.name:
            return
        # TODO(momohatt): Find least common superclass and check that
        # it is not 'object'
        if isinstance(ty1.instance, torch.nn.Module) and \
                isinstance(ty2.instance, torch.nn.Module):
            return

    raise UnifyError(ty1, ty2)


# return True iff ty1 <= ty2
def is_subtype(ty1, ty2):
    if isinstance(ty1, TyNone) and isinstance(ty2, (TyNone, TyOptional)):
        return True
    if isinstance(ty1, TyOptional) and isinstance(ty2, TyOptional):
        return is_subtype(ty1.ty, ty2.ty)
    if isinstance(ty1, TyVar) and ty1 is ty2:
        return True
    if isinstance(ty1, TyNum) and isinstance(ty2, TyNum):
        return ty1.kind <= ty2.kind
    if isinstance(ty1, TyString) and isinstance(ty2, TyString):
        return True
    if isinstance(ty1, TyList) and isinstance(ty2, TyList):
        return is_subtype(ty1.ty, ty2.ty) and is_subtype(ty2.ty, ty1.ty)
    if isinstance(ty1, TyTuple) and isinstance(ty2, TyTuple):
        if ty1.is_fixed_len:
            if ty2.is_fixed_len and len(ty1.get_tys()) == len(ty2.get_tys()):
                return all([is_subtype(t1, t2)
                            for t1, t2 in zip(ty1.get_tys(), ty2.get_tys())])
            return all([is_subtype(t1, ty2.get_ty()) for t1 in ty1.get_tys()])
        if ty2.is_fixed_len:
            return False
        return is_subtype(ty1.get_ty(), ty2.get_ty())
    if isinstance(ty1, TyDict) and isinstance(ty2, TyDict):
        return (is_subtype(ty1.keyty, ty2.keyty) and
                is_subtype(ty2.keyty, ty1.keyty) and
                is_subtype(ty1.valty, ty2.valty) and
                is_subtype(ty2.valty, ty1.valty))
    if isinstance(ty1, TyTensor) and isinstance(ty2, TyTensor):
        if (isinstance(ty1, TyTensor) and isinstance(ty2, TyTensor) and
                ty1.kind == ty2.kind and ty1.dtype == ty2.dtype and
                ty1.ndim == ty2.ndim):
            return is_subshape(ty1.shape, ty2.shape)
    if isinstance(ty1, TyDType) and isinstance(ty2, TyDType):
        return ty1.t == ty2.t
    if (isinstance(ty1, TyUserDefinedClass) and
            isinstance(ty2, TyUserDefinedClass) and
            ty1.name == ty2.name):
        return True
    return False


def is_sametype(ty1, ty2):
    return is_subtype(ty1, ty2) and is_subtype(ty2, ty1)


class JoinError(Exception):
    def __init__(self, ty1, ty2):
        self.msg = "JoinError: {} and {} are not joinable".format(ty1, ty2)


def join(ty1, ty2):
    ty1 = ty1.deref()
    ty2 = ty2.deref()

    if isinstance(ty1, TyNone) and isinstance(ty2, TyNone):
        return TyNone()

    if isinstance(ty1, TyNone):
        if isinstance(ty2, TyOptional):
            return ty2
        return TyOptional(ty2)
    if isinstance(ty2, TyNone):
        if isinstance(ty1, TyOptional):
            return ty1
        return TyOptional(ty1)

    if isinstance(ty1, TyOptional) and isinstance(ty2, TyOptional):
        return TyOptional(join(ty1.ty, ty2.ty))
    if isinstance(ty1, TyOptional):
        return TyOptional(join(ty1.ty, ty2))
    if isinstance(ty2, TyOptional):
        return TyOptional(join(ty1, ty2.ty))

    if isinstance(ty1, TyVar) and isinstance(ty2, TyVar):
        if ty1 is ty2:
            return ty1

    if isinstance(ty1, TyVar):
        return ty2
    if isinstance(ty2, TyVar):
        return ty1

    if isinstance(ty1, TyNum) and isinstance(ty2, TyNum):
        kind = max(ty1.kind, ty2.kind)
        if ty1.value == ty2.value:
            retty = TyNum(kind, ty1.value)
            retty.coerce_value()
            return retty
        return TyNum(kind)

    if isinstance(ty1, TyString) and isinstance(ty2, TyString):
        if ty1.value == ty2.value:
            return TyString(ty1.value)
        return TyString()

    if (isinstance(ty1, TyList) and isinstance(ty2, TyList) and
            is_subtype(ty1, ty2)):
        return ty1

    if isinstance(ty1, TyTuple) and isinstance(ty2, TyTuple):
        if ty1.is_fixed_len and ty2.is_fixed_len:
            if len(ty1.get_tys()) == len(ty2.get_tys()):
                return TyTuple([join(t1, t2)
                                for t1, t2 in zip(ty1.get_tys(), ty2.get_tys())])
        return TyTuple(join(ty1.get(), ty2.get()))

    if (isinstance(ty1, TyDict) and isinstance(ty2, TyDict) and
            is_subtype(ty1, ty2)):
        return ty1

    if isinstance(ty1, TyTensor) and isinstance(ty2, TyTensor):
        if ty1.kind == ty2.kind and ty1.dtype == ty2.dtype and \
                ty1.ndim == ty2.ndim:
            return TyTensor(ty1.kind, ty1.dtype, join_shape(ty1.shape, ty2.shape))

    if isinstance(ty1, TyDType) and isinstance(ty2, TyDType):
        if ty1.t == ty2.t:
            return TyDType(ty1.t)

    if isinstance(ty1, TyTree) and isinstance(ty2, TyTree):
        if ty1.type_dict.keys() == ty2.type_dict.keys():
            type_dict = {}
            for k in ty1.type_dict.keys():
                type_dict[k] = join(ty1.type_dict[k], ty2.type_dict[k])
            return TyTree(type_dict)

    if isinstance(ty1, TyUserDefinedClass) and \
            isinstance(ty2, TyUserDefinedClass):
        if ty1.name == ty2.name and ty1.instance is ty2.instance:
            return TyUserDefinedClass(ty1.name, ty1.instance)

    raise JoinError(ty1, ty2)


def joins(tys):
    return utils.foldl(join, TyVar(), tys)

def fixed_num_to_ast(ty):
    if isinstance(ty, TyNum) and ty.value is not None:
        return ast.Constant(value=ty.value, kind=None)
    if isinstance(ty, TyNone):
        return ast.Constant(value=None, kind=None)
    if isinstance(ty, TyTuple):
        if ty.size() is not None:
            ele_nodes = []
            for t in ty.get_tys():
                node = fixed_num_to_ast(t)
                if node is None: return None
                ele_nodes.append(node)
            return ast.Tuple(elts=ele_nodes, ctx=ast.Load())
    return None
