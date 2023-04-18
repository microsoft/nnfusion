import numpy as np
import math

from   ast_analyzer.shape_inference.ext.common import *
from   ast_analyzer.shape_inference.ext.utils  import *
from   ast_analyzer.shape_inference.types      import *

__all__ = [ 'numpy_attr_ty', 'numpy_func_ty' ]


class ty_NumpyAstype():
    def __call__(self, ty_args, ty_kwargs):
        x_type, dtype_type = ty_args
        if isinstance(dtype_type, TyString):
            return TyNdarray(np.dtype(dtype_type.value), shape=x_type.shape)
        return TyNdarray(dtype_type.t, shape=x_type.shape)


class ty_NumpyArray(ty_MakeTensor):
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        default_dtype = self.get_element_dtype(x_type)
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', default_dtype)

        if isinstance(x_type, TyTensor):
            return TyNdarray(dtype, x_type.shape)

        assert not lacks_dtype, "numpy.array: dtype couldn't be inferred"

        return TyNdarray(dtype, shape=self.calculate_shape(x_type))


class ty_NumpyIdentical():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        return copy_ty(x_type)


class ty_NumpyOnes():
    def __call__(self, ty_args, ty_kwargs):
        shape_type, = ty_args
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', np.dtype('float64'))

        assert not lacks_dtype

        if isinstance(shape_type, TyNum):
            assert shape_type.is_int()
        else:
            assert shape_type.is_fixed_len

        shape = extract_value_from_ty(shape_type)
        if isinstance(shape, int):
            shape = (shape,)

        return TyNdarray(dtype, shape=shape)


class ty_NumpyFull():
    def __call__(self, ty_args, ty_kwargs):
        shape_type, value_type = ty_args
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', tyobj_to_dtype(value_type))

        assert not lacks_dtype

        assert isinstance(shape_type, (TyNum, TyTuple))

        shape = extract_value_from_ty(shape_type)
        if not isinstance(shape_type, TyTuple):
            shape = (shape,)
        return TyNdarray(dtype, shape=shape)


numpy_attr_ty = {
        'shape'  : ty_Shape,
        'size'   : ty_Size,
        'dtype'  : ty_DType,
        }

numpy_func_ty = {
        np.ndarray.astype : ty_NumpyAstype(),
        np.ndarray.tolist : ty_TensorToList(),
        np.array          : ty_NumpyArray(),
        np.cumsum         : ty_NumpyIdentical(),
        np.full           : ty_NumpyFull(),
        np.ones           : ty_NumpyOnes(),
        np.zeros          : ty_NumpyOnes(),
        np.ceil           : ty_TensorCeil(TensorKind.ndarray),
        }
