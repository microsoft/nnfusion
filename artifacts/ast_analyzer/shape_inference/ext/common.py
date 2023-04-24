from ast_analyzer.shape_inference.ext.utils import *
from ast_analyzer.shape_inference.types import *
from ast_analyzer.shape_inference.shape_elem import *

__all__ = ['ty_MakeTensor', 'ty_Shape', 'ty_Size', 'ty_DType', 'ty_TensorArith', 'ty_TensorCeil', 'ty_TensorToList'
           ]


def ty_Shape(ty_obj):
    return TyTuple([TyInt(e.value) for e in ty_obj.shape])


def ty_Size(ty_obj):
    size = size_of_shape(ty_obj.shape)
    return TyInt(size.value)


def ty_DType(ty_obj):
    return TyDType(ty_obj.dtype)


class ty_MakeTensor():
    def calculate_shape(self, x_type):
        if not isinstance(x_type, TyList):
            return ()
        return (None,) + self.calculate_shape(x_type.ty)

    def get_element_dtype(self, ty):
        # get element dtype of nested TyList
        if isinstance(ty, TyList):
            return self.get_element_dtype(ty.ty)
        if isinstance(ty, TyTensor):
            return ty.dtype
        return tyobj_to_dtype(ty)


class ty_TensorArith():
    def __init__(self, kind, op):
        self.kind = kind
        self.op = op

    def __call__(self, ty_args, ty_kwargs):
        x_type, y_type = ty_args
        x_shape = x_type.shape if isinstance(x_type, TyTensor) else ()
        y_shape = y_type.shape if isinstance(y_type, TyTensor) else ()
        dtype = self.get_out_dtype(x_type, y_type, self.op)

        if len(x_shape) > len(y_shape):
            shape = self.infer_return_shape(x_shape, y_shape)
        else:
            shape = self.infer_return_shape(y_shape, x_shape)
        return TyTensor(self.kind, dtype, shape)

    def infer_return_shape(self, x_shape, y_shape):
        ret_shape = [None for _ in x_shape]
        for i in range(len(x_shape) - len(y_shape)):
            ret_shape[i] = copy_ShapeElem(x_shape[i])
        for i in range(1, len(y_shape) + 1):
            if x_shape[-i] == y_shape[-i]:
                if x_shape[-i].value is None:
                    ret_shape[-i] = copy_ShapeElem(y_shape[-i])
                elif y_shape[-i].value is None:
                    ret_shape[-i] = copy_ShapeElem(x_shape[-i])
                elif size_of_ShapeElem(x_shape[-i]) < size_of_ShapeElem(y_shape[-i]):
                    ret_shape[-i] = copy_ShapeElem(y_shape[-i])
                else:
                    ret_shape[-i] = copy_ShapeElem(x_shape[-i])
            elif x_shape[-i] == 1:
                ret_shape[-i] = copy_ShapeElem(y_shape[-i])
            elif y_shape[-i] == 1:
                ret_shape[-i] = copy_ShapeElem(x_shape[-i])
            else:
                assert False
        return ret_shape

    def get_out_dtype(self, x_type, y_type, op):
        if isinstance(x_type, TyTensor):
            x_type = TyTensor(x_type.kind, x_type.dtype, (1,))
        if isinstance(y_type, TyTensor):
            y_type = TyTensor(y_type.kind, y_type.dtype, (1,))
        x = generate_dummy_value(x_type)
        y = generate_dummy_value(y_type)
        return op(x, y).dtype


class ty_TensorCeil():
    def __init__(self, kind):
        self.kind = kind

    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        dtype = self.get_out_dtype(x_type)
        return TyTensor(self.kind, dtype, x_type.shape)

    def get_out_dtype(self, x_type):
        # Behavior of ceil is the same in np and torch
        x = np.array(1, dtype=x_type.dtype)
        return np.ceil(x).dtype


class ty_TensorToList():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        return self.get_out_type(x_type.dtype, x_type.shape)

    def get_out_type(self, dtype, shape):
        if shape == ():
            return dtype_to_tyobj(dtype)
        return TyList(self.get_out_type(dtype, shape[1:]))
