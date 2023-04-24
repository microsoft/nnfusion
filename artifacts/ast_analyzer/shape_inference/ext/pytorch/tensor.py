import math
import numpy as np

from chainer.utils import type_check

from ast_analyzer.shape_inference.ext.common import *
from ast_analyzer.shape_inference.ext.utils import *
from ast_analyzer.shape_inference.types import *
from ast_analyzer.shape_inference.ext.pytorch.utils import *

__all__ = ['ty_TorchIsTensor', 'ty_TorchIdentical', 'ty_TorchTensor', 'ty_TorchTensorOfShape', 'ty_TorchFromNumpy', 'ty_TorchRandint', 'ty_TorchCat', 'ty_TorchChunk', 'ty_TorchExpand', 'ty_TorchExpandAs', 'ty_TorchNumpy', 'ty_TorchReshape', 'ty_TorchSize', 'ty_TorchSplit', 'ty_TorchSqueeze', 'ty_TorchStack', 'ty_TorchTranspose', 'ty_TorchTranspose2D', 'ty_TorchPermute', 'ty_TorchUnsqueeze', 'ty_TorchArith', 'ty_TorchFlatten', 'ty_TorchMatmul', 'ty_TorchView', 'ty_TorchRepeat', 'ty_TorchModuleModules', 'ty_TorchPad', 'ty_TorchSum', 'ty_TorchMax', 'ty_TorchCompare', 'ty_TorchItem', 'ty_TorchNarrow', 'ty_TorchWhere', 'ty_TorchArgmax', 'ty_TorchAll', 'ty_TorchTopk'
           ]


def broadcast(shape1, shape2):
    assert(len(shape1) >= 1)
    assert(len(shape2) >= 1)
    if len(shape1) < len(shape2):
        shape1 = (1,) * (len(shape2) - len(shape1)) + tuple(shape1)
    elif len(shape1) > len(shape2):
        shape2 = (1,) * (len(shape1) - len(shape2)) + tuple(shape2)

    result_shape = []
    for x, y in zip(shape1, shape2):
        if x == 1:
            result_shape.append(y)
        elif y == 1:
            result_shape.append(x)
        else:
            assert(x == y)
            result_shape.append(x)

    return tuple(result_shape)


class ty_TorchIsTensor():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        return TyBool(isinstance(x_type, TyTensor) and x_type.is_torch_tensor())


class ty_TorchIdentical():
    def __init__(self, is_float_only=True, ndim_min=None, dtype=None):
        self.is_float_only = is_float_only
        self.ndim_min = ndim_min
        self.dtype = dtype

    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        if self.is_float_only:
            assert x_type.dtype.kind == 'f'
        if self.ndim_min:
            assert x_type.ndim >= self.ndim_min
        copied_ty = copy_ty(x_type)
        if self.dtype is not None:
            copied_ty.dtype = self.dtype
        return copied_ty

    def nn(self, obj, ty_args, ty_kwargs):
        check_dtype(obj, ty_args[0].dtype)
        return self(ty_args, ty_kwargs)

# Tensors
# Creation Ops


class ty_TorchTensor(ty_MakeTensor):
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        # TODO(momohatt): Use global default dtype
        dtype = self.get_element_dtype(x_type)
        return TyTorchTensor(dtype, shape=self.calculate_shape(x_type))


class ty_TorchTensorOfShape():
    def __call__(self, ty_args, ty_kwargs):
        if isinstance(ty_args[0], TyTuple):
            assert len(ty_args) >= 1
            shape = extract_value_from_ty(ty_args[0])
        else:
            for ty in ty_args:
                unify(ty, TyInt())
            shape = wrap_shape([extract_value_from_ty(ty) for ty in ty_args])

        # TODO: use global default dtype
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', np.dtype('float32'))
        assert not lacks_dtype

        return TyTorchTensor(dtype, shape=shape)


class ty_TorchFromNumpy():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.is_ndarray()
        x_type.kind = TensorKind.torch_tensor
        # XXX: Share reference of shape
        return TyTorchTensor(x_type.dtype, shape=x_type.shape)


class ty_TorchRandint():
    def __call__(self, ty_args, ty_kwargs):
        assert(len(ty_args) == 3)
        shape = wrap_shape([extract_value_from_ty(ty) for ty in ty_args[2]])
        return TyTorchTensor(np.dtype('int64'), shape=shape)


# Indexing, Slicing, Joining, Mutating Ops

class ty_TorchCat():
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        if isinstance(xs_type, TyList):
            self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)
            x_type = xs_type.ty
            return TyTorchTensor(x_type.dtype, ndim=x_type.ndim)
        elif isinstance(xs_type, TyTuple):
            self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)
            if not xs_type.is_fixed_len:
                raise NotImplementedError
            ty = xs_type.get_tys()
            if len(ty) == 0:
                raise NotImplementedError
            result_ty = joins(ty)
            assert isinstance(result_ty, TyTensor)
            for idx, x in enumerate(result_ty.shape):
                assert idx == self.dim or x is not None
            result_shape = list(result_ty.shape)
            result_shape[self.dim] = sum(x.shape[self.dim] for x in ty)

            return TyTorchTensor(result_ty.dtype, shape=result_shape)
        else:
            raise NotImplementedError


class ty_TorchChunk():
    def __call__(self, ty_args, ty_kwargs):
        x_type, chunk_type = ty_args
        assert isinstance(chunk_type, TyNum)
        chunks = chunk_type.value

        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)
        assert not lacks_dim
        return self.infer_return(x_type, chunks)

    def infer_return(self, x_type, chunks):
        ret_shape = list(x_type.shape)
        if chunks is None:
            ret_shape[self.dim] = None
            return TyTuple(TyTorchTensor(x_type.dtype, shape=ret_shape))

        # TODO(momohatt): Handle cases where dim is not divisible by chunks
        ret_shape[self.dim] = ret_shape[self.dim] // chunks
        return TyTuple([TyTorchTensor(x_type.dtype, shape=ret_shape)
                        for _ in range(chunks)])


class ty_TorchPad():
    def __call__(self, ty_args, ty_kwargs):
        x_type, pad = ty_args
        assert isinstance(pad, TyTuple)
        len_pad = pad.size()
        assert len_pad % 2 == 0
        len_pad = len_pad // 2
        shape = list(x_type.shape)
        assert len_pad <= len(shape)
        for i in range(len_pad):
            assert pad[i * 2].is_int()
            assert pad[i * 2 + 1].is_int()
            shape[-(i+1)] += pad[i * 2].value + pad[i * 2 + 1].value
        return TyTorchTensor(x_type.dtype, shape=shape)


class ty_TorchExpand():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert all([isinstance(ty, TyNum) for ty in ty_args[1:]])
        shape = [ty.value for ty in ty_args[1:]]
        return TyTorchTensor(x_type.dtype, shape=shape)


class ty_TorchExpandAs():
    def __call__(self, ty_args, ty_kwargs):
        assert isinstance(ty_args[0], TyTensor)
        assert isinstance(ty_args[1], TyTensor)
        x = ty_args[0].shape
        y = ty_args[1].shape
        assert(len(x) <= len(y))
        for ele_x, ele_y in zip(x, y[-len(x):]):
            if isinstance(ele_x, TyNum) and isinstance(ele_y, TyNum) and ele_x.value is not None and ele_y.value is not None:
                assert ele_x.value == ele_y.value or ele_x.value == 1
        return TyTorchTensor(ty_args[0].dtype, shape=ty_args[1].shape)


class ty_TorchNumpy():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.is_torch_tensor()
        return TyNdarray(x_type.dtype, x_type.shape)


class ty_TorchReshape():
    def __call__(self, ty_args, ty_kwargs):
        x_type, shape_type = ty_args
        assert isinstance(shape_type, TyTuple)
        assert shape_type.is_fixed_len

        self.shape = extract_value_from_ty(shape_type)
        return self.infer_return(x_type)

    def infer_return(self, x_type):
        ret_shape = calculate_reshape(x_type.shape, self.shape)
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


class ty_TorchSize():
    def __call__(self, ty_args, ty_kwargs):
        if len(ty_args) == 2:
            x_type, index_type = ty_args
            assert isinstance(index_type, TyNum)
            assert index_type.is_int()
            if index_type.value is None:
                return TyInt()
            return TyInt(x_type.shape[index_type.value].value)

        x_type, = ty_args
        return TyTuple([TyInt(e.value) for e in x_type.shape])


class ty_TorchSplit():
    def __call__(self, ty_args, ty_kwargs):
        x_type, split_size_or_sections_type = ty_args
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)

        # TODO: Handle cases where lacks_dim = True

        if isinstance(split_size_or_sections_type, TyNum):
            size = split_size_or_sections_type.value
            assert size is None or size > 0
            return self.infer_return_size(x_type, size)

        sections_type = split_size_or_sections_type
        assert isinstance(sections_type, TyTuple)
        return self.infer_return_sections(x_type, sections_type)

    def infer_return_size(self, x_type, size):
        if size is None:
            if self.dim is None:
                return TyTuple(TyTorchTensor(x_type.dtype, ndim=x_type.ndim))
            ret_shape = list(x_type.shape)
            ret_shape[self.dim] = None
            return TyTuple(TyTorchTensor(x_type.dtype, shape=ret_shape))

        if x_type.shape[self.dim].is_null():
            # TODO
            pass

        n_split = math.ceil(x_type.shape[self.dim].get_value() / size)
        if x_type.shape[self.dim] % size != 0:
            ret_shapes = [list(x_type.shape) for _ in range(n_split)]
            for i in range(n_split - 1):
                ret_shapes[i][self.dim] = size
            ret_shapes[-1][self.dim] = x_type.shape[self.dim] % size
            return TyTuple(
                [TyTorchTensor(x_type.dtype, shape=shape) for shape in ret_shapes])

        ret_shape = list(x_type.shape)
        ret_shape[self.dim] = size
        return TyTuple(
            [TyTorchTensor(x_type.dtype, shape=ret_shape)] * n_split)

    def infer_return_sections(self, x_type, sections_type):
        if not sections_type.is_fixed_len:
            ret_shape = list(x_type.shape)
            ret_shape[self.dim] = None
            return TyTuple(TyTorchTensor(x_type.dtype, shape=ret_shape))

        sections = extract_value_from_ty(sections_type)
        ret_shapes = [list(x_type.shape) for _ in sections]
        for i, n in enumerate(sections):
            ret_shapes[i][self.dim] = n
        return TyTuple([TyTorchTensor(x_type.dtype, shape=shape)
                        for shape in ret_shapes])


class ty_TorchSqueeze():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', None)

        if is_incomplete_shape(x_type.shape):
            if lacks_dim or self.dim is None:
                assert False, "torch.squeeze: cannot guess ndim of return type"

        return self.infer_return(x_type)

    def infer_return(self, x_type):
        if self.dim is not None:
            ret_shape = remove_dims(x_type.shape, (self.dim,))
        else:
            ret_shape = [s for s in x_type.shape if s != 1]
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


class ty_TorchStack():
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        assert isinstance(xs_type, TyList)

        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)

        if lacks_dim:
            x_type = xs_type.ty
            return TyTorchTensor(x_type.dtype, ndim=x_type.ndim + 1)

        self.dim %= xs_type.ty.ndim + 1
        return self.infer_return(xs_type)

    def infer_return(self, xs_type):
        ret_shape = list(xs_type.ty.shape)
        ret_shape.insert(self.dim, ShapeElem(None))
        return TyTorchTensor(xs_type.ty.dtype, shape=ret_shape)


class ty_TorchTranspose():
    def __call__(self, ty_args, ty_kwargs):
        x_type, dim1_type, dim2_type = ty_args
        assert isinstance(dim1_type, TyNum)
        assert isinstance(dim2_type, TyNum)
        dim1 = dim1_type.value
        dim2 = dim2_type.value
        assert dim1 is not None and dim2 is not None
        shape = list(x_type.shape)
        shape[dim1], shape[dim2] = x_type.shape[dim2], x_type.shape[dim1]
        return TyTorchTensor(x_type.dtype, shape=shape)


class ty_TorchTranspose2D():
    def __call__(self, ty_args, ty_kwargs):
        assert(len(ty_args) == 1)
        x_type = ty_args[0]
        shape = list(x_type.shape)
        assert(len(shape) <= 2)
        if (len(shape) == 2):
            shape[0], shape[1] = x_type.shape[1], x_type.shape[0]
        return TyTorchTensor(x_type.dtype, shape=shape)


class ty_TorchPermute():
    def __call__(self, ty_args, ty_kwargs):
        x_type, shape_perm = ty_args
        assert isinstance(shape_perm, TyTuple)
        assert shape_perm.is_fixed_len
        shape_perm = extract_value_from_ty(shape_perm)
        assert(len(shape_perm) == len(x_type.shape))
        new_shape = []
        for x in shape_perm:
            new_shape.append(x_type.shape[x])
        return TyTorchTensor(x_type.dtype, shape=new_shape)

    def infer_return(self, x_type):
        ret_shape = calculate_reshape(x_type.shape, self.shape)
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


class ty_TorchArgmax():
    def __call__(self, ty_args, ty_kwargs):
        print(ty_args)
        assert(len(ty_args) == 2)
        x_type = ty_args[0]
        dim_val = ty_args[1].value
        shape = list(x_type.shape)
        keepdim, lacks = get_kwarg(ty_kwargs, 'keepdim', default=False)
        if keepdim:
            shape[dim_val] = 1
        else:
            shape = shape[:dim_val] + shape[dim_val + 1:]
        return TyTorchTensor(np.int64, shape=shape)

class ty_TorchUnsqueeze():
    def __call__(self, ty_args, ty_kwargs):
        x_type, dim_type = ty_args
        assert isinstance(dim_type, TyNum)
        dim = extract_value_from_ty(dim_type)
        if dim is None:
            return TyTorchTensor(x_type.dtype, ndim=x_type.ndim + 1)

        shape = list(x_type.shape)
        shape.insert(dim, 1)
        return TyTorchTensor(x_type.dtype, shape=shape)


# Math operations
# Pointwise Ops

class ty_TorchArith(ty_TensorArith):
    def __init__(self, op):
        super().__init__(TensorKind.torch_tensor, op)


# Other operations

class ty_TorchFlatten():
    def __call__(self, ty_args, ty_kwargs):
        input_type, = ty_args
        shape = input_type.shape
        start_dim, _ = get_kwarg(ty_kwargs, 'start_dim', default=0)
        end_dim, _ = get_kwarg(ty_kwargs, 'end_dim', default=-1)

        prefix_shape = shape[:start_dim]
        middle_shape = shape[start_dim:end_dim] + (shape[end_dim],)
        postfix_shape = shape[end_dim:][2:]
        size = size_of_shape(middle_shape)
        out_shape = prefix_shape + (size,) + postfix_shape
        return TyTorchTensor(input_type.dtype, shape=out_shape)


class ty_TorchMatmul():
    def __call__(self, ty_args, ty_kwargs):
        assert len(ty_args) == 2
        x, y = ty_args
        assert isinstance(x, TyTensor)
        assert isinstance(y, TyTensor)
        assert(x.dtype == y.dtype)
        if len(x.shape) == 1 and len(y.shape) == 1:
            return TyTorchTensor(x.dtype, shape=())
        elif len(x.shape) == 2 and len(y.shape) == 2:
            return TyTorchTensor(x.dtype, shape=(x.shape[0], y.shape[1]))
        elif len(x.shape) == 1 and len(y.shape) == 2:
            assert(x.shape[0] == y.shape[0])
            return TyTorchTensor(x.dtype, shape=(y.shape[1],))
        elif len(x.shape) == 2 and len(y.shape) == 1:
            assert(x.shape[1] == y.shape[0])
            return TyTorchTensor(x.dtype, shape=(x.shape[0],))
        elif len(x.shape) >= 1 and len(y.shape) >= 1 and (len(x.shape) > 2 or len(y.shape) > 2):
            add_left = False
            if len(x.shape) == 1:
                add_left = True
                shape_left = (1, 1, x.shape[0])
            elif len(x.shape) == 2:
                shape_left = (1, x.shape[0], x.shape[1])
            else:
                shape_left = x.shape
            add_right = False
            if len(y.shape) == 1:
                add_right = True
                shape_right = (1, 1, y.shape[0])
            elif len(y.shape) == 2:
                shape_right = (1, y.shape[0], y.shape[1])
            else:
                shape_right = y.shape
            assert(shape_left[-1] == shape_right[-2])
            result_shape = broadcast(
                shape_left[:-2], shape_right[:-2]) + (shape_left[-2], shape_right[-1])
            if add_left:
                result_shape = result_shape[:-2] + result_shape[-1:]
            if add_right:
                result_shape = result_shape[:-1]
            return TyTorchTensor(x.dtype, shape=result_shape)
        else:
            raise ValueError


class ty_TorchSum():
    def __call__(self, ty_args, ty_kwargs):
        assert len(ty_args) == 1
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', None)
        self.keepdim, lacks_keep = get_kwarg(ty_kwargs, 'keepdim', False)
        x = ty_args[0]
        assert isinstance(x, TyTensor)
        if self.dim is None:
            self.dim = tuple(range(len(x.shape)))

        if isinstance(self.dim, int):
            self.dim = (self.dim,)

        reduce_dims = set(self.dim)
        shape = []
        for i in range(len(x.shape)):
            if i in reduce_dims or i - len(x.shape) in reduce_dims:
                if self.keepdim:
                    shape.append(1)
            else:
                shape.append(x.shape[i])
        return TyTorchTensor(x.dtype, shape=shape)


class ty_TorchAll():
    def __call__(self, ty_args, ty_kwargs):
        return TyBool()


class ty_TorchMax():
    def __call__(self, ty_args, ty_kwargs):
        assert len(ty_args) == 1
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', None)
        self.keepdim, lacks_keep = get_kwarg(ty_kwargs, 'keepdim', False)
        return_tensor = (len(ty_kwargs) == 0)
        x = ty_args[0]
        assert isinstance(x, TyTensor)
        if self.dim is None:
            self.dim = set(range(len(x.shape)))
            reduce_dims = self.dim
        else:
            reduce_dims = set(self.dim)

        shape = []
        for i in range(len(x.shape)):
            if i in reduce_dims:
                if self.keepdim:
                    shape.append(1)
            else:
                shape.append(x.shape[i])
        if return_tensor:
            return TyTorchTensor(x.dtype, shape=shape)
        else:
            return TyTuple([TyTorchTensor(x.dtype, shape=shape), TyTorchTensor(np.dtype("int64"), shape=shape)])


class ty_TorchCompare():
    def __call__(self, ty_args, ty_kwargs):
        assert len(ty_args) == 2
        x = ty_args[0]
        y = ty_args[1]
        assert isinstance(x, TyTensor)
        assert isinstance(y, TyTensor) or isinstance(y, TyNum)
        if isinstance(y, TyTensor):
            assert(x.shape == y.shape)

        return TyTorchTensor(np.dtype("bool_"), shape=x.shape)


# torch.Tensor

class ty_TorchView():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        shape_type = ty_args[1:]
        assert isinstance(x_type, TyTensor)

        out_shape = wrap_shape([extract_value_from_ty(t) for t in shape_type])
        ret_shape = calculate_reshape(x_type.shape, out_shape)
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


class ty_TorchRepeat():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        size_types = ty_args[1:]
        assert len(size_types) >= x_type.ndim
        assert all([isinstance(ty, TyNum) for ty in size_types])
        sizes = [ty.value for ty in size_types]
        return self.infer_return(x_type, sizes)

    def infer_return(self, x_type, sizes):
        n = len(sizes) - x_type.ndim
        for i in range(n, len(sizes)):
            sizes[i] *= x_type.shape[i - n]
        return TyTorchTensor(x_type.dtype, shape=sizes)


class ty_TorchModuleModules():
    def __call__(self):
        raise NotImplementedError


class ty_TorchItem():
    def __call__(self, ty_args, ty_kwargs):
        assert len(ty_args) == 1
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        assert len(x_type.shape) == 0
        if x_type.dtype == np.int32 or x_type.dtype == np.int64:
            return TyInt()
        raise NotImplementedError


class ty_TorchNarrow():
    def __call__(self, ty_args, ty_kwargs):
        assert len(ty_args) == 4
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        new_shape = list(x_type.shape)
        assert isinstance(ty_args[1], TyNum) and ty_args[1].is_int()
        assert isinstance(ty_args[2], TyNum) and ty_args[2].is_int()
        assert isinstance(ty_args[3], TyNum) and ty_args[3].is_int()

        new_shape[ty_args[1].value] = ty_args[3].value
        return TyTorchTensor(x_type.dtype, shape=tuple(new_shape))


class ty_TorchWhere():
    def __call__(self, ty_args, ty_kwargs):
        assert len(ty_args) == 3
        cond_ty = ty_args[0]
        true_ty = ty_args[1]
        false_ty = ty_args[2]

        assert(isinstance(cond_ty, TyTensor))
        assert(cond_ty.dtype == np.bool_)

        assert(isinstance(true_ty, TyTensor))
        assert(isinstance(false_ty, TyTensor))
        assert(true_ty.dtype == false_ty.dtype)
        if true_ty.shape != false_ty.shape:
            raise NotImplementedError

        return copy_ty(true_ty)


class ty_TorchTopk():
    def __call__(self, ty_args, ty_kwargs):
        assert len(ty_args) == 2
        x = ty_args[0]
        y = ty_args[1]
        assert isinstance(x, TyTensor)
        assert isinstance(y, TyNum) and y.is_int()
        dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', None)
        if dim is None: dim = len(x.shape) - 1
        new_shape = x.shape[:dim] + (y.value,) + x.shape[dim + 1:]

        return TyTuple([TyTorchTensor(x.dtype, new_shape), TyTorchTensor(np.dtype("int64"), new_shape)])