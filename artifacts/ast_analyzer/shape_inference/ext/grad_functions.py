from ast_analyzer.grad import impl as grad
from   ast_analyzer.shape_inference.types      import *

class ty_GradUnbroadcast():
    def __call__(self, ty_args, ty_kwargs):
        assert isinstance(ty_args[0], TyTensor)
        assert isinstance(ty_args[1], TyTensor)
        x = ty_args[0].shape
        y = ty_args[1].shape
        assert(len(x) >= len(y))
        for ele_x, ele_y in zip(x, y[-len(x):]):
            if isinstance(ele_x, TyNum) and isinstance(ele_y, TyNum) and ele_x.value is not None and ele_y.value is not None:
                assert ele_x.value == ele_y.value or ele_y.value == 1
        return TyTorchTensor(ty_args[0].dtype, shape=ty_args[1].shape)


grad_func_ty = {
    grad.unbroadcast: ty_GradUnbroadcast(),
}
