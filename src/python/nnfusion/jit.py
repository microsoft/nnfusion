import copy
import functools
from inspect import isfunction, ismethod, isclass

import torch

from .jit_utils import TorchModule, get_signature
from .runtime import NNFusionRT


def is_method_of_instance(obj, cls):
    return ismethod(obj) and isinstance(obj.__self__, cls)


def is_subclass_of_cls(obj, cls):
    return isclass(obj) and issubclass(obj, cls)


def get_nrt_forward(obj, signature, outputs, *inputs,
                    is_method=False, **kwargs):
    """
    Return a wrapped forward function that using nnf as runtime
    """

    if not isinstance(obj, torch.nn.Module):
        raise AssertionError(
            "Internal bug, please report to "
            "https://github.com/microsoft/nnfusion"
        )

    output_is_tensor = isinstance(outputs, torch.Tensor)
    if output_is_tensor:
        outputs = [outputs]

    nnf = NNFusionRT(obj, signature, **kwargs)
    nnf.compile(inputs, outputs)

    # TODO free outputs and only save desc?

    def forward(*inputs):
        if is_method:
            _, *inputs = inputs
        results = [
            torch.empty_like(output)
            for output in outputs
        ]
        inputs = list(inputs)
        nnf.run(inputs, results)

        if output_is_tensor:
            return results[0]
        return results
    return forward


def nrt_forward(obj, *inputs, is_method=False, signature=None, **kwargs):
    if signature is None:
        signature = get_signature(obj)

    if hasattr(obj, '_orig_forward'):
        # shallow copy is needed to avoid recursion
        # call instance forward -> call nnf_forward -> call instance forward
        obj_ = copy.copy(obj)
        obj_.forward = obj._orig_forward
        obj = obj_

    outputs = obj(*inputs)

    def jit_class_method_using_decorator():
        """
        Check if obj is a class method with @nnfusion.jit decorator.
        The cases of decorating class method with the @ symbol or applying it
        as function are different.
        """
        return isinstance(inputs[0], torch.nn.Module)

    if jit_class_method_using_decorator():
        self, *inputs = inputs

        # shallow copy is needed to avoid recursion when using jit as decorator:
        # export onnx -> call forward to trace -> call nnf jit func -> export onnx
        self_ = copy.copy(self)

        def forward(*args):
            if forward.first_call:
                forward.first_call = False
                return obj(self, *args)
            # handle the case that jit target function will call `forward`
            return self.forward(*args)
        forward.first_call = True
        self_.forward = forward

        return get_nrt_forward(self_, signature, outputs,
                               *inputs, is_method=True, **kwargs)

    if isfunction(obj) or is_method_of_instance(obj, torch.nn.Module):
        return get_nrt_forward(TorchModule(obj), signature, outputs,
                               *inputs, **kwargs)
    return get_nrt_forward(obj, signature, outputs, *inputs, **kwargs)


def jit(_obj=None, signature=None, **kwargs):
    def decorator_jit(obj):

        if not (
            isfunction(obj)
            or isinstance(obj, torch.nn.Module)
            or is_subclass_of_cls(obj, torch.nn.Module)
            or is_method_of_instance(obj, torch.nn.Module)
        ):
            raise RuntimeError(
                "Accept function or torch.nn.Module or class method of "
                f"torch.nn.Module but found {obj}"
            )

        if is_subclass_of_cls(obj, torch.nn.Module):
            class JITModule(obj):
                """
                Dummy class using dynamic inheritance to override forward
                function and keep its signature.
                """
                @jit(signature='.'.join([get_signature(obj), 'forward']),
                     **kwargs)
                def forward(self, *args, **kwargs):
                    return super().forward(*args, **kwargs)
            return JITModule

        @functools.wraps(obj)
        def wrapper(*args):  # TODO support kwargs?
            if wrapper.forward is None:
                wrapper.forward = nrt_forward(obj, *args,
                                              signature=signature, **kwargs)
            return wrapper.forward(*args)
        wrapper.forward = None

        # If jit an instance, return itself instead of only a function
        if isinstance(obj, torch.nn.Module):
            obj._orig_forward = obj.forward
            obj.forward = wrapper
            return obj
        return wrapper

    if _obj is None:
        return decorator_jit
    return decorator_jit(_obj)
