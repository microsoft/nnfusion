import copy
import functools
from inspect import isfunction, ismethod, isclass

import torch

from .jit_utils import TorchModule, get_signature
from .runtime import NNFusionRT
from .config import Config


def is_method_of_instance(obj, cls):
    return ismethod(obj) and isinstance(obj.__self__, cls)


def is_subclass_of_cls(obj, cls):
    return isclass(obj) and issubclass(obj, cls)


def get_nrt_forward(obj, signature, config, outputs, *inputs,
                    is_method=False):
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

    # TODO nnf = NNFusionRT(obj, signature, **kwargs)
    nnf = NNFusionRT(obj, config, signature)
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


def nrt_forward(obj, *inputs, config=None, signature=None, is_method=False):
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

        return get_nrt_forward(self_, signature, config, outputs,
                               *inputs, is_method=True)

    if isfunction(obj) or is_method_of_instance(obj, torch.nn.Module):
        return get_nrt_forward(TorchModule(obj), signature, config, outputs,
                               *inputs)
    return get_nrt_forward(obj, signature, config, outputs, *inputs)


def parse_config(tune, tuning_steps, config):
    if config is None:
        config = Config()
    elif type(config) is dict:
        config = Config(config)

    if not type(config) is Config:
        raise TypeError(
            "Expected optional 'config' argument of type dict or "
            "nnfusion.Config but found {config}"
        )

    if tuning_steps is not None:
        if not isinstance(tuning_steps, int):
            raise TypeError(
                "Expected optional 'tuning_steps' argument of type int "
                "but found {tuning_steps}"
            )
        if tune is False:
            raise ValueError(
                f"Conflict is detected: tune={tune} and "
                f"tuning_steps={tuning_steps}"
            )

        tune = True
        config['kernel_tuning_steps'] = tuning_steps

    if tune is not None:
        if not isinstance(tune, bool):
            raise TypeError(
                "Expected optional 'tune' argument of type bool "
                "but found {tune}"
            )
        config['antares_mode'] = True

    return config


def check_obj_type(obj):
    if not (
        isfunction(obj)
        or isinstance(obj, torch.nn.Module)
        or is_subclass_of_cls(obj, torch.nn.Module)
        or is_method_of_instance(obj, torch.nn.Module)
    ):
        raise TypeError(
            "Expected function or torch.nn.Module instance/method/class "
            f"but found {obj}"
        )


def jit_class(obj, config):
    """
    Return jitted class using dynamic inheritance to override the forward
    function and keep its signature.
    """
    class JITModule(obj):
        @jit(config=config,
             _signature='.'.join([get_signature(obj), 'forward']))
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)
    return JITModule


def jit(obj=None, *, tune=None, tuning_steps=None, config=None, _signature=None):
    """
    Parameters:
        obj:
        tune:
        tuning_steps:
        config:
        _signature:
    """

    config = parse_config(tune, tuning_steps, config)

    def _jit(_obj):

        check_obj_type(_obj)

        if is_subclass_of_cls(_obj, torch.nn.Module):
            return jit_class(_obj, config)

        @functools.wraps(_obj)
        def wrapper(*args):  # TODO support kwargs?
            if wrapper.forward is None:
                wrapper.forward = nrt_forward(_obj, *args,
                                              config=config,
                                              signature=_signature)
            return wrapper.forward(*args)
        wrapper.forward = None

        if isinstance(_obj, torch.nn.Module):
            _obj._orig_forward = _obj.forward
            _obj.forward = wrapper
            return _obj
        return wrapper

    if obj is None:
        return _jit
    return _jit(obj)
