import functools

import torch

from .jit_utils import TorchModule
from .runtime import NNFusionRT


def nrt_forward(obj, *inputs, **kwargs):
    if not isinstance(obj, torch.nn.Module):
        return nrt_forward(TorchModule(obj), *inputs)

    outputs = obj(*inputs)
    output_is_tensor = isinstance(outputs, torch.Tensor)

    if output_is_tensor:
        outputs = [outputs]

    nnf = NNFusionRT(obj, **kwargs)
    nnf.compile(inputs, outputs)

    # TODO free outputs and only save desc?

    def forward(*inputs):
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


def jit(_func=None, **kwargs):
    def decorator_jit(func):
        @functools.wraps(func)
        def wrapper(*args):  # TODO support kwargs?
            if wrapper.forward is None:
                wrapper.forward = nrt_forward(func, *args, **kwargs)
            return wrapper.forward(*args)
        wrapper.forward = None
        return wrapper

    if _func is None:
        return decorator_jit
    return decorator_jit(_func)
