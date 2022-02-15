import functools

import torch

from .runtime import NNFusionRT


class TorchModule(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def nrt_forward(obj, *inputs):
    if not isinstance(obj, torch.nn.Module):
        return nrt_forward(TorchModule(obj), *inputs)

    outputs = obj(*inputs)
    output_is_tensor = isinstance(outputs, torch.Tensor)

    if output_is_tensor:
        outputs = [outputs]

    # TODO Pass other arguments from nrt_jit?
    # def nrt_forward(obj, *inputs, **kwargs):
    #     ...
    #     NNFusionRT(obj, inputs, outputs, **kwargs)
    nnf = NNFusionRT(obj, inputs, outputs, server="127.0.0.1:8880", steps=2000)
    nnf.compile(buildall=True)

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


def jit(func):
    @functools.wraps(func)
    def wrapper(*args):  # TODO support kwargs?
        if wrapper.forward is None:
            wrapper.forward = nrt_forward(func, *args)
        return wrapper.forward(*args)
    wrapper.forward = None
    return wrapper
