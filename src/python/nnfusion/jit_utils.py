import inspect
import os
import re
from pathlib import Path

import torch


class TorchModule(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def get_signature(obj, suffix=''):
    """
    Signature of a function or torch.nn.Module instance to detect reusable
    kernel.
    For details, please refer to https://github.com/microsoft/nnfusion/pull/379
    """

    if isinstance(obj, torch.nn.Module):
        return get_signature(obj.__class__)

    if not (
        inspect.isfunction(obj)
        or inspect.ismethod(obj)
        or inspect.isclass(obj)
    ):
        raise Exception(f"Not support type {obj} for get_signature")

    def get_qualname():
        name = obj.__qualname__
        # Remove special chars to avoid the trouble of dealing with paths
        return re.sub("[<>]", "", name)

    def get_path():
        # Avoid collision between different files
        obj_path = inspect.getsourcefile(obj)
        relpath = os.path.relpath(obj_path)
        return "-".join(Path(os.path.splitext(relpath)[0]).parts)

    return "-".join(('nnf', get_path(), get_qualname())) + suffix
