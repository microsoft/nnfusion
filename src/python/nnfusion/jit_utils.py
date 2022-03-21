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


def get_signature(obj):
    """
    Signature of a function or torch.nn.Module instance to detect reusable
    kernel.
    """
    # For details, please refer to https://github.com/microsoft/nnfusion/pull/379
    def get_qualname():
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            name = obj.__qualname__
        else:
            name = obj.__class__.__qualname__
        # Remove special chars to avoid the trouble of dealing with paths
        return re.sub("[<>]", "", name)

    def get_path():
        # Avoid collision between different files
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            obj_path = inspect.getsourcefile(obj)
        else:
            obj_path = inspect.getsourcefile(obj.__class__)
        relpath = os.path.relpath(obj_path)
        return "-".join(Path(os.path.splitext(relpath)[0]).parts)

    return "-".join(('nnf', get_path(), get_qualname()))
