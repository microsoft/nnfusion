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
    Signature of an object to detect reusable kernel.
    """

    if isinstance(obj, torch.nn.Module):
        return get_signature(obj.__class__)

    if not (
        inspect.isfunction(obj)
        or inspect.ismethod(obj)
        or inspect.isclass(obj)
    ):
        raise Exception(f"Not support type {obj} for get_signature")

    signature = "-".join([obj.__module__, obj.__qualname__])

    # Remove special chars to avoid the trouble of dealing with paths
    return re.sub("[<>]", "", signature)
