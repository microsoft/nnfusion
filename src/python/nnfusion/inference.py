# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import numpy as np
import copy
import json

from .runner import PTRunner


class PTInference(object):
    """
    Trainer is a wrapper to inference PyTorch model in NNFusion.
    """
    def __init__(self,
                 model,
                 device="cuda:0",
                 codegen_flags=None,
                 **kwargs):
        """
        It builds a training graph as well as optimizer based on PyTorch model.
        Currently the optimizer is SGD and non-configurable.

        Parameters:
            model: torch.nn.Module, to be trained.
            device: a string representing training device.
            kwargs: arguments for underlying Runner.
        """
        self.model = model
        self.device = device
        inference_flags = {
            "extern_result_memory":
            True,
        }
        self._codegen_flags = inference_flags
        self._codegen_flags.update(copy.deepcopy(codegen_flags) or {})
        self.runner = PTRunner(self.model,
                               codegen_flags=self._codegen_flags,
                               **kwargs)

    def __call__(self, *args):
        return self.run_by_nnf(*args)

    def run_by_pytorch(self, *args):
        return self.model(*args)

    def run_by_nnf(self, *args):
        """
        Parameters:
            args: a list of input tensors and label for origin PyTorch model inference.
        
        Returns:
            PyTorch model inference outputs.
        """
        outs = self.runner(*args)
        return outs
