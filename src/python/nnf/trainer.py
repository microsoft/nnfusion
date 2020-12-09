# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import numpy as np

from .runner import Runner


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss_func):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss_func = loss_func

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss_func(output, target)
        return loss


class Trainer(object):
    """
    Trainer is a wrapper to train PyTorch model in NNFusion.
    """
    def __init__(self, model, loss_func=None, device="cuda:0", **kwargs):
        """
        It builds a training graph as well as optimizer based on PyTorch model.
        Currently the optimizer is SGD and non-configurable.

        Parameters:
            model: torch.nn.Module, to be trained.
            loss_func: Optional, PyTorch loss func. Its signature is accepting model 
                output and a label tensor, return a tensor representing loss. If not
                provided, training is directly on origin model.
            device: a string representing training device.
            kwargs: arguments for underlying Runner.
        """
        super(Trainer, self).__init__()
        self.model = model
        self.loss_func = loss_func
        if self.loss_func:
            self.model_with_loss = ModelWithLoss(self.model,
                                                 self.loss_func).to(device)
        else:
            self.model_with_loss = model
        self.device = device
        codegen_flags = {
            "autodiff": 1,  # add backward graph
            "training_mode": 1,  # move weight external
            "extern_result_memory": 1  # move result external
        }
        self.runner = Runner(self.model_with_loss,
                             codegen_flags=codegen_flags,
                             **kwargs)

    def __call__(self, *args):
        return self.run_by_nnf(*args)

    def run_by_pytorch(self, *args):
        return self.model_with_loss(*args)

    def run_by_nnf(self, *args):
        """
        Parameters:
            args: a list of input tensors and label for origin PyTorch model training.
        
        Returns:
            a PyTorch tensor representing loss.
        """
        outs = self.runner(*args)
        for out in outs:
            if np.prod(out.shape) == 1:
                return out
        assert False