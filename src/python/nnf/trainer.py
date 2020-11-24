
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
    def __init__(self,
                 model,
                 loss_func,
                 device="cuda:0"):
        super(Trainer, self).__init__()
        self.model = model
        self.loss_func = loss_func
        self.model_with_loss = ModelWithLoss(self.model, self.loss_func).to(device)
        self.device = device
        codegen_flags = {
            "autodiff": 1, # add backward graph
            "training_mode": 1, # move weight external
            "extern_result_memory": 1 # move result external
        }
        self.runner = Runner(self.model_with_loss, codegen_flags=codegen_flags)

    def __call__(self, *args):
        return self.run_by_nnf(*args)

    def run_by_pytorch(self, *args):
        return self.model_with_loss(*args)

    def run_by_nnf(self, *args):
        ## only return the first output, because nnf will add all optimizer to output
        outs = self.runner(*args)
        for out in outs:
            if np.prod(out.shape) == 1:
                return out
        assert False