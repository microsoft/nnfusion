import torch
import numpy as np

class GenTrainingModel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ^^FWD_INPUTS):
        # print("use functional forward")
        ^^FWD_CODE
        ^^SCALAR_TO_TENSOR
        ctx.save_for_backward(^^CTX_SAVE)
        return ^^FWD_RETURN
    
    @staticmethod
    def backward(ctx, ^^BWD_INPUTS):
        # print("use functional backward")
        ^^CTX_OR_SAVE = ctx.saved_tensors
        ^^TENSOR_TO_SCALAR
        ^^BWD_CODE
        ^^BWD_RETURN_STMT
