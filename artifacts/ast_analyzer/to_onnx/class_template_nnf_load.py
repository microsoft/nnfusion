from ast_analyzer.grad.cfg import backward, forward
import torch
import os
import sys
from time import time

NNFUSION_ROOT = os.path.expanduser("~/nnfusion")
os.environ["PATH"] = os.path.abspath(NNFUSION_ROOT) + ":" + os.environ["PATH"]
sys.path.insert(1, os.path.abspath(NNFUSION_ROOT + "/src/python"))

from nnfusion.session import codegen, modify_nnfusion_rt, build
from nnfusion.executor import Executor
from nnfusion.data_format import cast_pytorch_tensor

def load_model():
    executor = Executor("^^BEST_RT_DIR")
    return executor

cuda_device = torch.device("cuda:0")

class GenModel(torch.autograd.Function):
    forward_executor = load_model()

    @staticmethod
    def forward(ctx, ^^INPUTS):
        # print("use nnfusion forward")
        @.@INPUTS@@@Tensor@tmp^^NAME = cast_pytorch_tensor(^^NAME)@@General@tmp^^NAME=cast_pytorch_tensor(torch.full((), ^^NAME, device=cuda_device))@@@
        output_tensors = GenModel.forward_executor.alloc_output_buffer()
        output_casted = [cast_pytorch_tensor(x) for x in output_tensors]
        output_signatures = [x.pointer_type for x in output_casted]
        output_pointers = [x.pointer for x in output_casted]
    
        signatures = [
            @.@INPUTStmp^^NAME.pointer_type,
        ] + output_signatures
        pointers = [
            @.@INPUTStmp^^NAME.pointer,
        ] + output_pointers

        GenModel.forward_executor.feed_pointers(signatures, pointers)
        @.@OUTPUTS^^NAME = output_tensors[%%i]
        @.@PARAMS^^O_NAME = ^^I_NAME
        return ^^O_OUTPUTS

    @staticmethod
    def backward(ctx, ^^RETURNS):
        raise NotimplementedError
