from ast_analyzer.grad.cfg import backward, forward
import torch
import os
import sys
from time import time

NNFUSION_ROOT = os.path.expanduser("~/nnfusion")
os.environ["PATH"] = os.path.abspath(NNFUSION_ROOT) + ":" + os.environ["PATH"]
sys.path.insert(1, os.path.abspath(NNFUSION_ROOT + "/src/python"))

from nnfusion.session import modify_nnfusion_rt, build
from nnfusion.executor import Executor
from nnfusion.data_format import cast_pytorch_tensor



def load_model(model_path: str):
    assert(model_path.endswith('.onnx'))
    workdir = os.path.abspath(model_path[:-5])
    rt_dir = os.path.join(workdir, "nnfusion_rt/cuda_codegen")
    modify_nnfusion_rt(rt_dir)
    build(rt_dir)
    executor = Executor(rt_dir)
    return executor

class GenModel(torch.autograd.Function):
    @IF_EVALforward_executor = load_model("tmp/^^MODELNAME-forward.onnx")
    @IF_TRAINforward_executor = load_model("tmp/^^MODELNAME-train-fwd.onnx")
    @IF_TRAINbackward_executor = load_model("tmp/^^MODELNAME-train-bwd.onnx")

    @staticmethod
    def forward(ctx, ^^INPUTS):
        # print("use nnfusion forward")
        @.@INPUTS@@@Tensor@tmp^^NAME = cast_pytorch_tensor(^^NAME)@@General@tmp^^NAME = cast_pytorch_tensor(torch.tensor(^^NAME, device='cuda'))@@@
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
        @IF_TRAINctx.save_for_backward(^^CTX_SAVE)
        return ^^O_OUTPUTS

    @staticmethod
    def backward(ctx, ^^RETURNS):
        # print("use nnfusion backward")
        ctx_casted = [cast_pytorch_tensor(x) for x in ctx.saved_tensors]
        ctx_signatures = [x.pointer_type for x in ctx_casted]
        ctx_pointers = [x.pointer for x in ctx_casted]
        @.@RETURNS^^NAME = cast_pytorch_tensor(^^NAME)

        output_tensors = GenModel.backward_executor.alloc_output_buffer()
        output_casted = [cast_pytorch_tensor(x) for x in output_tensors]
        output_signatures = [x.pointer_type for x in output_casted]
        output_pointers = [x.pointer for x in output_casted]

        signatures = [
            @.@RETURNS^^NAME.pointer_type,
        ] + ctx_signatures + output_signatures
        pointers = [
            @.@RETURNS^^NAME.pointer,
        ] + ctx_pointers + output_pointers

        GenModel.backward_executor.feed_pointers(signatures, pointers)
        return output_tensors
        