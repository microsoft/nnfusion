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

def build_nnfusion(onnx_model_path, codegen_flags, workdir, rt_dir):
    flags_str = "-f onnx "
    flags_str += " ".join([
        "-f{}={}".format(k, v) for k, v in codegen_flags.items()
    ])
    os.system(f"rm -r {workdir}")
    os.system(f"mkdir -p {workdir}")
    codegen(onnx_model_path, flags_str, workdir)
    # os.system(f"cat {workdir}/codegen.log ")
    modify_nnfusion_rt(rt_dir)
    build(rt_dir)

def load_model(model_path: str):
    assert(model_path.endswith('.onnx'))
    workdir = os.path.abspath(model_path[:-5])
    codegen_flags = {'autodiff': False, 'training_mode': False, 'extern_result_memory': True, 'codegen_unexist_kernel': True, 'product_name': 'V100'}
    rt_dir = os.path.join(workdir, "nnfusion_rt/cuda_codegen")
    build_nnfusion(model_path, codegen_flags, workdir, rt_dir)
    executor = Executor(rt_dir)
    return executor

cuda_device = torch.device("cuda:0")

class GenModel(torch.autograd.Function):
    forward_executor = load_model("tmp/seq2seq_bs1_0-forward.onnx")

    @staticmethod
    def forward(ctx, _i0, _i1, _i2, _i3, _i4, _i5):
        # print("use nnfusion forward")
        tmp_i0 = cast_pytorch_tensor(_i0)
        tmp_i1 = cast_pytorch_tensor(_i1)
        tmp_i2 = cast_pytorch_tensor(_i2)
        tmp_i3 = cast_pytorch_tensor(_i3)
        tmp_i4 = cast_pytorch_tensor(_i4)
        tmp_i5 = cast_pytorch_tensor(_i5)
        output_tensors = GenModel.forward_executor.alloc_output_buffer()
        output_casted = [cast_pytorch_tensor(x) for x in output_tensors]
        output_signatures = [x.pointer_type for x in output_casted]
        output_pointers = [x.pointer for x in output_casted]
    
        signatures = [
            tmp_i0.pointer_type,
            tmp_i1.pointer_type,
            tmp_i2.pointer_type,
            tmp_i3.pointer_type,
            tmp_i4.pointer_type,
            tmp_i5.pointer_type,
        ] + output_signatures
        pointers = [
            tmp_i0.pointer,
            tmp_i1.pointer,
            tmp_i2.pointer,
            tmp_i3.pointer,
            tmp_i4.pointer,
            tmp_i5.pointer,
        ] + output_pointers

        GenModel.forward_executor.feed_pointers(signatures, pointers)
        _o0 = output_tensors[0]
        _o1 = output_tensors[1]
        _o2 = output_tensors[2]
        _o3 = output_tensors[3]
        _o4 = output_tensors[4]
        _o5 = output_tensors[5]
        return _o0, _o1, _o2, _o3, _o4, _o5

    @staticmethod
    def backward(ctx, _r0, _r1, _r2, _r3, _r4, _r5):
        # print("use nnfusion backward")
        ctx_casted = [cast_pytorch_tensor(x) for x in ctx.saved_tensors]
        ctx_signatures = [x.pointer_type for x in ctx_casted]
        ctx_pointers = [x.pointer for x in ctx_casted]
        _r0 = cast_pytorch_tensor(_r0)
        _r1 = cast_pytorch_tensor(_r1)
        _r2 = cast_pytorch_tensor(_r2)
        _r3 = cast_pytorch_tensor(_r3)
        _r4 = cast_pytorch_tensor(_r4)
        _r5 = cast_pytorch_tensor(_r5)

        output_tensors = GenModel.backward_executor.alloc_output_buffer()
        output_casted = [cast_pytorch_tensor(x) for x in output_tensors]
        output_signatures = [x.pointer_type for x in output_casted]
        output_pointers = [x.pointer for x in output_casted]

        signatures = [
            _r0.pointer_type,
            _r1.pointer_type,
            _r2.pointer_type,
            _r3.pointer_type,
            _r4.pointer_type,
            _r5.pointer_type,
        ] + ctx_signatures + output_signatures
        pointers = [
            _r0.pointer,
            _r1.pointer,
            _r2.pointer,
            _r3.pointer,
            _r4.pointer,
            _r5.pointer,
        ] + ctx_pointers + output_pointers

        GenModel.backward_executor.feed_pointers(signatures, pointers)
        return output_tensors
