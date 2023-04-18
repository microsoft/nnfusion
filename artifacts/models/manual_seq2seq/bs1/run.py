import torch
import os
import sys
from ast_analyzer.utils.timer import Timer
from ast_analyzer.workflow import profile_start, profile_stop, enable_profile


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
    # build_nnfusion(model_path, codegen_flags, workdir, rt_dir)
    executor = Executor(rt_dir)
    return executor

cuda_device = torch.device("cuda:0")

class GenModel(torch.autograd.Function):
    forward_executor = load_model("seq2seq_bs1_0-forward.onnx")

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


prefix = "../artifacts/data/seq2seq"
MAX_LENGTH = 50
OUTPUT_SIZE = 3797
HIDDEN_SIZE = 256
device='cuda'
import torch
import torch.nn as nn
import numpy as np

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.__seq2seq_bs1_0 = GenModel()

    def forward(self, encoder_output, std, h, c):
        batch_size = 1
        output_all = (torch.zeros(50, 1, dtype=torch.int64, device='cuda') + 0)
        output = torch.full((1,), 1, dtype=torch.int64, device='cuda')
        cond = True
        id = torch.zeros((), dtype=torch.int64, device='cuda')
        while cond:
            (h, output, c, id, _, cond) = self.__seq2seq_bs1_0.apply(h, output, c, id, output_all, std)
        return (output_all, h)

def load_model():
    attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE, OUTPUT_SIZE, dropout_p=0.1).to(device).eval()
    attn_decoder1 = attn_decoder1.eval()
    return attn_decoder1

def gen_mask_from_sequence(std):
    bs = std.shape[0]
    padded_std = torch.zeros((bs, MAX_LENGTH), dtype=std.dtype, device=device)
    padded_std[:, :std.shape[1]] = std
    mask = torch.zeros(bs, MAX_LENGTH, OUTPUT_SIZE, device=device)
    mask[torch.arange(bs).unsqueeze(1), torch.arange(MAX_LENGTH).unsqueeze(0), padded_std] = 1000000.0
    mask = mask.transpose(0, 1).contiguous().clone()
    return mask

def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = torch.from_numpy(np.fromfile(s + ".bin", dtype=dtype)).reshape(shape)
    return tensor


if __name__ == '__main__':
    with torch.no_grad():
        torch.manual_seed(0)
        batch_size = 1
        model = load_model()
        std = []
        MAX_LENGTH = 50
        device = 'cuda'
        encoder_output = torch.randn(MAX_LENGTH, batch_size, HIDDEN_SIZE, device=device)
        h = torch.randn(batch_size, HIDDEN_SIZE, device='cuda')
        c = torch.randn(batch_size, HIDDEN_SIZE, device='cuda')
        n_warmup = 200
        n_run = 100
        len_dataset = 6400
        tokens = read_bin('../../../artifacts/data/tatoeba-eng-fra/tokens', dtype=np.int64).cuda()
        masks = gen_mask_from_sequence(tokens)
        for i in range(0, len_dataset, batch_size):
            if i >= n_warmup * batch_size: break
            mask = masks[:, i:i+batch_size].contiguous()
            torch.cuda.synchronize()
            output_all, h = model(encoder_output, mask, h, c)
            if i == 0: print(output_all)
            torch.cuda.synchronize()
        # run
        timer = Timer("ms")
        enable_profile('V100')
        profile_start('V100')
        for i in range(0, len_dataset, batch_size):
            if i >= n_run * batch_size: break
            mask = masks[:, i:i+batch_size].contiguous()
            torch.cuda.synchronize()
            timer.start()
            _ = model.forward(encoder_output, mask, h, c)
            torch.cuda.synchronize()
            timer.log()
        timer.report()
        profile_stop('V100')