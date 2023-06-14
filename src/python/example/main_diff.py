#!D:\project\transfer_xbox\python\tools\python.exe
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import os
os.environ["PATH"] = os.path.abspath(
    "/home/yuqxia/nnfusion/build/src/tools/nnfusion") + ":" + os.environ["PATH"]

sys.path.insert(1, os.path.abspath("/home/yuqxia/nnfusion/src/python"))
from logging import raiseExceptions
import time
import argparse
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

from nnfusion.executor import Executor
from nnfusion.session import generate_sample
from nnfusion.data_format import cast_pytorch_tensor, cast_hlsl_tensor, HLSLTensor


def inference(nnf_model_path, total_iter):
    assert total_iter >= 1
    executor = Executor(nnf_model_path)
    input_dict, output_dict = {}, {}
    if False:#executor.host_mode:
        # host mode leverage pytorch tensor as storage
        for input in executor.get_inputs():
            input_dict[input.name] = cast_pytorch_tensor(generate_sample(input))
        for output in executor.get_outputs():
            output_dict[output.name] = cast_pytorch_tensor(generate_sample(output))
    else:
        if executor.device_type == 0:
            # cuda device
            for input in executor.get_inputs():
                input_dict[input.name] = cast_pytorch_tensor(generate_sample(input, "cuda"))
            for output in executor.get_outputs():
                output_dict[output.name] = cast_pytorch_tensor(generate_sample(output, "cuda"))
        elif executor.device_type == 3:
            # hlsl device
            for input in executor.get_inputs():
                input_dict[input.name] = cast_hlsl_tensor(HLSLTensor.build_from_torch(generate_sample(input)))
            for output in executor.get_outputs():
                output_dict[output.name] = cast_hlsl_tensor(HLSLTensor.build_from_torch(generate_sample(output)))
        else:
            raise Exception("only support device kernel_entry on cuda/hlsl backend.")
        
    # q = torch.randn(2, 8, 128, 64).cuda()
    # k = torch.randn(2, 8, 128, 64).cuda()
    # v = torch.Tensor(2, 8, 128, 64).cuda()
    # torch.save(q,'q128.f32.pt')
    # torch.save(k, 'k128.f32.pt')
    # torch.save(v, 'v128.f32.pt')
    # torch.save(latent_model_input_pt, "/home/yuqxia/sample.f32.pt")
    # torch.save(text_embeddings_pt, "/home/yuqxia/encoder_hidden_states.f32.pt")
    # torch.save(timesteps_pt, "timestep.f32.pt")
    # # print(input_dict)
    q = torch.load("/home/yuqxia/project/msa/qr.pt")
    # q = torch.randn(1, 32, 8192, 128).to(q)
    k = torch.load("/home/yuqxia/project/msa/kr.pt")#[:, :, :64  , :]
    v = torch.load("/home/yuqxia/project/msa/vr.pt")#[:, :, :64  , :]
    # mask = torch.ones_like(torch.load("/home/yuqxia/project/msa/mask.pt"))#[:, :, :64]
    mask = torch.load("/home/yuqxia/project/msa/mask.pt")#[:, :, :64]
    attn_acco = torch.zeros(1, 32, 8192, 256).to(q)
    expect = torch.load("/home/yuqxia/project/msa/output.pt")
    input_dict['q'] = cast_pytorch_tensor(q)
    input_dict['k'] = cast_pytorch_tensor(k)
    # input_dict['delta'] = cast_pytorch_tensor(torch.load("/home/yuqxia/nnfusion/delta.pt").half())
    # input_dict['do_'] = cast_pytorch_tensor(torch.load("/home/yuqxia/nnfusion/do.pt"))
    # input_dict['lse'] = cast_pytorch_tensor(torch.load("/home/yuqxia/nnfusion/lse.pt").half())
    input_dict['v'] = cast_pytorch_tensor(v)
    Br = 32
    Bc = 64
    Tr = 8192//Br
    Tc = 8192//Bc
    h = 32
    input_dict['mask'] = cast_pytorch_tensor(mask.view(h, Tr, Br, Tc, Bc).permute(0, 1, 3, 2, 4).contiguous())
    input_dict['attn_acco'] = cast_pytorch_tensor(attn_acco)
    output_dict['Identity_10_0_0'] = cast_pytorch_tensor(torch.zeros(1, 32, 8192, 256).to(q))
    # print("expect out: ",torch.load("/home/yuqxia/nnfusion/out128.f32.pt"))
    # warm up
    # print(expect)
    qr = q
    kr = k
    vr = v
    maskr = mask#.unsqueeze(0)
    attn = qr @ kr.transpose(-1, -2)
    attn = attn * maskr
    expect = torch.matmul(attn, vr)
    # print(expect)
    output = torch.zeros(1, 32, 8192, 256).to(qr)
    for i in range(128):
        q = qr
        k = kr[:, :, i*64: (i+1) *64, :]
        v = vr[:, :, i*64: (i+1) *64, :]
        m = mask[:, :, i*64: (i+1) *64]
        attn = q @ k.transpose(-1, -2)
        attn = attn * m
        attn = torch.matmul(attn, v)
        output += attn
    print(output)
    for _ in range(1):
        executor(input_dict, output_dict)
        for k, v in output_dict.items():
            out = v.to_pytorch_tensor()
            print(f"{k} = {out}")
    max_diff = 0
    total_diff  = 0
    diff = torch.abs(expect - output_dict['Identity_10_0_0'].to_pytorch_tensor())
    print(torch.max(diff), torch.mean(diff))
    # evaluate
    print(f"Begin evaluation of {total_iter} iters")
    start = time.time()
    perf_list = []
    for _ in range(total_iter):
        start_i = time.time()
        executor(input_dict, output_dict)
        end_i = time.time()
        #print(end_i - start_i)
        perf_list.append(end_i - start_i)
    end = time.time()

    latency_ms = np.array(perf_list) * 1000
    batch_size = list(input_dict.values())[0].shape[0]
    # print(f"average_latency = {np.mean(latency_ms)} ms")
    # print(f"latency_50 = {np.percentile(latency_ms, 50)} ms")
    # print(f"latency_75 = {np.percentile(latency_ms, 75)} ms")
    # print(f"latency_90 = {np.percentile(latency_ms, 90)} ms")
    # print(f"latency_95 = {np.percentile(latency_ms, 95)} ms")
    # print(f"latency_99 = {np.percentile(latency_ms, 99)} ms")
    print(f"throughput = {batch_size * (1000.0 / np.mean(latency_ms))} sample/s")
    # print(f"total elaspe {end - start} s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnf_model_path', type=str)
    parser.add_argument('--total_iter', type=int, default=1)
    args = parser.parse_args()
    inference(args.nnf_model_path, args.total_iter)


if __name__ == "__main__":
    main()