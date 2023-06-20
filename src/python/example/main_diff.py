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
    # q = torch.randn(1 , 32, 8192, 128).to(q)
    # k = torch.randn(1 , 32, 8192, 128).to(q)
    # v = torch.randn(1 , 32, 8192, 256).to(q)
    # mask = torch.randn(32, 8192, 8192).to(q)
    attn_acco = torch.zeros(1, 32, 8192, 256).to(q)
    expect = torch.load("/home/yuqxia/project/msa/output.pt")
    input_dict['q'] = cast_pytorch_tensor(q)
    input_dict['k'] = cast_pytorch_tensor(k)
    # input_dict['delta'] = cast_pytorch_tensor(torch.load("/home/yuqxia/nnfusion/delta.pt").half())
    # input_dict['do_'] = cast_pytorch_tensor(torch.load("/home/yuqxia/nnfusion/do.pt"))
    # input_dict['lse'] = cast_pytorch_tensor(torch.load("/home/yuqxia/nnfusion/lse.pt").half())
    input_dict['v'] = cast_pytorch_tensor(v)
    seq_k = 8192
    Br = 32
    Bc = 64
    Tr = 8192//Br
    Tc = seq_k//Bc
    h = 32
    # input_dict['mask'] = cast_pytorch_tensor(mask)
    input_dict['mask'] = cast_pytorch_tensor(mask.view(h, Tr, Br, Tc, Bc).permute(0, 1, 3, 2, 4).contiguous())
    # input_dict['acco'] = cast_pytorch_tensor(attn_acco)
    output_dict['out'] = cast_pytorch_tensor(torch.zeros(1, 32, 8192, 256).to(q))
    # input_dict['d'] = cast_pytorch_tensor(torch.zeros(1, 32, 8192).to(q))

    # print("expect out: ",torch.load("/home/yuqxia/nnfusion/out128.f32.pt"))
    # warm up
    # print(expect) 
    qr = q#[:, :, :3072]
   # qr.retain_grad()
    kr = k
    vr = v
    maskr = mask#[:, :3072]
    # exit(0)
    attn = qr @ kr.transpose(-1, -2)
    attn = attn * maskr
    d = attn.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
    # print(d.squeeze(-1))
    expect = torch.matmul(attn/d, vr)
    # exit(0)
    grad = torch.randn_like(expect)
    # exit(0)
    expect.backward(grad)
    q_grad = qr.grad
    k_grad = kr.grad
    v_grad = vr.grad
    # print(kr.grad)
    # print(vr.grad)

    #bwd
    seq_parallel = False
    if seq_parallel:
        q_chunk = 2048
        # k_chunk = 2048
        dq = torch.zeros_like(qr)
        dk = torch.zeros_like(kr)
        dv = torch.zeros_like(vr)
        # for j in range(4):
        for i in range(4):
            q = qr[:, :, i * q_chunk: (i+1) * q_chunk, :]
            k = kr#[:, :, j * k_chunk: (j+1) * k_chunk, :]
            v = vr#[:, :, j * k_chunk: (j+1) * k_chunk, :]
            m = maskr[:, i*q_chunk:(i+1) * q_chunk]#, j * k_chunk: (j+1) * k_chunk]
            d_ = d[:, :, i*q_chunk:(i+1) * q_chunk]
            do = grad[:, :, i*q_chunk:(i+1) * q_chunk, :]
            attn = q @ k.transpose(-1, -2)
            attn = attn * m
            s = attn.detach()/d_
            # dv[:, :, j * k_chunk: (j+1) * k_chunk, :] += s.transpose(-1, -2) @ do
            dv += s.transpose(-1, -2) @ do
            ds = do @ v.transpose(-1, -2)
            dqk = (ds/d_)*m
            # dk[:, :, j * k_chunk: (j+1) * k_chunk, :] += dqk.transpose(-1, -2) @ q
            dk += dqk.transpose(-1, -2) @ q
            dq[:, :, i * q_chunk: (i+1) * q_chunk, :] += dqk @ k
    else:
        q_chunk = 2048
        k_chunk = 2048
        dq = torch.zeros_like(qr)
        dk = torch.zeros_like(kr)
        dv = torch.zeros_like(vr)
        for j in range(4):
            for i in range(4):
                q = qr[:, :, i * q_chunk: (i+1) * q_chunk, :]
                k = kr[:, :, j * k_chunk: (j+1) * k_chunk, :]
                v = vr[:, :, j * k_chunk: (j+1) * k_chunk, :]
                m = maskr[:, i*q_chunk:(i+1) * q_chunk, j * k_chunk: (j+1) * k_chunk]
                d_ = d[:, :, i*q_chunk:(i+1) * q_chunk]
                do = grad[:, :, i*q_chunk:(i+1) * q_chunk, :]
                attn = q @ k.transpose(-1, -2)
                attn = attn * m
                s = attn.detach()/d_
                dv[:, :, j * k_chunk: (j+1) * k_chunk, :] += s.transpose(-1, -2) @ do
                ds = do @ v.transpose(-1, -2)
                dqk = (ds/d_)*m
                dk[:, :, j * k_chunk: (j+1) * k_chunk, :] += dqk.transpose(-1, -2) @ q
                dq[:, :, i * q_chunk: (i+1) * q_chunk, :] += dqk @ k
    
    diff_q = torch.abs(dq - q_grad)
    diff_k = torch.abs(dk - k_grad)
    diff_v = torch.abs(dv - v_grad)
    # print(q_grad)
    # print(dq)
    # print(diff_q)
    # indices = torch.where(diff_q == 0.1250)
    # first_index = tuple(coord[0] for coord in indices)
    # print(first_index)
    # print(dq[first_index], q_grad[first_index], diff_q[first_index])
    # print(dk)
    print(torch.max(diff_q), torch.mean(diff_q))
    print(torch.max(diff_k), torch.mean(diff_k))
    print(torch.max(diff_v), torch.mean(diff_v))






    # exit(0)
    # print(expect)
    # o = torch.zeros(1, 32, 8192, 256).to(qr)
    # d = torch.zeros(1, 32, 8192, 1).to(qr)
    # d_no_clamp = torch.zeros(1, 32, 8192, 1).to(qr)
    # for i in range(128):
    #     q = qr
    #     k = kr[:, :, i*64: (i+1) *64, :]
    #     v = vr[:, :, i*64: (i+1) *64, :]
    #     m = mask[:, :, i*64: (i+1) *64]

    #     attn = q @ k.transpose(-1, -2)
    #     v1 = attn * m
    #     cur_d_no_clamp = v1.detach().abs().sum(dim=-1, keepdim=True)
    #     d_no_clamp = d_no_clamp + cur_d_no_clamp
    #     d_new = cur_d_no_clamp.clamp(min=1) 
    #     o_new = torch.matmul(v1/d_new, v)

    #     o = d * o
    #     new_d = (d_no_clamp).clamp(min=1)
    #     # new_d = (d + d_new).clamp(min=1)
    #     o = (o + d_new*o_new)/new_d
    #     d = new_d
    # output
    # print(o)
    for _ in range(1):
        executor(input_dict, output_dict)
        for k, v in output_dict.items():
            out = v.to_pytorch_tensor()
            print(f"{k} = {out}")
    print(expect)
    # print(o)
    diff = torch.abs(expect - output_dict['out'].to_pytorch_tensor())
    print(torch.max(diff), torch.mean(diff))
    diff1 = torch.abs(d.squeeze(-1) - output_dict['d'].to_pytorch_tensor())
    print(torch.max(diff1), torch.mean(diff1))
    # diff = torch.abs(o - output_dict['Identity_13_0_0'].to_pytorch_tensor())
    # diff = torch.abs(expect - o)
    # print(diff)
    # print(torch.max(diff), torch.mean(diff))
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