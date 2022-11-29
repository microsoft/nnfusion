#!D:\project\transfer_xbox\python\tools\python.exe
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from logging import raiseExceptions
import time
import argparse
import numpy as np
import torch
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

from nnfusion.executor import Executor
from nnfusion.session import generate_sample
from nnfusion.data_format import cast_pytorch_tensor, cast_hlsl_tensor, HLSLTensor


def inference(nnf_model_path, total_iter):
    assert total_iter >= 1
    executor = Executor(nnf_model_path)
    input_dict, output_dict = {}, {}
    if executor.host_mode:
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

    
    # warm up
    for _ in range(5):
        executor(input_dict, output_dict)
        for k, v in output_dict.items():
            print(f"{k} = {v.to_pytorch_tensor()}")

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
    print(f"average_latency = {np.mean(latency_ms)} ms")
    print(f"latency_50 = {np.percentile(latency_ms, 50)} ms")
    print(f"latency_75 = {np.percentile(latency_ms, 75)} ms")
    print(f"latency_90 = {np.percentile(latency_ms, 90)} ms")
    print(f"latency_95 = {np.percentile(latency_ms, 95)} ms")
    print(f"latency_99 = {np.percentile(latency_ms, 99)} ms")
    print(f"throughput = {batch_size * (1000.0 / np.mean(latency_ms))} sample/s")
    print(f"total elaspe {end - start} s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnf_model_path', type=str)
    parser.add_argument('--total_iter', type=int, default=1)
    args = parser.parse_args()
    inference(args.nnf_model_path, args.total_iter)


if __name__ == "__main__":
    main()