from Modelattention_bs1_0_nnfusion import GenModel as Model0
from Modelattention_bs1_1_nnfusion import GenModel as Model1

import torch
import os
import sys
from ast_analyzer.utils.timer import Timer
from ast_analyzer.workflow import profile_start, profile_stop, enable_profile
import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.__attention_bs1_0 = Model0()
        self.__attention_bs1_1 = Model1()

    def forward(self, x, k, v):
        (gen_id, k, v) = self.__attention_bs1_1.apply(k, v)
        for i in range(32):
            (gen_id, _, x, _) = self.__attention_bs1_0.apply(gen_id, k, x, v)
        return (k, v, x)

START_LEN = 32
SEQ_LEN = 64
NUM_HEAD = 12
SIZE_PER_HEAD = 64
n_warmup = 1000
n_run = 1000

if __name__ == "__main__":
    batch_size = 1
    model = Attention().cuda().eval()
    x = torch.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD).cuda()
    k = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    k[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')

    for i in range(n_warmup):
        torch.cuda.synchronize()
        _ = model.forward(x, k, v)
    # run
    timer = Timer("ms")
    enable_profile('MI100')
    profile_start('MI100')
    for i in range(n_run):
        torch.cuda.synchronize()
        timer.start()
        _ = model.forward(x, k, v)
        torch.cuda.synchronize()
        timer.log()
    timer.report()
    profile_stop('MI100')