import torch
import numpy as np
from torch import nn
import time
import ctypes
import os.path as osp

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.lib = ctypes.CDLL(osp.dirname(__file__) + "/../build/libsimple_conv.so")
        self.mm = nn.Linear(128, 128, bias=False)

    def forward(self, x):
        out = self.mm(x)
        return out

torch.random.manual_seed(0)
net = Net().cuda()
x = torch.rand([4096, 128]).cuda()

def profile(net, x):
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        out = net.forward(x)
        # print(out)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    return np.mean(times)

tm = profile(net, x)
tm = profile(net, x)
print(tm)
