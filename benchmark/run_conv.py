import torch
import numpy as np
from torch import nn
import time
import ctypes
import os.path as osp

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lib = ctypes.CDLL(osp.dirname(__file__) + "/../build/libsimple_conv.so")
        self.conv = nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):
        old = x.clone()
        out = x.clone()
        for i in range(10):
            out, old = old, out
            self.lib.conv_tvm(
                ctypes.c_void_p(old.data_ptr()),
                ctypes.c_void_p(self.conv.weight.data_ptr()),
                ctypes.c_void_p(out.data_ptr())
            )
        return out

    def run_fused_common(self, x):
        out = x.clone()
        old = x.clone()
        self.lib.conv_fused(
            ctypes.c_void_p(old.data_ptr()),
            ctypes.c_void_p(self.conv.weight.data_ptr()),
            ctypes.c_void_p(out.data_ptr())
        )
        return out

torch.random.manual_seed(0)
net = Net().cuda()
x = torch.rand([1, 64, 64, 64]).cuda()

def profile(net, x):
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        out = net.run_fused_common(x)
        # print(out)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    return np.mean(times)

tm = profile(net, x)
tm = profile(net, x)
print(tm)
