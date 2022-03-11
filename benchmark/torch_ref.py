import torch
import numpy as np
from torch import nn
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=1,
                               stride=1, padding=0, bias=False)

    def forward(self, x):
        out = x
        for i in range(10):
            out = self.conv(out)
        return out

torch.random.manual_seed(0)
net = Net().cuda()
x = torch.rand([1, 64, 64, 64]).cuda()

def profile(net, x):
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        out = net(x)
        print(out)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    return np.mean(times)

tm = profile(net, x)
tm = profile(net, x)
print(tm)
