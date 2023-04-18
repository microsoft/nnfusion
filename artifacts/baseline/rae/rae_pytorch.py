import torch
import torch.nn as nn
from rae_pytorch_unroll import RAEUnroll
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
arguments = parser.parse_args()
platform = arguments.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from time import time
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

torch.manual_seed(0)
cuda_device = torch.device("cuda:0")
n_warmup = 100
n_run = 100

depth = 7
n = 2 ** depth - 1

class RAE(nn.Module):
    def __init__(self):
        super(RAE, self).__init__()
        self.encoder = nn.Linear(1024, 512)

    def forward(self, left, right, is_leaf, inp, root):
        if is_leaf[root]:
            output = inp[root] # (h,)
        else:
            a = self.forward(left, right, is_leaf, inp, left[root].item()) # (h,)
            b = self.forward(left, right, is_leaf, inp, right[root].item()) # (h,)
            ab = torch.cat((a, b)) # (2h,)
            e = self.encoder(ab)
            output = torch.tanh(e)
        # print(root, output)
        return output


class RAECell(nn.Module):
    def __init__(self):
        super(RAECell, self).__init__()
        self.encoder = nn.Linear(1024, 512)

    def forward(self, a, b):
        ab = torch.cat((a, b)) # (2h,)
        e = self.encoder(ab)
        output = torch.tanh(e)
        return output


class RAEScript(nn.Module):
    def __init__(self):
        super(RAEScript, self).__init__()
        self.cell = torch.jit.script(RAECell())
    
    def forward(self, left, right, is_leaf, inp, root):
        if is_leaf[root]:
            output = inp[root] # (h,)
        else:
            a = self.forward(left, right, is_leaf, inp, left[root].item()) # (h,)
            b = self.forward(left, right, is_leaf, inp, right[root].item()) # (h,)
            output = self.cell(a, b)
        return output


def test_model(enable_torch, enable_unroll, batch_size):
    device = torch.device("cuda")
    if enable_unroll:
        model = RAEUnroll().to(device)
    elif enable_torch:
        model = RAEScript().to(device)
    else:
        model = RAE().to(device)

    root = 64
    left = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, 28, 27, 26, 33, 25, 24, 23, 22, 21, 20, 32, 18, 17, 15, 14, 13, 12, 11, 46, 8, 6, 5, 53, 10, 52, 2, 1, 0, 4, 58, 44])
    right = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 19, 45, 16, 47, 48, 49, 50, 51, 9, 7, 54, 55, 56, 57, 3, 59, 60, 61, 62, 63])
    is_leaf = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    x = torch.ones([n, 512], device=device)
    left = left.cuda()
    right = right.cuda()
    is_leaf = is_leaf.cuda()
    if enable_torch and enable_unroll:
        args = (x,)
        model = torch.jit.script(model)
        print(model)
    else:
        args = (left, right, is_leaf, x, root)
    model.eval()

    print("----batch_size={}---torchscript={}----".format(batch_size, enable_torch))
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        _ = model(*args)
        torch.cuda.synchronize()
        # print("Time {} ms".format((time() - t0) * 1000))

    timer = Timer("ms")
    torch.cuda.synchronize()
    profile_start(platform)
    for i in range(n_run):
        timer.start()
        _ = model(*args)
        torch.cuda.synchronize()
        timer.log()
    profile_stop(platform)
    timer.report()

def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = torch.from_numpy(np.fromfile(s + ".bin", dtype=dtype)).reshape(shape).cuda()
    return tensor


def load_trees():
    prefix = '../../artifacts/data/sst/'
    left_tensor = read_bin(prefix + 'left', np.int64)
    right_tensor = read_bin(prefix + 'right', np.int64)
    is_leaf_tensor = read_bin(prefix + 'is_leaf', np.bool)
    root_tensor = read_bin(prefix + 'root', np.int64)
    input_tensor = read_bin(prefix + 'input')
    output_tensor = read_bin(prefix + 'output')
    return left_tensor, right_tensor, is_leaf_tensor, root_tensor, input_tensor, output_tensor

def test_model_sst(enable_torch, enable_unroll, batch_size):
    assert batch_size == 1
    device = torch.device("cuda")
    torch.manual_seed(2333)
    if enable_unroll:
        model = RAEUnroll().to(device)
    elif enable_torch:
        model = RAEScript().to(device)
    else:
        model = RAE().to(device)

    left_tensor, right_tensor, is_leaf_tensor, root_tensor, input_tensor, output_tensor = load_trees()

    n_warmup = 100
    n_run = 100
    for i in range(n_warmup):
        torch.cuda.synchronize()
        out = model(left_tensor[i].clone(), right_tensor[i].clone(), is_leaf_tensor[i].clone(), input_tensor, root_tensor[i].item())
        np.testing.assert_allclose(out.cpu().detach().numpy(), output_tensor[i].cpu().detach().numpy(), rtol=1e-5, atol=1e-5)
        torch.cuda.synchronize()
    
    timer = Timer('ms')
    profile_start(platform)
    for i in range(n_run):
        left = left_tensor[i].clone()
        right = right_tensor[i].clone()
        is_leaf = is_leaf_tensor[i].clone()
        root = root_tensor[i].item()
        torch.cuda.synchronize()
        timer.start()
        out = model(left, right, is_leaf, input_tensor, root)
        torch.cuda.synchronize()
        timer.log()
    timer.report()
    profile_stop(platform)


if __name__ == '__main__':
    with torch.no_grad():
        if not arguments.overhead_test:
            test_model_sst(False, False, arguments.bs)
        else:
            test_model(arguments.unroll, arguments.unroll, arguments.bs) # enable torchscript only when unrolled
