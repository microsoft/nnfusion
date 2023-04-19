# Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions


from torch.utils.cpp_extension import load

import argparse
import torch
import torch.nn as nn
import sys
import ctypes
import numpy as np
from ast_analyzer.utils.timer import Timer
from ast_analyzer.shape_inference.types import *
from ast_analyzer import workflow_fix_flag, test_torch_eval, workflow_search_flag
from ast_analyzer.utils.argparser import get_parser
from ast_analyzer.utils import config
from ast_analyzer.to_onnx import to_torch_func
import os
import random

from ast_analyzer.utils.nvprof import enable_profile, profile_start, profile_stop
parser = get_parser()
from ast_analyzer.tensor_opt import buttom_up_feed
buttom_up_feed.SEARCH_ALL_SUBAST = True

parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--opt', type=int, default=-1)
args = parser.parse_args()
torch.manual_seed(0)

depth = 7
n = 2 ** depth - 1

class RAE(nn.Module):
    def __init__(self):
        super(RAE, self).__init__()
        self.encoder = nn.Linear(1024, 512)
        # self.encoder.weight.data.cpu().detach().numpy().tofile("tmp/rae-in/weight.bin")
        # self.encoder.bias.data.cpu().detach().numpy().tofile("tmp/rae-in/bias.bin")

    def forward(
        self,
        left: TyTorchTensor(np.int64, (127,)),
        right: TyTorchTensor(np.int64, (127,)),
        is_leaf: TyTorchTensor(np.bool_, (127,)),
        inp: TyTorchTensor(np.float32, (127, 512)),
        root: TyInt()
    ) -> TyTorchTensor(np.float32, (512,)):
        if is_leaf[root]:
            output = inp[root] # (h,)
        else:
            a = self.forward(left, right, is_leaf, inp, left[root].item()) # (h,)
            b = self.forward(left, right, is_leaf, inp, right[root].item()) # (h,)
            ab = torch.cat((a, b)) # (2h,)
            e = self.encoder(ab)
            output = torch.tanh(e)
        return output


def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = torch.from_numpy(np.fromfile(s + ".bin", dtype=dtype)).reshape(shape).cuda()
    return tensor


def load_trees():
    prefix = '../data/sst/'
    left_tensor = read_bin(prefix + 'left', np.int64)
    right_tensor = read_bin(prefix + 'right', np.int64)
    is_leaf_tensor = read_bin(prefix + 'is_leaf', np.bool)
    root_tensor = read_bin(prefix + 'root', np.int64)
    input_tensor = read_bin(prefix + 'input')
    output_tensor = read_bin(prefix + 'output')
    return left_tensor, right_tensor, is_leaf_tensor, root_tensor, input_tensor, output_tensor


def main_full():
    torch.manual_seed(2333)
    device = torch.device("cuda")
    model = RAE().to(device).eval()

    left = torch.zeros(n, dtype=torch.int64)
    right = torch.zeros(n, dtype=torch.int64)
    is_leaf = torch.ones(n, dtype=torch.bool)
    root = 64
    left[:65] = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, 28, 27, 26, 33, 25, 24, 23, 22, 21, 20, 32, 18, 17, 15, 14, 13, 12, 11, 46, 8, 6, 5, 53, 10, 52, 2, 1, 0, 4, 58, 44])
    right[:65] = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 19, 45, 16, 47, 48, 49, 50, 51, 9, 7, 54, 55, 56, 57, 3, 59, 60, 61, 62, 63])
    is_leaf[:65] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    x = torch.randn([n, 512], device=device)
    # x.cpu().detach().numpy().tofile("tmp/rae-in/x.bin")
    left = left.cuda()
    right = right.cuda()
    is_leaf = is_leaf.cuda()
    torch.cuda.synchronize()
    torch.set_printoptions(threshold=50)
    out = model.forward(left, right, is_leaf, x, root)
    if args.run_pytorch:
        test_torch_eval(model, (left, right, is_leaf, x, root), args.profile)
    if args.run_sys:
        # best
        to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            'recursive_stack': True,
            'recursive_unroll_depth': 4,
            'max_grid_dim': 480,
        }
        # recursive in cuda
        # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'recursive_stack': False, 'stack_size': 65536}
        # manual stack with global memory
        # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'recursive_stack': True, 'stack_in_glb': True}
        # manual stack with shared memory
        # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'recursive_stack': True}
        workflow_fix_flag(model, 'rae', (left, right, is_leaf, x, root), args.platform, args.measure, allow_233=True, enable_control_flow=args.cf)
        # with open("tmp/bin/output_ref_0.bin", "wb") as f:
        #     out.cpu().detach().numpy().tofile(f)


def main_sst():
    torch.manual_seed(2333)
    device = torch.device("cuda")
    model = RAE().to(device).eval()

    left = torch.zeros(n, dtype=torch.int64)
    right = torch.zeros(n, dtype=torch.int64)
    is_leaf = torch.ones(n, dtype=torch.bool)

    root = 64
    left[:65] = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, 28, 27, 26, 33, 25, 24, 23, 22, 21, 20, 32, 18, 17, 15, 14, 13, 12, 11, 46, 8, 6, 5, 53, 10, 52, 2, 1, 0, 4, 58, 44])
    right[:65] = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 19, 45, 16, 47, 48, 49, 50, 51, 9, 7, 54, 55, 56, 57, 3, 59, 60, 61, 62, 63])
    is_leaf[:65] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    x = torch.randn([n, 512], device=device)
    # x.cpu().detach().numpy().tofile("tmp/rae-in/x.bin")
    left = left.cuda()
    right = right.cuda()
    is_leaf = is_leaf.cuda()

    torch.cuda.synchronize()
    torch.set_printoptions(threshold=50)
    out = model.forward(left, right, is_leaf, x, root)
    if args.run_pytorch:
        test_torch_eval(model, (left, right, is_leaf, x, root), args.profile)
    if args.run_sys:
        # best
        if args.breakdown:
            if args.opt == 1:
                to_torch_func.NNFUSION_CODEGEN_FLAGS = {'recursive_stack': False, 'stack_size': 65536, 'max_grid_dim': 64}
            elif args.opt == 2:
                to_torch_func.NNFUSION_CODEGEN_FLAGS = {'recursive_stack': True, 'stack_in_glb': True}
            elif args.opt == 3:
                to_torch_func.NNFUSION_CODEGEN_FLAGS = {'recursive_stack': True}
            elif args.opt == 4:
                to_torch_func.NNFUSION_CODEGEN_FLAGS = {
                    'recursive_stack': True,
                    'recursive_unroll_depth': 4,
                    'max_grid_dim': 480,
                }
            else:
                raise NotImplementedError
            # to_torch_func.NNFUSION_CODEGEN_FLAGS['check_result'] = True
            workflow_fix_flag(model, 'rae_breakdown', (left, right, is_leaf, x, root), args.platform, False, allow_233=True, enable_control_flow=args.cf)
        elif args.cf:
            workflow_search_flag(model, 'rae', (left, right, is_leaf, x, root), args.platform, False, allow_233=True, enable_control_flow=args.cf)
        else:
            to_torch_func.NNFUSION_CODEGEN_FLAGS = {
                'log_kerneldb_request': config.KERNELDB_REQUEST_FNAME
            }
            workflow_fix_flag(model, 'base_rae', (left, right, is_leaf, x, root), args.platform, False, allow_233=True, enable_control_flow=args.cf)
        # to_torch_func.NNFUSION_CODEGEN_FLAGS = {
        #    'recursive_stack': True,
        #    'recursive_unroll_depth': 4,
        #    'max_grid_dim': 480,
        # }
        # recursive in cuda
        # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'recursive_stack': False, 'stack_size': 65536}
        # manual stack with global memory
        # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'recursive_stack': True, 'stack_in_glb': True}
        # manual stack with shared memory (best in rocm)
        # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'recursive_stack': True}
        # pytorch + our kernel
        # to_torch_func.NNFUSION_CODEGEN_FLAGS = {}
        left_tensor, right_tensor, is_leaf_tensor, root_tensor, input_tensor, output_tensor = load_trees()
        # workflow_fix_flag(model, 'rae', (left, right, is_leaf, x, root), args.platform, False, allow_233=True, enable_control_flow=args.cf)
        if not args.measure: exit(0)
        n_warmup = 100
        n_run = 100
        for i in range(n_warmup):
            torch.cuda.synchronize()
            out = model.forward(left_tensor[i].clone(), right_tensor[i].clone(), is_leaf_tensor[i].clone(), input_tensor, root_tensor[i].clone())
            np.testing.assert_allclose(out.cpu().detach().numpy(), output_tensor[i].cpu().detach().numpy(), rtol=1e-5, atol=1e-5)
            torch.cuda.synchronize()
        
        timer = Timer('ms')
        enable_profile(args.platform)
        profile_start(args.platform)
        for i in range(n_run):
            left = left_tensor[i].clone()
            right = right_tensor[i].clone()
            is_leaf = is_leaf_tensor[i].clone()
            root = root_tensor[i].clone()
            torch.cuda.synchronize()
            timer.start()
            out = model.forward(left, right, is_leaf, input_tensor, root)
            torch.cuda.synchronize()
            timer.log()
        timer.report()
        profile_stop(args.platform)


if __name__ == '__main__':
    with torch.no_grad():
        if not args.overhead_test:
            main_sst()
        else:
            main_full()
