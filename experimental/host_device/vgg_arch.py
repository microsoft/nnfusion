import torch
from torch import nn
from torch import functional as f
import math
from opbase import OpBase
import time

class MultiConv(OpBase):
    def __init__(self, in_channels, out_channels, num_conv, device):
        super().__init__(device)
        self.weights = []
        self.bias = []
        for i in range(num_conv):
            ch_in = in_channels if i == 0 else out_channels
            weight = torch.empty(out_channels, ch_in, 3, 3, device=self.device)
            torch.nn.init.xavier_normal_(weight)
            self.weights.append(weight)
            self.bias.append(torch.zeros(out_channels, device=self.device))
        self.out_channels = out_channels
        self.num_conv = num_conv

    def get_dependency(self, n):
        _, _, hh, ww = self.tiles[0]
        _, _, i, j = self.get_tile_index(n)
        return [[None, None,
            [hh * i - self.num_conv, hh * (i + 1) + self.num_conv],
            [ww * j - self.num_conv, ww * (j + 1) + self.num_conv]]]

    def compute(self, args):
        n, x = args[0], args[1][0]
        torch.cuda.set_stream(self.kernel_stream)
        _, _, i, j = self.get_tile_index(n)
        pad_top = 1 if i == 0 else 0
        pad_bottom = 1 if i == self.run_shape[2] - 1 else 0
        pad_left = 1 if j == 0 else 0
        pad_right = 1 if j == self.run_shape[3] - 1 else 0
        for i in range(self.num_conv):
            x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
            x = torch.conv2d(x, self.weights[i], self.bias[i], padding=0)
            x = torch.relu(x)
        self.kernel_stream.synchronize()
        return [x]

    def __call__(self, x, n_step=4):
        self.inputs = [x]
        self.outputs = [torch.empty(size=[x.shape[0], self.out_channels, x.shape[2], x.shape[3]])]
        self.tiles = [[x.shape[0], self.out_channels, math.ceil(x.shape[2] / n_step), math.ceil(x.shape[3] / n_step)]]
        self.run_pipeline()

        result = self.outputs[0]
        self.outputs, self.inputs = [], []

        return result

    def ref(self, x):
        x = x.to(self.device)
        for i in range(self.num_conv):
            x = torch.conv2d(x, self.weights[i], self.bias[i], padding=1)
            x = torch.relu(x)
        return x.cpu()
