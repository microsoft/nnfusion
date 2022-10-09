import torch
from torch import nn
from torch import functional as f
import math
from opbase import OpBase

class DoubleConv(OpBase):
    def __init__(self, in_channels, out_channels, device):
        super().__init__(device)
        self.weight_1 = torch.empty(out_channels, in_channels, 3, 3, device=self.device)
        self.bias_1 = torch.zeros(out_channels, device=self.device)
        self.weight_2 = torch.empty(out_channels, out_channels, 3, 3, device=self.device)
        self.bias_2 = torch.zeros(out_channels, device=self.device)
        torch.nn.init.xavier_normal_(self.weight_1)
        torch.nn.init.xavier_normal_(self.weight_2)
        self.out_channels = out_channels

    def get_dependency(self, n):
        _, _, hh, ww = self.tiles[0]
        _, _, i, j = self.get_tile_index(n)
        return [[None, None, [hh * i - 2, hh * (i + 1) + 2], [ww * j - 2, ww * (j + 1) + 2]]]

    def compute(self, args):
        n, x = args[0], args[1][0]
        torch.cuda.set_stream(self.kernel_stream)
        _, _, i, j = self.get_tile_index(n)
        pad_top = 1 if i == 0 else 0
        pad_bottom = 1 if i == self.run_shape[2] - 1 else 0
        pad_left = 1 if j == 0 else 0
        pad_right = 1 if j == self.run_shape[3] - 1 else 0
        sliced_x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
        sliced_x = torch.conv2d(sliced_x, self.weight_1, self.bias_1, padding=0)
        sliced_x = torch.relu(sliced_x)
        sliced_x = torch.nn.functional.pad(sliced_x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
        sliced_x = torch.conv2d(sliced_x, self.weight_2, self.bias_2, padding=0)
        sliced_x = torch.relu(sliced_x)
        self.kernel_stream.synchronize()
        return [sliced_x]

    def __call__(self, x, n_step=4):
        self.inputs = [x]
        self.outputs = [torch.empty(size=[x.shape[0], self.out_channels, x.shape[2], x.shape[3]])]
        self.tiles = [[x.shape[0], self.out_channels, math.ceil(x.shape[2] / n_step), math.ceil(x.shape[3] / n_step)]]
        self.run_pipeline()

        result = self.outputs[0]
        self.outputs, self.inputs = [], []

        return result

    def ref(self, x):
        x = torch.conv2d(x, self.weight_1, self.bias_1, padding=1)
        x = torch.relu(x)
        x = torch.conv2d(x, self.weight_2, self.bias_2, padding=1)
        x = torch.relu(x)
        return x.cpu()

class UpConv(DoubleConv):

    def __init__(self, in_channels, out_channels, device):
        super().__init__(in_channels, out_channels, device)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2).to(self.device).requires_grad_(False)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def get_dependency(self, n):
        _, _, hh, ww = self.tiles[0]
        _, _, i, j = self.get_tile_index(n)
        return [[None, None, [hh * i - 2, hh * (i + 1) + 2], [ww * j - 2, ww * (j + 1) + 2]],
            [None, None, [(hh * i - 2) // 2, (hh * (i + 1) + 2) // 2], [(ww * j - 2) // 2, (ww * (j + 1) + 2) // 2]]]

    def compute(self, args):
        torch.cuda.set_stream(self.kernel_stream)
        n, x0, x = args[0], args[1][0], args[1][1]
        x = torch.cat([x0, self.upsample(x)], dim=1)
        return super().compute([n, [x]])

    def __call__(self, x0, x, n_step=4):
        self.inputs = [x0, x]
        self.outputs = [torch.empty(size=[x0.shape[0], self.out_channels, x0.shape[2], x0.shape[3]])]
        self.tiles = [[x0.shape[0], self.out_channels, math.ceil(x0.shape[2] / n_step), math.ceil(x0.shape[3] / n_step)]]
        self.run_pipeline()

        result = self.outputs[0]
        self.outputs, self.inputs = [], []

        return result

    def ref(self, x0, x):
        x = torch.cat([x0.to(self.device), self.upsample(x.to(self.device))], dim=1)
        x = torch.conv2d(x, self.weight_1, self.bias_1, padding=1)
        x = torch.relu(x)
        x = torch.conv2d(x, self.weight_2, self.bias_2, padding=1)
        x = torch.relu(x)
        return x.cpu()

class Conv(OpBase):
    def __init__(self, in_channels, out_channels, device):
        super().__init__(device)
        self.weight_1 = torch.empty(out_channels, in_channels, 3, 3, device=self.device)
        self.bias_1 = torch.zeros(out_channels, device=self.device)
        torch.nn.init.xavier_normal_(self.weight_1)
        self.out_channels = out_channels

    def get_dependency(self, n):
        _, _, hh, ww = self.tiles[0]
        _, _, i, j = self.get_tile_index(n)
        return [[None, None, [hh * i - 1, hh * (i + 1) + 1], [ww * j - 1, ww * (j + 1) + 1]]]

    def compute(self, args):
        n, x = args[0], args[1][0]
        torch.cuda.set_stream(self.kernel_stream)
        _, _, i, j = self.get_tile_index(n)
        pad_top = 1 if i == 0 else 0
        pad_bottom = 1 if i == self.run_shape[2] - 1 else 0
        pad_left = 1 if j == 0 else 0
        pad_right = 1 if j == self.run_shape[3] - 1 else 0
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
        x = torch.conv2d(x, self.weight_1, self.bias_1, padding=0)
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
        x = torch.conv2d(x, self.weight_1, self.bias_1, padding=1)
        return x.cpu()

class UpSample(OpBase):
    def __init__(self, in_channels, device):
        super().__init__(device)
        self.out_channels = in_channels
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2).to(self.device).requires_grad_(False)

    def get_dependency(self, n):
        _, _, hh, ww = self.tiles[0]
        _, _, i, j = self.get_tile_index(n)
        return [[None, None, [hh * i, hh * (i + 1)], [ww * j, ww * (j + 1)]],
            [None, None, [(hh * i) // 2, (hh * (i + 1)) // 2], [(ww * j) // 2, (ww * (j + 1)) // 2]]]

    def compute(self, args):
        torch.cuda.set_stream(self.kernel_stream)
        _, x0, x = args[0], args[1][0], args[1][1]
        x = torch.cat([x0, self.upsample(x)], dim=1)
        return [x]

    def __call__(self, x0, x, n_step=4):
        self.inputs = [x0, x]
        self.outputs = [torch.empty(size=[x0.shape[0], self.out_channels, x0.shape[2], x0.shape[3]])]
        self.tiles = [[x0.shape[0], self.out_channels, math.ceil(x0.shape[2] / n_step), math.ceil(x0.shape[3] / n_step)]]
        self.run_pipeline()

        result = self.outputs[0]
        self.outputs, self.inputs = [], []

        return result
