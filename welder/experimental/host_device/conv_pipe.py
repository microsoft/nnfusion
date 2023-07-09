import torch
import ctypes
import time
import math
from opbase import OpBase

libcuda = ctypes.CDLL("libcuda.so")
torch.random.manual_seed(0)

class DoubleConv(OpBase):
    def __init__(self, in_channels, out_channels, device):
        super().__init__(device)
        self.weight_1 = torch.rand(out_channels, in_channels, 3, 3, device=self.device)
        self.bias_1 = torch.rand(out_channels, device=self.device)
        self.weight_2 = torch.rand(out_channels, in_channels, 3, 3, device=self.device)
        self.bias_2 = torch.rand(out_channels, device=self.device)

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
        sliced_x = torch.nn.functional.pad(sliced_x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
        res_part = torch.conv2d(sliced_x, self.weight_2, self.bias_2, padding=0)
        self.kernel_stream.synchronize()
        return [res_part]

    def __call__(self, x, n_step=4):
        self.inputs = [x]
        self.outputs = [torch.empty(size=x.shape)]
        self.tiles = [[x.shape[0], x.shape[1], math.ceil(x.shape[2] / n_step), math.ceil(x.shape[3] / n_step)]]
        self.run_pipeline()

        result = self.outputs[0]
        self.outputs, self.inputs = [], []

        # x = x.to(self.device)
        # x = torch.conv2d(x, self.weight_1, self.bias_1, padding=1)
        # x = torch.conv2d(x, self.weight_2, self.bias_2, padding=1).cpu()
        # print(torch.max(torch.abs(x - result)))

        return result

class Conv(OpBase):
    def __init__(self, in_channels, out_channels, device):
        super().__init__(device)
        self.weight_1 = torch.rand(out_channels, in_channels, 3, 3, device=self.device)
        self.bias_1 = torch.rand(out_channels, device=self.device)

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
        sliced_x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
        res_part = torch.conv2d(sliced_x, self.weight_1, self.bias_1, padding=0)
        self.kernel_stream.synchronize()
        return [res_part]

    def __call__(self, x, n_step=4):
        self.inputs = [x]
        self.outputs = [torch.empty(size=x.shape)]
        self.tiles = [[x.shape[0], x.shape[1], math.ceil(x.shape[2] / n_step), math.ceil(x.shape[3] / n_step)]]
        self.run_pipeline()

        result = self.outputs[0]
        self.outputs, self.inputs = [], []

        # x = x.to(self.device)
        # x = torch.conv2d(x, self.weight_1, self.bias_1, padding=1).cpu()
        # print(torch.max(torch.abs(x - result)))

        return result


with torch.no_grad():
    conv = DoubleConv(64, 64, 3)
    x = torch.rand(1, 64, 1024, 1024)
    repeats=5
    for i in range(repeats):
        if i == repeats - 1:
            libcuda.cuProfilerStart()
        start = time.time()
        _ = conv(x)
        end = time.time()
        print(end - start)

# with torch.no_grad():
#     conv1, conv2 = Conv(64, 64, 3), Conv(64, 64, 3)
#     x = torch.rand(1, 64, 8192, 8192)
#     repeats=5
#     for i in range(repeats):
#         if i == repeats - 1:
#             libcuda.cuProfilerStart()
#         start = time.time()
#         _ = conv2(conv1(x))
#         end = time.time()
#         print(end - start)
