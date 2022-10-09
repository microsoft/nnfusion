import torch
import ctypes
import time

libcuda = ctypes.CDLL("libcuda.so")
torch.random.manual_seed(0)

class HostConv():
    def __init__(self, in_channels, out_channels, device):
        self.device = device
        self.weight = torch.rand(out_channels, in_channels, 3, 3, device=self.device)
        self.bias = torch.rand(out_channels, device=self.device)

    def __call__(self, x, n_step=4):
        h, w = x.shape[2], x.shape[3]
        hh = h // n_step
        ww = w // n_step
        result = torch.empty(size=x.shape)
        for i in range(n_step):
            for j in range(n_step):
                out_h = [hh * i, hh * (i + 1)]
                out_w = [ww * j, ww * (j + 1)]
                pad_top = 1 if i == 0 else 0
                pad_bottom = 1 if i == n_step - 1 else 0
                pad_left = 1 if j == 0 else 0
                pad_right = 1 if j == n_step - 1 else 0
                sliced_x = x[:, :, max(out_h[0] - 1, 0):min(out_h[1] + 1, h), max(out_w[0] - 1, 0):min(out_w[1] + 1, w)]
                sliced_x = torch.empty(size=sliced_x.shape, pin_memory=True).copy_(sliced_x).to(self.device, non_blocking=True)
                sliced_x = torch.nn.functional.pad(sliced_x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
                res_part = torch.conv2d(sliced_x, self.weight, self.bias, padding=0)
                output_buffer = torch.empty(size=res_part.shape, pin_memory=True).copy_(res_part)
                result[:, :, out_h[0]:out_h[1], out_w[0]:out_w[1]] = output_buffer
        # x = x.to(self.device)
        # x = torch.conv2d(x, self.weight, self.bias, padding=1).cpu()
        # print(torch.max(torch.abs(x - result)))

        return result

class DoubleConv():
    def __init__(self, in_channels, out_channels, device):
        self.device = device
        self.weight_1 = torch.rand(out_channels, in_channels, 3, 3, device=self.device)
        self.bias_1 = torch.rand(out_channels, device=self.device)
        self.weight_2 = torch.rand(out_channels, in_channels, 3, 3, device=self.device)
        self.bias_2 = torch.rand(out_channels, device=self.device)

    def __call__(self, x, n_step=4):
        h, w = x.shape[2], x.shape[3]
        hh = h // n_step
        ww = w // n_step
        result = torch.empty(size=x.shape)
        for i in range(n_step):
            for j in range(n_step):
                out_h = [hh * i, hh * (i + 1)]
                out_w = [ww * j, ww * (j + 1)]
                pad_top = 1 if i == 0 else 0
                pad_bottom = 1 if i == n_step - 1 else 0
                pad_left = 1 if j == 0 else 0
                pad_right = 1 if j == n_step - 1 else 0
                sliced_x = x[:, :, max(out_h[0] - 2, 0):min(out_h[1] + 2, h), max(out_w[0] - 2, 0):min(out_w[1] + 2, w)]
                sliced_x = torch.empty(size=sliced_x.shape, pin_memory=True).copy_(sliced_x).to(self.device, non_blocking=True)
                sliced_x = torch.nn.functional.pad(sliced_x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
                sliced_x = torch.conv2d(sliced_x, self.weight_1, self.bias_1, padding=0)
                sliced_x = torch.nn.functional.pad(sliced_x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
                res_part = torch.conv2d(sliced_x, self.weight_2, self.bias_2, padding=0)
                output_buffer = torch.empty(size=res_part.shape, pin_memory=True).copy_(res_part)
                result[:, :, out_h[0]:out_h[1], out_w[0]:out_w[1]] = output_buffer
        # x = x.to(self.device)
        # x = torch.conv2d(x, self.weight_1, self.bias_1, padding=1)
        # x = torch.conv2d(x, self.weight_2, self.bias_2, padding=1).cpu()
        # print(torch.max(torch.abs(x - result)))

        return result

# with torch.no_grad():
#     conv1 = HostConv(64, 64, 0)
#     conv2 = HostConv(64, 64, 0)
#     x = torch.rand(1, 64, 8192, 8192)
#     repeats=5
#     for i in range(repeats):
#         if i == repeats - 1:
#             libcuda.cuProfilerStart()
#         start = time.time()
#         _ = conv2(conv1(x))
#         end = time.time()
#         print(end - start)

with torch.no_grad():
    conv = DoubleConv(64, 64, 0)
    x = torch.rand(1, 64, 8192, 8192)
    repeats=5
    for i in range(repeats):
        if i == repeats - 1:
            libcuda.cuProfilerStart()
        start = time.time()
        _ = conv(x)
        torch.cuda.synchronize()
        end = time.time()
        print(end - start)
