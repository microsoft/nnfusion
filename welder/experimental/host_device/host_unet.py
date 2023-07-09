import torch
from torch import nn
from torch import functional as f
from unet_arch import DoubleConv, UpConv, Conv, UpSample

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = 0

        self.inc = DoubleConv(n_channels, 64, self.device)
        self.maxpool = nn.MaxPool2d(2)
        self.down1 = DoubleConv(64, 128, self.device)
        self.down2 = DoubleConv(128, 256, self.device)
        self.down3 = DoubleConv(256, 512, self.device)
        self.down4 = DoubleConv(512, 1024, self.device)
        self.up1 = UpConv(1024, 512, self.device)
        self.up2 = UpConv(512, 256, self.device)
        self.up3 = UpConv(256, 128, self.device)
        self.up4 = UpConv(128, 64, self.device)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(self.maxpool(x1))
        x3 = self.down2(self.maxpool(x2))
        x4 = self.down3(self.maxpool(x3))
        x5 = self.down4(self.maxpool(x4))
        x = x5
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        return x

class UNetBase(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetBase, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = 0

        self.inc_a, self.inc_b = Conv(n_channels, 64, self.device), Conv(64, 64, self.device)
        self.maxpool = nn.MaxPool2d(2)
        self.down1a, self.down1b = Conv(64, 128, self.device), Conv(128, 128, self.device)
        self.down2a, self.down2b = Conv(128, 256, self.device), Conv(256, 256, self.device)
        self.down3a, self.down3b = Conv(256, 512, self.device), Conv(512, 512, self.device)
        self.down4a, self.down4b = Conv(512, 1024, self.device), Conv(1024, 1024, self.device)

        self.up1a = UpSample(1024, self.device)
        self.up1b, self.up1c = Conv(1024, 512, self.device), Conv(512, 512, self.device)
        self.up2a = UpSample(512, self.device)
        self.up2b, self.up2c = Conv(512, 256, self.device), Conv(256, 256, self.device)
        self.up3a = UpSample(256, self.device)
        self.up3b, self.up3c = Conv(256, 128, self.device), Conv(128, 128, self.device)
        self.up4a = UpSample(128, self.device)
        self.up4b, self.up4c = Conv(128, 64, self.device), Conv(64, 64, self.device)

    def forward(self, x):
        x1 = self.inc_b(self.inc_a(x))
        x2 = self.down1b(self.down1a(self.maxpool(x1)))
        x3 = self.down2b(self.down2a(self.maxpool(x2)))
        x4 = self.down3b(self.down3a(self.maxpool(x3)))
        x5 = self.down4b(self.down4a(self.maxpool(x4)))
        x = x5
        x = self.up1c(self.up1b(self.up1a(x4, x)))
        x = self.up2c(self.up2b(self.up2a(x3, x)))
        x = self.up3c(self.up3b(self.up3a(x2, x)))
        x = self.up4c(self.up4b(self.up4a(x1, x)))
        return x

import time
torch.random.manual_seed(0)
with torch.no_grad():
    net = UNetBase(3, 1000)
    x = torch.rand(1, 3, 8192, 8192)
    repeats=5
    for i in range(repeats):
        start = time.time()
        _ = net(x)

        end = time.time()
        print(end - start)
