import torch
from torch import nn
from torch import functional as f
from vgg_arch import MultiConv
import time

class VGG16(nn.Module):
    def __init__(self, n_channels, n_classes, device=0):
        super(VGG16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = device

        self.conv1 = MultiConv(3, 64, 2, self.device)
        self.conv2 = MultiConv(64, 128, 2, self.device)
        self.conv3 = MultiConv(128, 256, 3, self.device)
        self.conv4 = MultiConv(256, 512, 3, self.device)
        self.conv5 = MultiConv(512, 512, 3, self.device)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.maxpool(x))
        x = self.conv3(self.maxpool(x))
        x = self.conv4(self.maxpool(x))
        x = self.conv5(self.maxpool(x))
        return x

class VGG16Base(nn.Module):
    def __init__(self, n_channels, n_classes, device=0):
        super(VGG16Base, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = device

        self.conv1_1 = MultiConv(3, 64, 1, self.device)
        self.conv1_2 = MultiConv(64, 64, 1, self.device)
        self.conv2_1 = MultiConv(64, 128, 1, self.device)
        self.conv2_2 = MultiConv(128, 128, 1, self.device)
        self.conv3_1 = MultiConv(128, 256, 1, self.device)
        self.conv3_2 = MultiConv(256, 256, 1, self.device)
        self.conv3_3 = MultiConv(256, 256, 1, self.device)
        self.conv4_1 = MultiConv(256, 512, 1, self.device)
        self.conv4_2 = MultiConv(512, 512, 1, self.device)
        self.conv4_3 = MultiConv(512, 512, 1, self.device)
        self.conv5_1 = MultiConv(512, 512, 1, self.device)
        self.conv5_2 = MultiConv(512, 512, 1, self.device)
        self.conv5_3 = MultiConv(512, 512, 1, self.device)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1_2(self.conv1_1(x))
        x = self.conv2_2(self.conv2_1(self.maxpool(x)))
        x = self.conv3_3(self.conv3_2(self.conv3_1(self.maxpool(x))))
        x = self.conv4_3(self.conv4_2(self.conv4_1(self.maxpool(x))))
        x = self.conv5_3(self.conv5_2(self.conv5_1(self.maxpool(x))))
        return x

if __name__ == "__main__":
    torch.random.manual_seed(0)
    repeats=5
    x = torch.rand(1, 3, 8192, 8192)
    with torch.no_grad():
        print("evaluating VGG Welder Optimized")
        net = VGG16(3, 1000)
        for i in range(repeats):
            start = time.time()
            _ = net(x)
            end = time.time()
            print(end - start)

        print("evaluating VGG Welder Base")
        net = VGG16Base(3, 1000)
        for i in range(repeats):
            start = time.time()
            _ = net(x)
            end = time.time()
            print(end - start)
