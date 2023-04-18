from symbol import testlist_comp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import tqdm
from torch.autograd import Variable
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
arguments = parser.parse_args()
platform = arguments.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

n_warmup = 100
n_run = 100

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out


class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        x = torch.cat((x, torch.zeros_like(x)), dim=1)
        return x


class FlatResNet32(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32, self).__init__()

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.avgpool = nn.AvgPool2d(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def seed(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        return x


    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample


class BlockDrop(nn.Module):
    def __init__(self, rnet) -> None:
        super().__init__()
        self.rnet = rnet
    
    def forward(self, inputs):
        # FlatResNet
        x = self.rnet.seed(inputs)

        # layer 00
        residual = self.rnet.ds[0](x)
        fx = self.rnet.blocks[0][0](x)
        x = torch.relu(residual + fx)
        
        # layer 01
        residual = x
        fx = self.rnet.blocks[0][1](x)
        x = torch.relu(residual + fx)
        
        # layer 02
        residual = x
        fx = self.rnet.blocks[0][2](x)
        x = torch.relu(residual + fx)
        
        # layer 03
        residual = x
        fx = self.rnet.blocks[0][3](x)
        x = torch.relu(residual + fx)

        # layer 04
        residual = x
        fx = self.rnet.blocks[0][4](x)
        x = torch.relu(residual + fx)

        # layer 10
        residual = self.rnet.ds[1](x)
        fx = self.rnet.blocks[1][0](x)
        x = torch.relu(residual + fx)

        # layer 11
        residual = x
        fx = self.rnet.blocks[1][1](x)
        x = torch.relu(residual + fx)

        # layer 12
        residual = x
        fx = self.rnet.blocks[1][2](x)
        x = torch.relu(residual + fx)

        # layer 13
        residual = x
        fx = self.rnet.blocks[1][3](x)
        x = torch.relu(residual + fx)

        # layer 14
        residual = x
        fx = self.rnet.blocks[1][4](x)
        x = torch.relu(residual + fx)

        # layer 20
        residual = self.rnet.ds[2](x)
        fx = self.rnet.blocks[2][0](x)
        x = torch.relu(residual + fx)
        
        # layer 21
        residual = x
        fx = self.rnet.blocks[2][1](x)
        x = torch.relu(residual + fx)

        # layer 22
        residual = x
        fx = self.rnet.blocks[2][2](x)
        x = torch.relu(residual + fx)

        # layer 23
        residual = x
        fx = self.rnet.blocks[2][3](x) 
        x = torch.relu(residual + fx)

        # layer 24
        residual = x
        fx = self.rnet.blocks[2][4](x)
        x = torch.relu(residual + fx)

        x = self.rnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.rnet.fc(x)
        return x
 

layer_config = [5, 5, 5]
rnet = FlatResNet32(BasicBlock, layer_config, num_classes=10)
rnet.eval().cuda()
torch.manual_seed(0)
model = BlockDrop(rnet).eval()

def run(batch_size):
    inputs = torch.randn(batch_size, 3, 32, 32).cuda()
    # print("----batch_size={}---torchscript={}----".format(batch_size, False))
    # # warmup
    # for i in range(n_warmup):
    #     torch.cuda.synchronize()
    #     _ = model(inputs)
    #     torch.cuda.synchronize()
    # # run
    # timer = Timer("ms")
    # for i in range(n_run):
    #     torch.cuda.synchronize()
    #     timer.start()
    #     _ = model(inputs)
    #     torch.cuda.synchronize()
    #     timer.log()
    # timer.report()
    print("----batch_size={}---torchscript={}----".format(batch_size, True))
    script_model = torch.jit.script(model)
    # warmup
    for i in range(n_warmup):
        torch.cuda.synchronize()
        _ = script_model(inputs)
        torch.cuda.synchronize()
    # run
    timer = Timer("ms")
    profile_start(platform)
    for i in range(n_run):
        torch.cuda.synchronize()
        timer.start()
        _ = script_model(inputs)
        torch.cuda.synchronize()
        timer.log()
    timer.report()
    profile_stop(platform)
    torch.onnx.export(script_model, (inputs, ), f"resnet18.b{batch_size}.onnx", opset_version=12)
    
if __name__ == '__main__':
    with torch.no_grad():
        run(arguments.bs)