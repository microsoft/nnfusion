import logging
import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import math
from tqdm import tqdm
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
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out


class BottleneckDownSample(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckDownSample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class LSTMCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.weight_ih_l0 = nn.Parameter(torch.randn(4 * hidden_size, input_size, dtype=torch.float32))
        self.weight_hh_l0 = nn.Parameter(torch.randn(4 * hidden_size, input_size, dtype=torch.float32))
        self.bias_ih_l0 = nn.Parameter(torch.randn(4 * hidden_size, dtype=torch.float32))
        self.bias_hh_l0 = nn.Parameter(torch.randn(4 * hidden_size, dtype=torch.float32))
        self.weight_ih_l0_t = nn.Parameter(torch.empty(4, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh_l0_t = nn.Parameter(torch.empty(4, input_size, hidden_size, dtype=torch.float32))
        self.bias_ih_l0_t = nn.Parameter(torch.empty(4, 1, hidden_size, dtype=torch.float32))
        self.bias_hh_l0_t = nn.Parameter(torch.empty(4, 1, hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        self.input_size = input_size
    
    def update_param(self):
        self.state_dict()[f"weight_ih_l0_t"][:] = torch.transpose(self.weight_ih_l0.view(4, self.hidden_size, self.input_size), 1, 2)
        self.state_dict()[f"bias_ih_l0_t"][:] = self.bias_ih_l0.reshape((4, 1, self.hidden_size))
        self.state_dict()[f"weight_hh_l0_t"][:] = torch.transpose(self.weight_hh_l0.view(4, self.hidden_size, self.input_size), 1, 2)
        self.state_dict()[f"bias_hh_l0_t"][:] = self.bias_hh_l0.reshape((4, 1, self.hidden_size))

    def forward(self, x, h, c):
        ih = torch.matmul(x, self.weight_ih_l0_t) + self.bias_ih_l0_t
        hh = torch.matmul(h, self.weight_hh_l0_t) + self.bias_hh_l0_t
        ih0, ih1, ih2, ih3 = torch.split(ih, (1, 1, 1, 1), dim=0)
        hh0, hh1, hh2, hh3 = torch.split(hh, (1, 1, 1, 1), dim=0)
        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        state_c = (forgetgate * c) + (ingate * cellgate)
        state_h = outgate * torch.tanh(state_c)

        return state_h, state_h, state_c


class RNNGatePolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(RNNGatePolicy, self).__init__()

        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.rnn = LSTMCell(input_dim, hidden_dim)
        self.hidden = None

        self.proj = nn.Conv2d(in_channels=hidden_dim, out_channels=1,
                              kernel_size=1, stride=1)
        # self.actions = []
        # self.prob = nn.Sigmoid()

    def forward(self, x, ch, cc):
        batch_size = x.size(0)
        out, ch, cc = self.rnn(x.view(1, batch_size, self.input_dim), ch, cc)
        out = out.view(batch_size, self.hidden_dim, 1, 1) # need to verify
        proj = self.proj(out)
        proj = proj.view(batch_size) # need to verify
        prob = torch.sigmoid(proj)

        cond = torch.gt(prob, torch.full_like(prob, 0.5))
        action = torch.where(cond, torch.ones_like(prob), torch.zeros_like(prob))
        # self.actions.append(action)
        action = action.view(action.size(0), 1, 1, 1)
        # print(action)
        return action, ch, cc


class RecurrentGatedRLResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, embed_dim=10,
                 hidden_dim=10, **kwargs):
        self.inplanes = 64
        super(RecurrentGatedRLResNet, self).__init__()

        self.num_layers = layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # going to have 4 groups of layers. For the easiness of skipping,
        # We are going to break the sequential of layers into a list of layers.
        self._make_group(block, 64, layers[0], group_id=1, pool_size=56)
        self._make_group(block, 128, layers[1], group_id=2, pool_size=28)
        self._make_group(block, 256, layers[2], group_id=3, pool_size=14)
        self._make_group(block, 512, layers[3], group_id=4, pool_size=7)

        self.control = RNNGatePolicy(embed_dim, hidden_dim, rnn_type='lstm')

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.softmax = nn.Softmax()

        # save everything
        self.saved_actions = {}
        self.saved_dists = {}
        self.saved_outputs = {}
        self.saved_targets = {}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        self.device = torch.device("cuda:0")

    def _make_group(self, block, planes, layers, group_id=1, pool_size=56):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, block, planes, stride=1, pool_size=56):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        if downsample is None:
            layer = Bottleneck(self.inplanes, planes, stride, downsample)
        else:
            layer = BottleneckDownSample(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(in_channels=planes * block.expansion,
                      out_channels=self.embed_dim,
                      kernel_size=1,
                      stride=1))
        return downsample, layer, gate_layer
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.group1_layer0(x)
        x = self.group1_layer1(x)
        x = self.group1_layer2(x)
        x = self.group2_layer0(x)
        x = self.group2_layer1(x)
        x = self.group2_layer2(x)
        x = self.group2_layer3(x)
        x = self.group3_layer0(x)
        x = self.group3_layer1(x)
        x = self.group3_layer2(x)
        x = self.group3_layer3(x)
        x = self.group3_layer4(x)
        x = self.group3_layer5(x)
        x = self.group3_layer6(x)
        x = self.group3_layer7(x)
        x = self.group3_layer8(x)
        x = self.group3_layer9(x)
        x = self.group3_layer10(x)
        x = self.group3_layer11(x)
        x = self.group3_layer12(x)
        x = self.group3_layer13(x)
        x = self.group3_layer14(x)
        x = self.group3_layer15(x)
        x = self.group3_layer16(x)
        x = self.group3_layer17(x)
        x = self.group3_layer18(x)
        x = self.group3_layer19(x)
        x = self.group3_layer20(x)
        x = self.group3_layer21(x)
        x = self.group3_layer22(x)
        x = self.group4_layer0(x)
        x = self.group4_layer1(x)
        x = self.group4_layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

torch.manual_seed(2333)
model = RecurrentGatedRLResNet(Bottleneck, [3, 4, 23, 3], embed_dim=10, hidden_dim=10).cuda().eval()


def run(batch_size):
    inputs = torch.randn(batch_size, 3, 224, 224).cuda()
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
    torch.onnx.export(script_model, (inputs,), f"resnet101.b{batch_size}.onnx", opset_version=12)

if __name__ == '__main__':
    with torch.no_grad():
        run(arguments.bs)
