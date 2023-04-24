import logging
import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import math
n_warmup = 100
n_run = 100

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
parser.add_argument('--rate', type=int, default=-1)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
arguments = parser.parse_args()
platform = arguments.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

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
    
    def forward_fix(self, x, ch, cc, cond_control):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        prev = x  # input of next layer

        if torch.sum(mask + cond_control[0]) > 0.0:
            x = x
            x = self.group1_layer1(x)
            x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group1_gate1(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[1]) > 0.0:
            x = x
            x = self.group1_layer2(x)
            x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group1_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        # exit(1)
        prev = self.group2_ds0(prev)

        if torch.sum(mask + cond_control[2]) > 0.0:
            x = x
            x = self.group2_layer0(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group2_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[3]) > 0.0:
            x = x
            x = self.group2_layer1(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group2_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[4]) > 0.0:
            x = x
            x = self.group2_layer2(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group2_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[5]) > 0.0:
            x = x
            x = self.group2_layer3(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group2_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        prev = self.group3_ds0(prev)

        if torch.sum(mask + cond_control[6]) > 0.0:
            x = x
            x = self.group3_layer0(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[7]) > 0.0:
            x = x
            x = self.group3_layer1(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[8]) > 0.0:
            x = x
            x = self.group3_layer2(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[9]) > 0.0:
            x = x
            x = self.group3_layer3(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[10]) > 0.0:
            x = x
            x = self.group3_layer4(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate4(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[11]) > 0.0:
            x = x
            x = self.group3_layer5(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate5(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[12]) > 0.0:
            x = x
            x = self.group3_layer6(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate6(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[13]) > 0.0:
            x = x
            x = self.group3_layer7(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate7(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[14]) > 0.0:
            x = x
            x = self.group3_layer8(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate8(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[15]) > 0.0:
            x = x
            x = self.group3_layer9(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate9(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[16]) > 0.0:
            x = x
            x = self.group3_layer10(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate10(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[17]) > 0.0:
            x = x
            x = self.group3_layer11(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate11(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[18]) > 0.0:
            x = x
            x = self.group3_layer12(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate12(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[19]) > 0.0:
            x = x
            x = self.group3_layer13(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate13(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[20]) > 0.0:
            x = x
            x = self.group3_layer14(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate14(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[21]) > 0.0:
            x = x
            x = self.group3_layer15(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate15(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[22]) > 0.0:
            x = x
            x = self.group3_layer16(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate16(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[23]) > 0.0:
            x = x
            x = self.group3_layer17(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate17(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[24]) > 0.0:
            x = x
            x = self.group3_layer18(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate18(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[25]) > 0.0:
            x = x
            x = self.group3_layer19(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate19(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[26]) > 0.0:
            x = x
            x = self.group3_layer20(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate20(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[27]) > 0.0:
            x = x
            x = self.group3_layer21(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate21(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[28]) > 0.0:
            x = x
            x = self.group3_layer22(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate22(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group4_ds0(prev)

        if torch.sum(mask + cond_control[29]) > 0.0:
            x = x
            x = self.group4_layer0(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group4_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[30]) > 0.0:
            x = x
            x = self.group4_layer1(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group4_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask + cond_control[31]) > 0.0:
            x = x
            x = self.group4_layer2(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, ch, cc
    
    def forward_unroll_0(self, x, ch, cc):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        prev = x  # input of next layer

        x = prev
        gate_feature = self.group1_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group1_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        prev = self.group2_ds0(prev)

        x = prev
        gate_feature = self.group2_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group2_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group2_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group2_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group3_ds0(prev)
        x = prev
        gate_feature = self.group3_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate4(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate5(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate6(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate7(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate8(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate9(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate10(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate11(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate12(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate13(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate14(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate15(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate16(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate17(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate18(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate19(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate20(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate21(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate22(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group4_ds0(prev)

        x = prev
        gate_feature = self.group4_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group4_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, ch, cc
    
    def forward_unroll_25(self, x, ch, cc):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        prev = x  # input of next layer

        x = prev
        gate_feature = self.group1_gate1(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        x = x
        x = self.group1_layer2(x)
        x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group1_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        # exit(1)
        prev = self.group2_ds0(prev)

        x = prev
        gate_feature = self.group2_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group2_layer1(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group2_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group2_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        prev = self.group3_ds0(prev)

        x = prev
        gate_feature = self.group3_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer4(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate4(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate5(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate6(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer7(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate7(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate8(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate9(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer10(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate10(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate11(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate12(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer13(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate13(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate14(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate15(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer16(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate16(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate17(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate18(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate19(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate20(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate21(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate22(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group4_ds0(prev)

        x = prev
        gate_feature = self.group4_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group4_layer1(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group4_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, ch, cc

    def forward_unroll_50(self, x, ch, cc):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        prev = x  # input of next layer

        x = prev
        gate_feature = self.group1_gate1(x)

        x = x
        x = self.group1_layer2(x)
        x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group1_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group2_ds0(prev)

        x = prev
        gate_feature = self.group2_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group2_layer1(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group2_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group2_layer3(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        prev = self.group3_ds0(prev)

        x = prev
        gate_feature = self.group3_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer1(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer3(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate4(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer5(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate5(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate6(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer7(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate7(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate8(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer9(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate9(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate10(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer11(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate11(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate12(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer13(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate13(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate14(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer15(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate15(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate16(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer17(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate17(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate18(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer19(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate19(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate20(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer21(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate21(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate22(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group4_ds0(prev)

        x = x
        x = self.group4_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group4_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group4_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group4_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, ch, cc

    def forward_unroll_75(self, x, ch, cc):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        prev = x  # input of next layer

        x = self.group1_layer1(x)
        x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
        prev = x
        gate_feature = self.group1_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group1_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        prev = self.group2_ds0(prev)

        x = x
        x = self.group2_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x
        gate_feature = self.group2_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group2_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group2_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group2_layer3(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group3_ds0(prev)
        x = x
        x = self.group3_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer1(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer3(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate4(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer5(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate5(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer6(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate6(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate7(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer8(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0

        gate_feature = self.group3_gate8(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        x = x
        x = self.group3_layer9(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate9(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev

        gate_feature = self.group3_gate10(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        x = x
        x = self.group3_layer11(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate11(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer12(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate12(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate13(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer14(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate14(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer15(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate15(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate16(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer17(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate17(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer18(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate18(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer19(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate19(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer20(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate20(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer21(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate21(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer22(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate22(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group4_ds0(prev)

        x = x
        x = self.group4_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group4_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group4_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        x = x
        x = self.group4_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, ch, cc

    def forward_unroll_100(self, x, ch, cc):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        prev = x  # input of next layer

        x = x
        x = self.group1_layer1(x)
        x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group1_gate1(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        x = x
        x = self.group1_layer2(x)
        x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group1_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        prev = self.group2_ds0(prev)

        x = x
        x = self.group2_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group2_layer1(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group2_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group2_layer3(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        prev = self.group3_ds0(prev)

        x = x
        x = self.group3_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer1(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer3(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer4(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate4(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer5(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate5(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer6(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate6(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer7(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate7(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer8(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate8(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer9(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate9(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer10(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate10(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer11(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate11(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer12(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate12(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer13(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate13(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer14(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate14(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer15(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate15(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer16(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate16(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer17(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate17(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer18(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate18(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer19(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate19(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer20(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate20(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer21(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate21(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer22(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate22(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group4_ds0(prev)

        x = x
        x = self.group4_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group4_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group4_layer1(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group4_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group4_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, ch, cc
    
    def forward(self, x, ch, cc): # , target_var, reinforce=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        prev = x  # input of next layer

        if torch.sum(mask) > 0.0:
            x = x
            x = self.group1_layer1(x)
            x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group1_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group1_layer2(x)
            x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group1_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group2_ds0(prev)

        if torch.sum(mask) > 0.0:
            x = x
            x = self.group2_layer0(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group2_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group2_layer1(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group2_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group2_layer2(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group2_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group2_layer3(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group2_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        prev = self.group3_ds0(prev)

        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer0(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer1(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer2(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer3(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer4(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate4(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer5(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate5(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer6(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate6(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer7(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate7(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer8(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate8(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer9(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate9(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer10(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate10(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer11(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate11(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer12(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate12(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer13(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate13(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer14(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate14(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer15(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate15(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer16(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate16(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer17(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate17(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer18(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate18(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer19(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate19(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer20(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate20(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer21(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate21(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group3_layer22(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group3_gate22(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group4_ds0(prev)

        if torch.sum(mask) > 0.0:
            x = x
            x = self.group4_layer0(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group4_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group4_layer1(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev
        gate_feature = self.group4_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        if torch.sum(mask) > 0.0:
            x = x
            x = self.group4_layer2(x)
            x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
            prev = x * 1.0
        else:
            x = prev

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, ch, cc


    def forward_skip(self, x, ch, cc):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)

        mask, ch, cc = self.control(gate_feature, ch, cc)
        prev = x  # input of next layer

        x = self.group1_layer1(x)
        x = mask.expand_as(x)*x + (1.0-mask).expand_as(prev)*prev
        prev = x
        gate_feature = self.group1_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group1_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        prev = self.group2_ds0(prev)

        x = x
        x = self.group2_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x
        gate_feature = self.group2_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group2_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group2_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group2_layer3(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group2_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group3_ds0(prev)
        x = x
        x = self.group3_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate2(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer3(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate3(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate4(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer5(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate5(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer6(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate6(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate7(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer8(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0

        gate_feature = self.group3_gate8(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        x = x
        x = self.group3_layer9(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate9(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev

        gate_feature = self.group3_gate10(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        x = x
        x = self.group3_layer11(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate11(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer12(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate12(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group3_gate13(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = x
        x = self.group3_layer14(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate14(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer15(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate15(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate16(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer17(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate17(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer18(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate18(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = prev
        gate_feature = self.group3_gate19(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer20(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate20(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer21(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate21(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        
        x = x
        x = self.group3_layer22(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group3_gate22(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        prev = self.group4_ds0(prev)

        x = x
        x = self.group4_layer0(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0
        gate_feature = self.group4_gate0(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)

        x = prev
        gate_feature = self.group4_gate1(x)
        mask, ch, cc = self.control(gate_feature, ch, cc)
        x = x
        x = self.group4_layer2(x)
        x = mask.expand_as(x) * x + (1.0 - mask).expand_as(prev)*prev
        prev = x * 1.0

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, ch, cc

    def forward2(self, x, ch, cc): # , target_var, reinforce=False):
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
        return x, ch, cc

def load_checkpoint():
    model = RecurrentGatedRLResNet(Bottleneck, [3, 4, 23, 3], embed_dim=10,
                                   hidden_dim=10)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = os.path.expanduser("../../data/skipnet/resnet-101-rnn-imagenet.pth.tar")
    logging.info('=> loading checkpoint `{}`'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.module
    return model


def prepare_test_data(batch_size, shuffle=True, num_workers=4):
    imagenet_dir = '/mnt/zoltan/public/dataset/rawdata'
    valdir = os.path.join(imagenet_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_validate = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transform_validate),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    return val_loader

len_dataset = 6400


def prepare_data():
    model = load_checkpoint()
    model.eval()
    model.control.rnn.update_param()
    batch_size = 64
    val_loader = prepare_test_data(batch_size=batch_size)
    with torch.no_grad():
        inputs_all = torch.empty((len_dataset, 3, 224, 224))
        actions_all = torch.empty((len_dataset, 32))
        outputs_all = torch.empty((len_dataset, 1000))
        ch_all = torch.empty((len_dataset, 10))
        cc_all = torch.empty((len_dataset, 10))
        top1 = 0
        cases = 0
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if (batch_idx == 100): break
            inputs, targets = inputs.cuda(), targets.cuda()
            ch = torch.zeros((1, batch_size, model.hidden_dim), device='cuda')
            cc = torch.zeros((1, batch_size, model.hidden_dim), device='cuda')
            # torch.cuda.synchronize()
            # end = time.time()
            outputs, ch, cc = model.forward(inputs, ch, cc)
            bs = inputs.shape[0]
            inputs_all[batch_idx * 64: batch_idx * 64 + bs] = inputs.cpu()
            actions_all[batch_idx * 64: batch_idx * 64 + bs] = torch.stack(model.control.actions, 1).cpu()
            model.control.actions.clear()
            outputs_all[batch_idx * 64: batch_idx * 64 + bs] = outputs.cpu()
            ch_all[batch_idx * 64: batch_idx * 64 + bs] = ch.cpu()
            cc_all[batch_idx * 64: batch_idx * 64 + bs] = cc.cpu()

            pred = outputs.argmax(dim=1)
            top1 += pred.eq(targets).int().sum()
            cases += inputs.shape[0]
        print(f"top1 = {top1}, cases = {cases}")

        prefix = "../../data/skipnet/"
        with open(os.path.join(prefix, "inputs.shape"), "w") as f: f.write(" ".join(str(x) for x in inputs_all.shape))
        with open(os.path.join(prefix, "actions.shape"), "w") as f: f.write(" ".join(str(x) for x in actions_all.shape))
        with open(os.path.join(prefix, "outputs.shape"), "w") as f: f.write(" ".join(str(x) for x in outputs_all.shape))
        with open(os.path.join(prefix, "ch.shape"), "w") as f: f.write(" ".join(str(x) for x in ch_all.shape))
        with open(os.path.join(prefix, "cc.shape"), "w") as f: f.write(" ".join(str(x) for x in cc_all.shape))
        inputs_all.detach().numpy().tofile(os.path.join(prefix, "inputs.bin"))
        actions_all.detach().numpy().tofile(os.path.join(prefix, "actions.bin"))
        outputs_all.detach().numpy().tofile(os.path.join(prefix, "outputs.bin"))
        ch_all.detach().numpy().tofile(os.path.join(prefix, "ch.bin"))
        cc_all.detach().numpy().tofile(os.path.join(prefix, "cc.bin"))

# prepare_data()
# exit(0)

def read_bin(s):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = torch.from_numpy(np.fromfile(s + ".bin", dtype=np.float32)).reshape(shape)
    return tensor


model = load_checkpoint()
model.eval()
model.control.rnn.update_param()
prefix = "../../data/skipnet/"
inputs_all = read_bin(os.path.join(prefix, "inputs")).cuda()
# actions_all = read_bin(os.path.join(prefix, "actions")).cuda()
# outputs_all = read_bin(os.path.join(prefix, "outputs")).cuda()
torch.manual_seed(233)
def run(batch_size):
    assert(model.forward.__qualname__ != '_forward_unimplemented')
    print("----batch_size={}---torchscript={}----".format(batch_size, True))
    script_model = torch.jit.script(model)
    # warmup
    for i in range(0, len_dataset, batch_size):
        if i >= n_warmup * batch_size: break
        inputs = inputs_all[i: i + batch_size].contiguous()
        ch = torch.zeros((1, batch_size, model.hidden_dim), device='cuda')
        cc = torch.zeros((1, batch_size, model.hidden_dim), device='cuda')
        torch.cuda.synchronize()
        _ = script_model(inputs, ch, cc)
        torch.cuda.synchronize()
    # run
    timer = Timer("ms")
    profile_start(platform)
    for i in range(0, len_dataset, batch_size):
        if i >= n_run * batch_size: break
        inputs = inputs_all[i: i + batch_size].contiguous()
        ch = torch.zeros((1, batch_size, model.hidden_dim), device='cuda')
        cc = torch.zeros((1, batch_size, model.hidden_dim), device='cuda')
        torch.cuda.synchronize()
        timer.start()
        _ = script_model(inputs, ch, cc)
        torch.cuda.synchronize()
        timer.log()
    timer.report()
    profile_stop(platform)
    os.system("mkdir -p onnx")
    torch.onnx.export(script_model, (inputs, ch, cc), f"onnx/skipnet.b{batch_size}.onnx", opset_version=12)

def run_unroll(unroll, action, rate):
    model = load_checkpoint()
    model.eval()
    model.control.rnn.update_param()
    # print(dir(model.forward))
    # assert(model.forward.__qualname__ == '_forward_unimplemented')
    if unroll:
        if rate == -1:
            model.forward = model.forward_skip
            print("run model.forward_skip")
        else:
            model.forward = getattr(model, f"forward_unroll_{rate}")
            print(f"run model.forward_unroll_{rate}")
    else:
        model.forward = model.forward_fix
        print("run model.forward_real", action)
    
    script_model = torch.jit.script(model)
    batch_size = 1
    inputs = torch.randn((batch_size, 3, 224, 224)).cuda()
    # print("inputs:", inputs)
    ch = torch.zeros((1, batch_size, model.hidden_dim), device='cuda')
    cc = torch.zeros((1, batch_size, model.hidden_dim), device='cuda')
    cond_control = torch.tensor(action, dtype=torch.float32).cuda()
    cond_control = cond_control * 1000000 - 500000
    if unroll:
        args = (inputs, ch, cc)
    else:
        args = (inputs, ch, cc, cond_control)

    # warmup
    for i in range(0, len_dataset, batch_size):
        if i >= n_warmup * batch_size: break
        torch.cuda.synchronize()
        out = script_model(*args)
        # if i == 0: print(out)
        torch.cuda.synchronize()
    # run
    timer = Timer("ms")
    profile_start(platform)
    for i in range(0, len_dataset, batch_size):
        if i >= n_run * batch_size: break
        torch.cuda.synchronize()
        timer.start()
        _ = script_model(*args)
        torch.cuda.synchronize()
        timer.log()
    timer.report()
    profile_stop(platform)
    unroll_tag = "unroll" if unroll else "fix"
    rate_tag = "skip" if rate == -1 else f"{rate}"
    os.system("mkdir -p onnx")
    torch.onnx.export(script_model, args, f"onnx/skipnet.b{batch_size}.{unroll_tag}.{rate_tag}.onnx", opset_version=12)


if __name__ == '__main__':
    with torch.no_grad():
        if not arguments.overhead_test:
            run(arguments.bs)
        else:
            if arguments.rate == -1: # real case
                actions = [
                    1, 0,
                    1, 0, 1, 1,
                    1, 0, 1, 1, 0,
                    1, 1, 0, 1, 1,
                    0, 1, 1, 0, 1,
                    1, 0, 1, 1, 0,
                    1, 1, 1,
                    1, 0, 1, 1,
                ]
            elif arguments.rate == 0:
                actions = [0] * 32
            elif arguments.rate == 25:
                actions = [
                    0, 1,
                    0, 1, 0, 0,
                    0, 0, 0, 0, 1,
                    0, 0, 1, 0, 0,
                    1, 0, 0, 1, 0,
                    0, 1, 0, 0, 0,
                    0, 0, 0,
                    0, 1, 0
                ] 
            elif arguments.rate == 50:
                actions = [
                    0, 1, 0, 1, 0, 1, 0, 1,
                    0, 1, 0, 1, 0, 1, 0, 1,
                    0, 1, 0, 1, 0, 1, 0, 1,
                    0, 1, 0, 1, 0, 1, 0, 1,
                ]
            elif arguments.rate == 75:
                actions = [
                    1, 0,
                    1, 0, 1, 1,
                    1, 1, 1, 1, 0,
                    1, 1, 0, 1, 1,
                    0, 1, 1, 0, 1,
                    1, 0, 1, 1, 1,
                    1, 1, 1,
                    1, 0, 1,
                ]
            elif arguments.rate == 100:
                actions = [1] * 32
            else:
                raise NotImplementedError
            run_unroll(arguments.unroll, actions, arguments.rate)
