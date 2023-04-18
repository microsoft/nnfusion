import argparse
import logging
from tabnanny import check
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import torch.autograd as autograd
from torch.autograd import Variable
import math
import torch.backends.cudnn as cudnn
import time
from ast_analyzer import workflow_fix_flag, test_torch_eval, workflow_search_flag
from ast_analyzer.utils.argparser import get_parser
from tqdm import tqdm
from ast_analyzer.to_onnx import to_torch_func
from ast_analyzer.utils.timer import Timer
from ast_analyzer.utils.nvprof import enable_profile, profile_start, profile_stop
from ast_analyzer.utils import config

parser = get_parser()
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--rate', type=int, default=-1)

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

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
                 hidden_dim=10, forward_func_name='forward', **kwargs):
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
        self.forward_func_name = forward_func_name

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
        mask, ch, cc = self.control(gate_feature, ch, cc)

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

def forward_real(self, x, ch, cc): # , target_var, reinforce=False):
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
    checkpoint = os.path.expanduser("../data/skipnet/resnet-101-rnn-imagenet.pth.tar")
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

def read_bin(s):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = torch.from_numpy(np.fromfile(s + ".bin", dtype=np.float32)).reshape(shape)
    return tensor

def set_codegen_flag_for_breakdown(bs):
    pass
    # condition in cuda
    # to_torch_func.NNFUSION_CODEGEN_FLAGS = {
    #     'biasadd_fix': True,
    #     'check_result': True,
    #     'conv_cnhw': True,
    #     'max_grid_dim': 240,
    #     'cf_level': 1,
    #     'branch_fine_grained': False,
    #     'branch_split': False
    # }
    # condition in cuda + prefetch small op
    # to_torch_func.NNFUSION_CODEGEN_FLAGS = {
    #     'biasadd_fix': True,
    #     'check_result': True,
    #     'conv_cnhw': True,
    #     'max_grid_dim': 240,
    #     'cf_level': 1,
    #     'branch_fine_grained': False,
    #     'branch_split': True
    # }
    # naive fuse if and else branch
    # to_torch_func.NNFUSION_CODEGEN_FLAGS = {
    #     'biasadd_fix': True,
    #     'check_result': True,
    #     'conv_cnhw': True,
    #     'max_grid_dim': 400,
    #     'cf_level': 1,
    #     'branch_fine_grained': True,
    #     'if_launch_then_else_naive': True,
    #     'branch_split': False
    # }
    # fuse if and else branch
    # to_torch_func.NNFUSION_CODEGEN_FLAGS = {
    #     'biasadd_fix': True,
    #     'check_result': True,
    #     'conv_cnhw': True,
    #     'max_grid_dim': 400,
    #     'cf_level': 1,
    #     'branch_fine_grained': True,
    #     'branch_split': False
    # }
    # fuse if and else branch + prefetch small op
    # to_torch_func.NNFUSION_CODEGEN_FLAGS = {
    #     'biasadd_fix': True,
    #     'check_result': True,
    #     'conv_cnhw': True,
    #     'max_grid_dim': 400,
    #     'cf_level': 1,
    #     'branch_fine_grained': True,
    #     'branch_split': True
    # }
    # d2h + small op to cpu
    # to_torch_func.NNFUSION_CODEGEN_FLAGS = {
    #     'biasadd_fix': True,
    #     'check_result': True,
    #     'conv_cnhw': True,
    #     'max_grid_dim': 400,
    #     'cf_level': 2,
    #     'branch_fine_grained': False,
    #     'branch_split': False,
    #     'enable_cpu': True,
    # }
    # d2h + prefetch small op + small op to cpu
    # to_torch_func.NNFUSION_CODEGEN_FLAGS = {
    #     'biasadd_fix': True,
    #     'check_result': True,
    #     'conv_cnhw': True,
    #     'max_grid_dim': 400,
    #     'cf_level': 2,
    #     'branch_fine_grained': False,
    #     'branch_split': True,
    #     'enable_cpu': True,
    # }
    # no control flow opt
    # to_torch_func.NNFUSION_CODEGEN_FLAGS = {
    #     'biasadd_fix': True,
    #     'check_result': True,
    #     'conv_cnhw': True,
    #     'max_grid_dim': 400,
    #     'cf_level': 2,
    #     'branch_fine_grained': False,
    #     'branch_split': False,
    #     'enable_cpu': False,
    # }

torch.manual_seed(2333)
len_dataset = 6400

def run_sys(model):
    inp = torch.load(f"tmp/skipnet-in/input-imagenet.{args.bs}.pt").cuda()
    ch = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')
    cc = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')

    # inp = torch.randn(torch.Size([args.bs, 1024, 14, 14]), device='cuda')
    # inp = torch.randn(torch.Size([args.bs, 1, 14, 14]), device='cuda')
    # ch = torch.randn((1, args.bs, model.hidden_dim), device='cuda')
    # cc = torch.randn((1, args.bs, model.hidden_dim), device='cuda')
    # mask = torch.ones((args.bs, 1, 1, 1), device='cuda')
    # feature = torch.ones((args.bs, 10, 1, 1), device='cuda')
    model.convl3 = nn.Conv2d(1, 1024, kernel_size=3, stride=1, padding=1, bias=False)
    model = model.cuda().eval()

    torch.set_printoptions(precision=10)
    if args.run_pytorch:
        test_torch_eval(model, (inp, ch, cc), args.profile)
    if args.run_sys:
        if args.breakdown:
            from ast_analyzer.tensor_opt import search_best_flags
            search_best_flags.SEARCH_IF_MOVE_OUT = False
            workflow_search_flag(model, f"skipnet_bs{args.bs}_breakdown", (inp, ch, cc), args.platform, time_measure=False, enable_control_flow=args.cf)
        elif args.cf:
            workflow_search_flag(model, f"skipnet_bs{args.bs}", (inp, ch, cc), args.platform, time_measure=False, enable_control_flow=args.cf)
        else:
            to_torch_func.NNFUSION_CODEGEN_FLAGS = {
                'biasadd_fix': True,
                'check_result': True,
                'conv_cnhw': True,
                'max_grid_dim': 400,
                'cf_level': 2,
                'branch_fine_grained': False,
                'branch_split': False,
                'log_kerneldb_request': config.KERNELDB_REQUEST_FNAME
            }
            workflow_fix_flag(model, f"base_skipnet_bs{args.bs}", (inp, ch, cc), args.platform, time_measure=False, enable_control_flow=args.cf)
        # if not args.cf:
        #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
        #         'biasadd_fix': True,
        #         'check_result': True,
        #         'conv_cnhw': True,
        #         'max_grid_dim': 400,
        #         'cf_level': 2,
        #         'branch_fine_grained': False,
        #         'branch_split': False
        #     }
        # else:
        #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
        #         'biasadd_fix': True,
        #         'check_result': True,
        #         'conv_cnhw': True,
        #         'max_grid_dim': 400,
        #         'cf_level': 1,
        #         'branch_fine_grained': True,
        #         'branch_split': True,
        #         "enable_cpu": False
        #     }
        # # set_codegen_flag_for_breakdown(args.bs)
        # workflow_fix_flag(model, f"skipnet_bs{args.bs}", (inp, ch, cc), args.platform, time_measure=False, enable_control_flow=args.cf)
        if not args.measure: exit(0)
        n_warmup = 100
        n_run = 100
        # warmup
        prefix = "../data/skipnet/"
        inputs_all = read_bin(os.path.join(prefix, "inputs")).cuda()
        # probs_all = read_bin(os.path.join(prefix, "probs")).cuda()
        # outputs_all = read_bin(os.path.join(prefix, "outputs")).cuda()
        for i in range(0, len_dataset, args.bs):
            if i >= n_warmup * args.bs: break
            inputs = inputs_all[i: i + args.bs].contiguous()
            ch = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')
            cc = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')
            # probs = probs_all[i: i + args.bs].contiguous()
            torch.cuda.synchronize()
            _ = model.forward(inputs, ch, cc)
            torch.cuda.synchronize()
        # run
        timer = Timer("ms")
        enable_profile(args.platform)
        profile_start(args.platform)
        for i in range(0, len_dataset, args.bs):
            if i >= n_run * args.bs: break
            inputs = inputs_all[i: i + args.bs].contiguous()
            ch = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')
            cc = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')
            torch.cuda.synchronize()
            timer.start()
            _ = model.forward(inputs, ch, cc)
            torch.cuda.synchronize()
            timer.log()
        timer.report()
        profile_stop(args.platform)


def prepare_data():
    model = load_checkpoint()
    model.eval()
    model.control.rnn.update_param()
    val_loader = prepare_test_data(batch_size=args.bs)
    with torch.no_grad():
        inputs_all = torch.empty((len_dataset, 3, 224, 224))
        actions_all = torch.empty((len_dataset, 32))
        outputs_all = torch.empty((len_dataset, 1000))
        top1 = 0
        cases = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if (batch_idx == 100): break
            inputs, targets = inputs.cuda(), targets.cuda()
            ch = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')
            cc = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')
            # torch.cuda.synchronize()
            # end = time.time()
            outputs, _, _ = model.forward(inputs, ch, cc)
            bs = inputs.shape[0]
            inputs_all[batch_idx * 64: batch_idx * 64 + bs] = inputs.cpu()
            actions_all[batch_idx * 64: batch_idx * 64 + bs] = torch.stack(model.control.actions, 1).cpu()
            model.control.actions.clear()
            outputs_all[batch_idx * 64: batch_idx * 64 + bs] = outputs.cpu()

            pred = outputs.argmax(dim=1)
            top1 += pred.eq(targets).int().sum()
            cases += inputs.shape[0]
        print(f"top1 = {top1}, cases = {cases}")

        prefix = "../data/skipnet/"
        with open(os.path.join(prefix, "inputs.shape"), "w") as f: f.write(" ".join(str(x) for x in inputs_all.shape))
        inputs_all.detach().numpy().tofile(os.path.join(prefix, "inputs.bin"))


def test_model():
    model = load_checkpoint()
    model.eval()
    model.control.rnn.update_param()
    run_sys(model)


def test_model_with_fix_data():
    model = load_checkpoint()
    model.control.rnn.update_param()
    model = model.cuda().eval()
    inp = torch.rand((args.bs, 3, 224, 224)).cuda()
    ch = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')
    cc = torch.zeros((1, args.bs, model.hidden_dim), device='cuda')
    if args.rate == -1:
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
    elif args.rate == 0:
        actions = [0] * 32
    elif args.rate == 25:
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
    elif args.rate == 50:
        actions = [
            0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
        ]
    elif args.rate == 70:
        actions = torch.tensor([
            1, 0,
            1, 0, 1, 1,
            1, 0, 1, 1, 0,
            1, 1, 0, 1, 1,
            0, 1, 1, 0, 1,
            1, 0, 1, 1, 0,
            1, 1, 1,
            1, 0, 1, 1,
        ], dtype=torch.float32).cuda()
    elif args.rate == 75:
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
    elif args.rate == 100:
        actions = [1] * 32
    cond_control = torch.tensor(actions, dtype=torch.float32).cuda()
    cond_control = cond_control * 1000000 - 500000
    torch.set_printoptions(precision=10)
    rate_tag = "skip" if args.rate == -1 else f"{args.rate}"
    if args.unroll:
        to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            'biasadd_fix': True,
            'check_result': True,
            'conv_cnhw': True,
            'max_grid_dim': 400,
            'cf_level': 1,
            'branch_fine_grained': True,
            'branch_split': False
        }
        workflow_fix_flag(model, f"skipnet_bs{args.bs}_unroll_{rate_tag}", (inp, ch, cc,), args.platform, time_measure=args.measure, enable_control_flow=args.cf)
    else:
        to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            'biasadd_fix': True,
            'check_result': True,
            'conv_cnhw': True,
            'max_grid_dim': 400,
            'cf_level': 1,
            'branch_fine_grained': True,
            'branch_split': False,
            "enable_cpu": False
        }
        workflow_fix_flag(model, f"skipnet_bs{args.bs}_fix_{rate_tag}", (inp, ch, cc, cond_control), args.platform, time_measure=args.measure, enable_control_flow=args.cf)
    

if __name__ == '__main__':
    with torch.no_grad():
        # prepare_data()
        if not args.overhead_test:
            RecurrentGatedRLResNet.forward = forward_real
            test_model()
        else:
            if args.unroll:
                if args.rate == -1:
                    RecurrentGatedRLResNet.forward = forward_skip
                else:
                    RecurrentGatedRLResNet.forward = globals()[f"forward_unroll_{args.rate}"]
            else:
                RecurrentGatedRLResNet.forward = forward_fix
            test_model_with_fix_data()