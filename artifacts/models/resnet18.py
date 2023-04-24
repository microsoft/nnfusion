from ast_analyzer.shape_inference.types import *
from ast_analyzer import workflow_fix_flag
import torch
import torch.nn as nn
import math
import numpy as np
import sys
from ast_analyzer.utils.argparser import get_parser
from ast_analyzer.to_onnx import to_torch_func
parser = get_parser()

parser.add_argument('--bs', type=int, default=1)
args = parser.parse_args()

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
 


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)
    input_size = 256
    batch_size = args.bs
    hidden_size = 256
    seq_len = 1000
    layer_config = [5, 5, 5]
    rnet = FlatResNet32(BasicBlock, layer_config, num_classes=10)
    rnet.eval().cuda()
    torch.manual_seed(0)
    model = BlockDrop(rnet).eval()

    inputs = torch.randn(args.bs, 3, 32, 32).cuda()
    
    model.eval()
    o = model(inputs)

    with torch.no_grad():
        to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            'biasadd_fix': True,
            'check_result': True,
            'conv_cnhw': True,
            'max_grid_dim': 256,
        }
        workflow_fix_flag(model, f"resnet18_bs{args.bs}", (inputs,), args.platform, args.measure, enable_control_flow=args.cf)

    # with torch.no_grad():
    #     workflow_fix_flag(model, 'nasrnn', (inputs,), args.platform, args.measure)