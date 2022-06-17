'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PA(nn.Module):
    '''PA is pixel attention'''

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class PA_UP(nn.Module):
    def __init__(self, nf, unf, out_nc, scale=4, conv=nn.Conv2d):
        super(PA_UP, self).__init__()
        self.upconv1 = conv(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = conv(unf, unf, 3, 1, 1, bias=True)

        if scale == 4:
            self.upconv2 = conv(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = conv(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = conv(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, fea):
        fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.att1(fea))
        fea = self.lrelu(self.HRconv1(fea))
        fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.att2(fea))
        fea = self.lrelu(self.HRconv2(fea))
        fea = self.conv_last(fea)
        return fea

class PA_UP_Dropout(nn.Module):
    def __init__(self, nf, unf, out_nc, scale=4, conv=nn.Conv2d, P=0.7):
        super(PA_UP_Dropout, self).__init__()
        self.upconv1 = conv(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = conv(unf, unf, 3, 1, 1, bias=True)

        if scale == 4:
            self.upconv2 = conv(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = conv(unf, unf, 3, 1, 1, bias=True)
        
        self.dropout = nn.Dropout(p=P)
        self.conv_last = conv(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, fea):
        fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.att1(fea))
        fea = self.lrelu(self.HRconv1(fea))
        fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.att2(fea))
        fea = self.lrelu(self.HRconv2(fea))
        fea = self.dropout(fea)
        fea = self.conv_last(fea)
        return fea

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        super(PixelShuffleDirect, self).__init__()
        self.upsampleOneStep = UpsampleOneStep(scale, num_feat, num_out_ch, input_resolution=None)

    def forward(self, x):
        return self.upsampleOneStep(x)


class PixelShuffleBlcok(nn.Module):
    def __init__(self, in_feat, num_feat, num_out_ch):
        super(PixelShuffleBlcok, self).__init__()
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(in_feat, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x


class NearestConv(nn.Module):
    def __init__(self, in_ch, num_feat, num_out_ch, conv=nn.Conv2d):
        super(NearestConv, self).__init__()
        self.conv_before_upsample = nn.Sequential(
            conv(in_ch, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_up1 = conv(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = conv(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = conv(num_feat, num_feat, 3, 1, 1)
        self.conv_last = conv(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        return x


class NearestConvDropout(nn.Module):
    def __init__(self, in_ch, num_feat, num_out_ch, P=0.7):
        super(NearestConvDropout, self).__init__()
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(in_ch, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dropout = nn.Dropout(p=P)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.act(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.act(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.dropout(x)
        x = self.conv_last(self.act(self.conv_hr(x)))
        return x