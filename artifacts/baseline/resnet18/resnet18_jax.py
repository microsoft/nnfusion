# https://flax.readthedocs.io/en/latest/advanced_topics/convert_pytorch_to_flax.html

import numpy as np
from functools import partial
from typing import Any, Callable, Sequence, Tuple, List

from flax import linen as nn
import jax.numpy as jnp
import jax

import torch
import os
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
arguments = parser.parse_args()
platform = arguments.platform

sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)
from jax import random

ModuleDef = Any

def conv3x3(out_planes, strides=1):
    "3x3 convolution with padding"
    return nn.Conv(out_planes, kernel_size=(3, 3), strides=strides, padding=1, use_bias=False)

class BasicBlock(nn.Module):
    planes: int
    stride: int = 1
    def setup(self):
        self.conv1 = conv3x3(self.planes, self.stride)
        self.bn1 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.conv2 = conv3x3(self.planes)
        self.bn2 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        
    def __call__(self, x, residual):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.relu(out + residual)
        return out


class DownsampleB(nn.Module):
    stride: int
    def setup(self):
        self.avg = partial(nn.avg_pool, window_shape=(self.stride, self.stride), strides=(self.stride, self.stride))
    
    def __call__(self, x):
        out = self.avg(x)
        out = jnp.concatenate([out, jnp.zeros_like(out)], axis=3)
        return out


class FlatResNet32(nn.Module):
    inplanes: int = 16
    layers: Sequence[int] = (5, 5, 5)
    num_classes: int = 10

    def setup(self):
        self.conv1 = conv3x3(16)
        self.bn1 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.avgpool = partial(nn.avg_pool, window_shape=(8, 8), strides=(8, 8))

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self_blocks = []
        self_ds = []
        inplanes = self.inplanes

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, self.layers, strides)):
            blocks, ds, inplanes = self._make_layer(inplanes, filt_size, num_blocks, stride=stride)
            self_blocks.append(blocks)
            self_ds.append(ds)
        self.blocks = self_blocks
        self.ds = self_ds

        self.fc = nn.Dense(10)

    def _make_layer(self, inplanes: int, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = DownsampleB(stride)

        layers = []
        layers.append(BasicBlock(planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes))

        return layers, downsample, planes
    
    def __call__(self, inputs):
        inputs = inputs.transpose(0, 2, 3, 1)
        x = nn.relu(self.bn1(self.conv1(inputs)))

        for i in range(len(self.blocks)):
            if self.ds[i] is not None:
                residual = self.ds[i](x)
            else:
                residual = x
            for j in range(len(self.blocks[i])):
                residual = x = self.blocks[i][j](x, residual)

        out = self.avgpool(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = np.fromfile(s + ".bin", dtype=dtype).reshape(shape)
    return tensor


n_warmup = 100
n_run = 100

def run(batch_size):
    model = FlatResNet32()
    key = random.PRNGKey(batch_size) # batch_size is used as seed
    example_x = random.normal(key, (batch_size, 3, 32, 32))
    params = model.init(jax.random.PRNGKey(0), example_x)
    jit_model = jax.jit(model.apply)
    for i in range(n_warmup):
        out = jit_model(params, example_x)
        out.block_until_ready()
    timer = Timer("ms")
    profile_start(platform)
    for i in range(n_run):
        # test time
        timer.start()
        out = jit_model(params, example_x)
        out.block_until_ready()
        timer.log()
    profile_stop(platform)
    timer.report()

run(arguments.bs)
