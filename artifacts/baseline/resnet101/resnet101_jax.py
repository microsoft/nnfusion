# https://flax.readthedocs.io/en/latest/advanced_topics/convert_pytorch_to_flax.html

import numpy as np
from functools import partial
from typing import Any, Sequence

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

class DownSample(nn.Module):
    planes: int
    stride: int
    block_expansion: int
    def setup(self):
        self.conv = nn.Conv(self.planes * self.block_expansion,
                          kernel_size=(1, 1), strides=self.stride, use_bias=False)
        self.bn = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
    
    def __call__(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out



class Bottleneck(nn.Module):
    inplanes: int
    planes: int
    expansion: int = 4
    stride: int = 1

    def setup(self):
        self.conv1 = nn.Conv(self.planes, kernel_size=(1, 1), use_bias=False)
        self.bn1 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.conv2 = nn.Conv(self.planes, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)
        self.bn2 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.conv3 = nn.Conv(self.planes * 4, kernel_size=(1, 1), use_bias=False)
        self.bn3 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = nn.relu(out + residual)
        return out


class BottleneckDownSample(nn.Module):
    inplanes: int
    planes: int
    expansion: int = 4
    stride: int = 1

    def setup(self):
        self.conv1 = nn.Conv(self.planes, kernel_size=(1, 1), strides=self.stride, use_bias=False)
        self.bn1 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.conv2 = nn.Conv(self.planes, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)
        self.bn2 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.conv3 = nn.Conv(self.planes * 4, kernel_size=(1, 1), use_bias=False)
        self.bn3 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.downsample = DownSample(self.planes, self.stride, self.expansion)


    @staticmethod
    def load_weight(weight, prefix):
        conv1_params, _ = conv_weight_from_torch(weight, prefix + ".conv1")
        bn1_params, bn1_stats = bn_weight_from_torch(weight, prefix + ".bn1")
        conv2_params, _ = conv_weight_from_torch(weight, prefix + ".conv2")
        bn2_params, bn2_stats = bn_weight_from_torch(weight, prefix + ".bn2")
        conv3_params, _ = conv_weight_from_torch(weight, prefix + ".conv3")
        bn3_params, bn3_stats = bn_weight_from_torch(weight, prefix + ".bn3")
        downsample_params, downsample_stats = DownSample.load_weight(weight, prefix + ".downsample")
        return {
            'conv1': conv1_params,
            'bn1': bn1_params,
            'conv2': conv2_params,
            'bn2': bn2_params,
            'conv3': conv3_params,
            'bn3': bn3_params,
            'downsample': downsample_params,
        }, {
            'bn1': bn1_stats,
            'bn2': bn2_stats,
            'bn3': bn3_stats,
            'downsample': downsample_stats,
        }

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        residual = self.downsample(x)

        out = nn.relu(out + residual)
        return out


class RecurrentGatedRLResNet(nn.Module):
    num_layers: Sequence[int] = (3, 4, 23, 3)
    num_classes: int = 1000
    embed_dim: int = 10
    hidden_dim: int = 10
    block_expansion: int = 4


    def setup(self):
        self.conv1 = nn.Conv(64, kernel_size=(7, 7), strides=2, padding=3, use_bias=False)
        self.bn1 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.relu = nn.relu
        self.maxpool = partial(nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding='VALID') # need check

        # going to have 4 groups of layers. For the easiness of skipping,
        # We are going to break the sequential of layers into a list of layers.
        inplanes = 64
        inplanes = self._make_group(64, self.num_layers[0], inplanes, group_id=1, pool_size=56)
        inplanes = self._make_group(128, self.num_layers[1], inplanes, group_id=2, pool_size=28)
        inplanes = self._make_group(256, self.num_layers[2], inplanes, group_id=3, pool_size=14)
        inplanes = self._make_group(512, self.num_layers[3], inplanes, group_id=4, pool_size=7)

        # self.control = RNNGatePolicy(self.embed_dim, self.hidden_dim)
        # self.control_rng = self.make_rng('control_rng_key')

        self.avgpool = partial(nn.avg_pool, window_shape=(7, 7), strides=(7, 7))
        self.fc = nn.Dense(self.num_classes)

        self.softmax = nn.softmax

    def _make_group(self, planes, layers, inplanes, group_id=1, pool_size=56):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta, inplanes = self._make_layer_v2(planes, inplanes, stride=stride,
                                       pool_size=pool_size)

            # TODO: check the following codes

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            # setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, planes, inplanes, stride=1, pool_size=56):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or inplanes != planes * self.block_expansion:
            downsample = DownSample(planes=planes, stride=stride, block_expansion=self.block_expansion)
        if downsample is None:
            layer = Bottleneck(inplanes, planes, stride=stride)
        else:
            layer = BottleneckDownSample(inplanes, planes, stride=stride)
        inplanes = planes * self.block_expansion

        return (downsample, layer,), inplanes

    def __call__(self, x):
        batch_size = x.shape[0]
        x = x.transpose(0, 2, 3, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant', constant_values=-jnp.inf)
        x = self.maxpool(x)
        x = self.group1_layer0(x)
        prev = x
        layer_id = 0
        for g in range(len(self.num_layers)):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                
                prev = x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                layer_id += 1
        x = self.avgpool(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x



n_warmup = 100
n_run = 100


def run(batch_size):
    model = RecurrentGatedRLResNet()
    key = random.PRNGKey(batch_size) # batch_size is used as seed
    example_x = random.normal(key, (batch_size, 3, 224, 224))
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