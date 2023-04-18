# https://flax.readthedocs.io/en/latest/advanced_topics/convert_pytorch_to_flax.html

import numpy as np
from functools import partial
from typing import Any, Sequence

from flax import linen as nn
import jax.numpy as jnp
import jax

import os
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
parser.add_argument('--rate', type=int, default=-1)
parser.add_argument('--rand_weight', dest='real_weight', action='store_false')
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

def conv_weight_from_torch(weight, prefix):
    if prefix.startswith("."):
        prefix = prefix[1:]
    kernel = jnp.array(weight[prefix + ".weight"].cpu()).transpose(2, 3, 1, 0)
    state = {'kernel': kernel}
    if prefix + ".bias" in weight:
        bias = jnp.array(weight[prefix + ".bias"].cpu())
        state['bias'] = bias
    return state, None


def bn_weight_from_torch(weight, prefix):
    if prefix.startswith("."):
        prefix = prefix[1:]
    return {
        'scale': jnp.array(weight[prefix + ".weight"].cpu()),
        'bias': jnp.array(weight[prefix + ".bias"].cpu()),
    }, {
        'mean': jnp.array(weight[prefix + ".running_mean"].cpu()),
        'var': jnp.array(weight[prefix + ".running_var"].cpu()),
    }


def dense_weight_from_torch(weight, prefix):
    if prefix.startswith("."):
        prefix = prefix[1:]
    kernel = jnp.array(weight[prefix + ".weight"].cpu()).transpose(1, 0)
    bias = jnp.array(weight[prefix + ".bias"].cpu())
    return {
        'kernel': kernel,
        'bias': bias,
    }, None


def lstm_weight_from_torch(weight, prefix):
    if prefix.startswith("."):
        prefix = prefix[1:]
    sz = 10
    weight_hh_l0 = jnp.array(weight[prefix + ".weight_hh_l0"].cpu()).reshape(4, sz, sz).transpose(0, 2, 1)
    weight_ih_l0 = jnp.array(weight[prefix + ".weight_ih_l0"].cpu()).reshape(4, sz, sz).transpose(0, 2, 1)
    bias_hh_l0 = jnp.array(weight[prefix + ".bias_hh_l0"].cpu()).reshape(4, sz)
    bias_ih_l0 = jnp.array(weight[prefix + ".bias_ih_l0"].cpu()).reshape(4, sz)
    return {
        'hi': {'kernel': weight_hh_l0[0], 'bias': bias_hh_l0[0] + bias_ih_l0[0]},
        'hf': {'kernel': weight_hh_l0[1], 'bias': bias_hh_l0[1] + bias_ih_l0[1]},
        'hg': {'kernel': weight_hh_l0[2], 'bias': bias_hh_l0[2] + bias_ih_l0[2]},
        'ho': {'kernel': weight_hh_l0[3], 'bias': bias_hh_l0[3] + bias_ih_l0[3]},
        'ii': {'kernel': weight_ih_l0[0]},
        'if': {'kernel': weight_ih_l0[1]},
        'ig': {'kernel': weight_ih_l0[2]},
        'io': {'kernel': weight_ih_l0[3]},
    }, None


class DownSample(nn.Module):
    planes: int
    stride: int
    block_expansion: int
    def setup(self):
        self.conv = nn.Conv(self.planes * self.block_expansion,
                          kernel_size=(1, 1), strides=(self.stride, self.stride), use_bias=False)
        self.bn = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
    
    @staticmethod
    def load_weight(weight, prefix):
        conv_params, _ = conv_weight_from_torch(weight, prefix + ".0")
        bn_params, bn_state = bn_weight_from_torch(weight, prefix + ".1")
        return {
            "conv": conv_params,
            "bn": bn_params,
        }, {
            "bn": bn_state,
        }
    
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
        self.conv2 = nn.Conv(self.planes, kernel_size=(3, 3), strides=(1, 1), padding=[(1,1), (1,1)], use_bias=False)
        self.bn2 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.conv3 = nn.Conv(self.planes * 4, kernel_size=(1, 1), use_bias=False)
        self.bn3 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)

    @staticmethod
    def load_weight(weight, prefix):
        conv1_params, _ = conv_weight_from_torch(weight, prefix + ".conv1")
        bn1_params, bn1_stats = bn_weight_from_torch(weight, prefix + ".bn1")
        conv2_params, _ = conv_weight_from_torch(weight, prefix + ".conv2")
        bn2_params, bn2_stats = bn_weight_from_torch(weight, prefix + ".bn2")
        conv3_params, _ = conv_weight_from_torch(weight, prefix + ".conv3")
        bn3_params, bn3_stats = bn_weight_from_torch(weight, prefix + ".bn3")
        return {
            'conv1': conv1_params,
            'bn1': bn1_params,
            'conv2': conv2_params,
            'bn2': bn2_params,
            'conv3': conv3_params,
            'bn3': bn3_params,
        }, {
            'bn1': bn1_stats,
            'bn2': bn2_stats,
            'bn3': bn3_stats,
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
        self.conv2 = nn.Conv(self.planes, kernel_size=(3, 3), strides=(1, 1), padding=[(1,1), (1,1)], use_bias=False)
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
        # jax.debug.print("out={out}", out=out.transpose(0, 3, 1, 2))
        return out


class CondBottleneck(nn.Module):
    bottleneck: ModuleDef

    @staticmethod
    def load_weight(weight, prefix):
        params, stats = Bottleneck.load_weight(weight, prefix)
        return {
            'bottleneck': params,
        }, {
            'bottleneck': stats,
        }

    @staticmethod
    def load_ds_weight(weight, prefix):
        params, stats = BottleneckDownSample.load_weight(weight, prefix)
        return {
            'bottleneck': params,
        }, {
            'bottleneck': stats,
        }

    @nn.compact
    def __call__(self, action, x, residual, cond_control):
        def resnet_true_branch(self, x, residual, mask):
            x = self.bottleneck(x)
            out = mask * x + (1 - mask) * residual
            return out

        def resnet_false_branch(self, x, residual, mask):
            return residual

        if self.is_mutable_collection('params'):
            return resnet_true_branch(self, x, residual, action)
        else:
            sum_is_positive = jnp.sum(action) + cond_control > 0
            out = nn.cond(sum_is_positive, resnet_true_branch, resnet_false_branch, self, x, residual, action)
            return out
    


# need check
class LSTMCell(nn.Module):

    def setup(self):
        self.lstm = nn.OptimizedLSTMCell()

    @staticmethod
    def load_weight(weight, prefix):
        lstm_params, lstm_stats = lstm_weight_from_torch(weight, prefix)
        return {
            'lstm': lstm_params,
        }, {
            'lstm': lstm_stats,
        }

    def __call__(self, x, state):
        return self.lstm(x, state)


class RNNGatePolicy(nn.Module):
    input_dim: int
    hidden_dim: int

    def setup(self):
        self.rnn = LSTMCell()
        self.proj = nn.Conv(1, kernel_size=(1, 1), strides=(1,1))
    
    @staticmethod
    def load_weight(weight, prefix):
        params, stats = LSTMCell.load_weight(weight, prefix + ".rnn")
        conv_params, _ = conv_weight_from_torch(weight, prefix + ".proj")
        return {
            'rnn': params,
            'proj': conv_params,
        }, {
            'rnn': stats,
        }

    def __call__(self, x, carry):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        carry, out = self.rnn(carry, x)
        out = out.reshape(batch_size, 1, 1, self.hidden_dim)
        proj = self.proj(out)
        proj = proj.reshape(batch_size)
        prob = nn.sigmoid(proj)

        cond = jnp.greater(prob, jnp.full_like(prob, 0.5))
        action = jnp.where(cond, jnp.ones_like(prob), jnp.zeros_like(prob))
        action = action.reshape(batch_size, 1, 1, 1)

        return action, carry


class Gate(nn.Module):
    pool_size: int
    embed_dim: int

    def setup(self):
        self.pool = partial(nn.avg_pool, window_shape=(self.pool_size, self.pool_size), strides=(self.pool_size, self.pool_size))
        self.conv = nn.Conv(self.embed_dim, kernel_size=(1, 1), strides=(1,1))
    
    @staticmethod
    def load_weight(weight, prefix):
        conv_params, _ = conv_weight_from_torch(weight, prefix + ".1")
        return {"conv": conv_params}, {}

    def __call__(self, x):
        out = self.pool(x)
        out = self.conv(out)
        return out


class RecurrentGatedRLResNet(nn.Module):
    num_layers: Sequence[int] = (3, 4, 23, 3)
    num_classes: int = 1000
    embed_dim: int = 10
    hidden_dim: int = 10
    block_expansion: int = 4


    def setup(self):
        self.conv1 = nn.Conv(64, kernel_size=(7, 7), strides=(2,2), padding=[(3,3), (3,3)], use_bias=False)
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

        self.control = RNNGatePolicy(self.embed_dim, self.hidden_dim)
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
                                       pool_size=pool_size, with_cond = (i != 0 or group_id != 1))

            # TODO: check the following codes

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, planes, inplanes, stride=1, pool_size=56, with_cond=True):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or inplanes != planes * self.block_expansion:
            downsample = DownSample(planes=planes, stride=stride, block_expansion=self.block_expansion)
        if downsample is None:
            layer = Bottleneck(inplanes, planes, stride=stride)
        else:
            layer = BottleneckDownSample(inplanes, planes, stride=stride)
        if with_cond:
            layer = CondBottleneck(layer)
        inplanes = planes * self.block_expansion

        gate_layer = Gate(pool_size=pool_size, embed_dim=self.embed_dim)
        return (downsample, layer, gate_layer), inplanes
    

    def load_weight(self, weight, prefix):
        conv1_params, _ = conv_weight_from_torch(weight, prefix + ".conv1")
        bn1_params, bn1_state = bn_weight_from_torch(weight, prefix + ".bn1")
        block_param, block_state = {}, {}
        for i in range(len(self.num_layers)):
            param, state = DownSample.load_weight(weight, prefix + ".group{}_ds{}".format(i+1, 0))
            block_param['group{}_ds0'.format(i+1)] = param
            block_state['group{}_ds0'.format(i+1)] = state

            for j in range(self.num_layers[i]):
                if i == 0 and j == 0:
                    param, state = BottleneckDownSample.load_weight(weight, prefix + ".group{}_layer{}".format(i+1, j))
                elif j == 0:
                    param, state = CondBottleneck.load_ds_weight(weight, prefix + ".group{}_layer{}".format(i+1, j))
                else:
                    param, state = CondBottleneck.load_weight(weight, prefix + ".group{}_layer{}".format(i+1, j))
                block_param['group{}_layer{}'.format(i+1, j)] = param
                block_state['group{}_layer{}'.format(i+1, j)] = state

                param, state = Gate.load_weight(weight, prefix + ".group{}_gate{}".format(i+1, j))
                block_param['group{}_gate{}'.format(i+1, j)] = param
                block_state['group{}_gate{}'.format(i+1, j)] = state

        control_params, _ = RNNGatePolicy.load_weight(weight, prefix + ".control")
        fc_params, _ = dense_weight_from_torch(weight, prefix + ".fc")

        return {
            'conv1': conv1_params,
            'bn1': bn1_params,
            'control': control_params,
            'fc': fc_params,
            **block_param,
        }, {
            'bn1': bn1_state,
            ** block_state,
        }


    def __call__(self, x, cond_control):
        batch_size = x.shape[0]
        x = x.transpose(0, 2, 3, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant', constant_values=-jnp.inf)
        x = self.maxpool(x)
        hidden = (jnp.zeros((batch_size, self.hidden_dim)), jnp.zeros((batch_size, self.hidden_dim)))
        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)
        mask, hidden = self.control(gate_feature, hidden)
        prev = x
        layer_id = 0
        for g in range(len(self.num_layers)):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                
                prev = x = getattr(self, 'group{}_layer{}'.format(g+1, i))(mask, x, prev, cond_control[layer_id])

                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                mask, hidden = self.control(gate_feature, hidden)
                layer_id += 1
        x = self.avgpool(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x, hidden


class RecurrentGatedRLResNetUnroll(nn.Module):
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

        self.control = RNNGatePolicy(self.embed_dim, self.hidden_dim)
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
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

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

        gate_layer = Gate(pool_size=pool_size, embed_dim=self.embed_dim)
        return (downsample, layer, gate_layer), inplanes


    def __call__(self, x, action):
        batch_size = x.shape[0]
        x = x.transpose(0, 2, 3, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant', constant_values=-jnp.inf)
        x = self.maxpool(x)
        hidden = (jnp.zeros((batch_size, self.hidden_dim)), jnp.zeros((batch_size, self.hidden_dim)))
        x = self.group1_layer0(x)
        gate_feature = self.group1_gate0(x)
        mask, hidden = self.control(gate_feature, hidden)
        prev = x
        layer_id = 0
        for g in range(len(self.num_layers)):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                
                if layer_id == 0 or action[layer_id - 1] == 1:
                    prev = x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                else:
                    x = prev

                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                mask, hidden = self.control(gate_feature, hidden)
                layer_id += 1
        x = self.avgpool(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x, hidden


def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = np.fromfile(s + ".bin", dtype=dtype).reshape(shape)
    return tensor


if arguments.real_weight:
    import torch
    torch_params = torch.load(os.path.expanduser(os.path.expanduser("../../artifacts/data/skipnet/resnet-101-rnn-imagenet.pth.tar")), map_location=torch.device('cpu'))['state_dict']
prefix = "../../artifacts/data/skipnet/"
inputs_all = read_bin(os.path.join(prefix, "inputs"))
actions_all = read_bin(os.path.join(prefix, "actions"))

len_dataset= 6400

n_warmup = 100
n_run = 100

def recursive_block_ready(out):
    for o in out:
        if hasattr(o, 'block_until_ready'):
            o.block_until_ready()
        else:
            recursive_block_ready(o)


def run(batch_size):
    model = RecurrentGatedRLResNet()
    example_x = inputs_all[:batch_size]
    example_cond_control = actions_all[:batch_size].sum(axis=0).reshape(-1)
    example_cond_control = example_cond_control * 1000000 - 500000 # positive->inf, zero->-inf

    params = model.init(jax.random.PRNGKey(0), example_x, example_cond_control)
    # init_params_shape = jax.tree_map(jnp.shape, init_params)
    # print(init_params_shape)
    # model.apply(init_params, example_x, example_cond_control)
    if arguments.real_weight:
        params, state = model.load_weight(torch_params, "module")
        params = {
            'params': params,
            'batch_stats': state,
        }

    jit_model = jax.jit(model.apply)
    for i in range(0, len_dataset, batch_size):
        if i >= n_warmup * batch_size: break
        inputs = jnp.array(inputs_all[i: i + batch_size]).block_until_ready()
        cond_control = jnp.array(actions_all[:batch_size]).sum(axis=0).reshape(-1)
        cond_control = cond_control * 1000000 - 500000
        cond_control = cond_control.block_until_ready()
        out = jit_model(params, inputs, cond_control)
        recursive_block_ready(out)
        # np.testing.assert_almost_equal(np.array(out), outputs_all[i: i + batch_size], decimal = 4) # wrong answer now
    profile_start(platform)
    timer = Timer("ms")
    for i in range(0, len_dataset, batch_size):
        if i >= n_run * batch_size: break
        # test time
        inputs = jnp.array(inputs_all[i: i + batch_size]).block_until_ready()
        cond_control = jnp.array(actions_all[:batch_size]).sum(axis=0).reshape(-1)
        cond_control = cond_control * 1000000 - 500000
        cond_control = cond_control.block_until_ready()
        timer.start()
        out = jit_model(params, inputs, cond_control)
        recursive_block_ready(out)
        timer.log()
    profile_stop(platform)
    timer.report()


def test_fix_model(unroll, actions):
    if unroll:
        model = RecurrentGatedRLResNetUnroll()
    else:
        model = RecurrentGatedRLResNet()
    key = random.PRNGKey(233)
    batch_size = 1
    x = random.normal(key, (batch_size, 3, 224, 224)).block_until_ready()
    cond_control = jnp.array(actions, dtype=jnp.float32) * 1000000 - 500000 # positive->inf, zero->-inf
    cond_control.block_until_ready()
    if unroll:
        actions = tuple(actions)
        params = model.init(jax.random.PRNGKey(0), x, actions)
        args = (x, actions)
        jit_model = jax.jit(model.apply, static_argnums=(2,))
    else:
        params = model.init(jax.random.PRNGKey(0), x, cond_control)
        args = (x, cond_control)
        jit_model = jax.jit(model.apply)

    for i in range(n_warmup):
        if unroll:
            out = jit_model(params, x, actions)
        else:
            out = jit_model(params, x, cond_control)
        recursive_block_ready(out)

    profile_start(platform)
    timer = Timer("ms")
    for i in range(n_run):
        # test time
        timer.start()
        if unroll:
            out = jit_model(params, x, actions)
        else:
            out = jit_model(params, x, cond_control)
        recursive_block_ready(out)
        timer.log()
    profile_stop(platform)
    timer.report()

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
    test_fix_model(arguments.unroll, actions)
