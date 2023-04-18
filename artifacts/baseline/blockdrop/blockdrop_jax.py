# https://flax.readthedocs.io/en/latest/advanced_topics/convert_pytorch_to_flax.html

import numpy as np
from functools import partial
from typing import Any, Callable, Sequence, Tuple, List

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

def conv3x3(out_planes, strides=1):
    "3x3 convolution with padding"
    return nn.Conv(out_planes, kernel_size=(3, 3), strides=(strides,strides), padding=[(1, 1), (1, 1)], use_bias=False)

def conv_weight_from_torch(weight, prefix):
    if prefix.startswith("."):
        prefix = prefix[1:]
    kernel = jnp.array(weight[prefix + ".weight"].cpu()).transpose(2, 3, 1, 0)
    return {'kernel': kernel}, None


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


class BasicBlock(nn.Module):
    planes: int
    stride: int = 1
    def setup(self):
        self.conv1 = conv3x3(self.planes, self.stride)
        self.bn1 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
        self.conv2 = conv3x3(self.planes)
        self.bn2 = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=1e-5)
    
    @staticmethod
    def load_weight(weight, prefix):
        conv1_params, _ = conv_weight_from_torch(weight, prefix + ".conv1")
        bn1_params, bn1_state = bn_weight_from_torch(weight, prefix + ".bn1")
        conv2_params, _ = conv_weight_from_torch(weight, prefix + ".conv2")
        bn2_params, bn2_state = bn_weight_from_torch(weight, prefix + ".bn2")
        return {
            'conv1': conv1_params,
            'bn1': bn1_params,
            'conv2': conv2_params,
            'bn2': bn2_params,
        }, {
            'bn1': bn1_state,
            'bn2': bn2_state,
        }

    
    def __call__(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        return out


class DownsampleB(nn.Module):
    stride: int
    def setup(self):
        self.avg = partial(nn.avg_pool, window_shape=(self.stride, self.stride), strides=(self.stride, self.stride))
    
    def __call__(self, x):
        out = self.avg(x)
        out = jnp.concatenate([out, jnp.zeros_like(out)], axis=3)
        return out


class CondBasicBlock(nn.Module):
    # block: ModuleDef
    planes: int
    stride: int = 1

    @staticmethod
    def load_weight(weight, prefix):
        params, state = BasicBlock.load_weight(weight, prefix)
        return {
            'block': params,
        }, {
            'block': state,
        }

    @nn.compact
    def __call__(self, action, x, residual):
        def resnet_true_branch(self, x, residual, action):
            action_mask = jnp.expand_dims(action, axis=(1, 2, 3))
            # print("action", action.shape, "action_mask", action_mask.shape)
            fx = BasicBlock(self.planes, self.stride, name="block")(x)
            # print("residual", residual.shape, "fx", fx.shape)
            fx = nn.relu(residual + fx)
            x = fx * action_mask + residual * (1.0 - action_mask)
            return x

        def resnet_false_branch(self, x, residual, action):
            return residual

        if self.is_mutable_collection('params'):
            # print("handle params")
            return resnet_true_branch(self, x, residual, action)
        else:
            # print("real run")
            sum_is_positive = jnp.sum(action) > 0
            out = nn.cond(sum_is_positive, resnet_true_branch, resnet_false_branch, self, x, residual, action)
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
        layers.append(CondBasicBlock(planes, stride))
        for _ in range(1, blocks):
            layers.append(CondBasicBlock(planes))

        return layers, downsample, planes
    
    def load_weight(self, weight, prefix):
        conv1_params, _ = conv_weight_from_torch(weight, prefix + ".conv1")
        bn1_params, bn1_state = bn_weight_from_torch(weight, prefix + ".bn1")
        fc_params, _ = dense_weight_from_torch(weight, prefix + ".fc")
        block_param, block_state = {}, {}
        for i in range(len(self.layers)):
            for j in range(self.layers[i]):
                param, state = CondBasicBlock.load_weight(weight, prefix + f".blocks.{i}.{j}")
                block_param[f"blocks_{i}_{j}"] = param
                block_state[f"blocks_{i}_{j}"] = state
        return {
            'conv1': conv1_params,
            'bn1': bn1_params,
            'fc': fc_params,
            **block_param,
        }, {
            'bn1': bn1_state,
            **block_state
        }

    def __call__(self, inputs, probs):
        inputs = inputs.transpose(0, 2, 3, 1)
        cond = jnp.less(probs, jnp.full_like(probs, 0.5))
        policy = jnp.where(cond, jnp.full_like(probs, 0.0), jnp.full_like(probs, 1.0))
        policy = policy.transpose(1, 0)

        x = nn.relu(self.bn1(self.conv1(inputs)))

        layer = 0
        for i in range(len(self.blocks)):
            if self.ds[i] is not None:
                residual = self.ds[i](x)
            else:
                residual = x
            for j in range(len(self.blocks[i])):
                x = self.blocks[i][j](policy[layer], x, residual)
                # print("i, j, x.shape", i, j, x.shape)
                layer += 1
                residual = x

        out = self.avgpool(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class FlatResNet32Unroll(nn.Module):
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
    
    def load_weight(self, weight, prefix):
        conv1_params, _ = conv_weight_from_torch(weight, prefix + ".conv1")
        bn1_params, bn1_state = bn_weight_from_torch(weight, prefix + ".bn1")
        fc_params, _ = dense_weight_from_torch(weight, prefix + ".fc")
        block_param, block_state = {}, {}
        for i in range(len(self.layers)):
            for j in range(self.layers[i]):
                param, state = CondBasicBlock.load_weight(weight, prefix + f".blocks.{i}.{j}")
                block_param[f"blocks_{i}_{j}"] = param
                block_state[f"blocks_{i}_{j}"] = state
        return {
            'conv1': conv1_params,
            'bn1': bn1_params,
            'fc': fc_params,
            **block_param,
        }, {
            'bn1': bn1_state,
            **block_state
        }

    def __call__(self, inputs, probs, actions):
        inputs = inputs.transpose(0, 2, 3, 1)
        cond = jnp.less(probs, jnp.full_like(probs, 0.5))
        policy = jnp.where(cond, jnp.full_like(probs, 0.0), jnp.full_like(probs, 1.0))
        policy = policy.transpose(1, 0)

        x = nn.relu(self.bn1(self.conv1(inputs)))

        layer = 0
        for i in range(len(self.blocks)):
            if self.ds[i] is not None:
                residual = self.ds[i](x)
            else:
                residual = x
            for j in range(len(self.blocks[i])):
                if actions[layer] == 0:
                    x = residual
                    layer += 1
                    continue
                action_mask = policy[layer]
                fx = self.blocks[i][j](x)
                fx = nn.relu(residual + fx)
                x = fx * action_mask + residual * (1.0 - action_mask)
                layer += 1
                residual = x

        out = self.avgpool(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = jnp.array(np.fromfile(s + ".bin", dtype=dtype)).reshape(shape)
    return tensor

prefix = "../../artifacts/data/blockdrop/"

if arguments.real_weight:
    import torch
    torch_params = torch.load(os.path.join(prefix, 'ckpt_E_730_A_0.913_R_2.73E-01_S_6.92_#_53.t7'))["resnet"]

inputs_all = read_bin(os.path.join(prefix, "inputs"))
probs_all = read_bin(os.path.join(prefix, "probs"))
outputs_all = read_bin(os.path.join(prefix, "outputs"))
len_dataset = 10000

n_warmup = 100
n_run = 100

def run(batch_size):
    model = FlatResNet32()
    example_x = inputs_all[:batch_size]
    example_probs = probs_all[:batch_size]
    if arguments.real_weight:
        params, state = model.load_weight(torch_params, "")
        params = {
            'params': params,
            'batch_stats': state,
        }
    else:
        params = model.init(jax.random.PRNGKey(0), example_x, example_probs)
    jit_model = jax.jit(model.apply)
    for i in range(0, len_dataset, batch_size):
        if i >= n_warmup * batch_size: break
        inputs = inputs_all[i: i + batch_size]
        probs = probs_all[i: i + batch_size]
        out = jit_model(params, inputs, probs)
        out.block_until_ready()
        if arguments.real_weight:
            np.testing.assert_almost_equal(np.array(out), outputs_all[i: i + batch_size], decimal = 4)
    timer = Timer("ms")
    profile_start(platform)
    for i in range(0, len_dataset, batch_size):
        if i >= n_run * batch_size: break
        # test time
        inputs = inputs_all[i: i + batch_size]
        probs = probs_all[i: i + batch_size]
        timer.start()
        out = jit_model(params, inputs, probs)
        out.block_until_ready()
        timer.log()
    profile_stop(platform)
    # generate timeline:
    # inputs = jnp.array(inputs_all[0: batch_size]).block_until_ready()
    # probs = jnp.array(probs_all[0: batch_size]).block_until_ready()
    # with jax.profiler.trace("./trace", create_perfetto_link=True):
    #     out = jit_model(params, inputs, probs)
    #     out.block_until_ready()
    timer.report()

def test_fix_policy(batch_size, unroll, actions):
    actions = tuple(actions)
    policy = jnp.array(actions, dtype=jnp.float32).reshape(-1, 15).block_until_ready()
    if not unroll:
        model = FlatResNet32()
    else:
        model = FlatResNet32Unroll()
    key = random.PRNGKey(233)
    inputs = random.normal(key, (batch_size, 3, 32, 32)).block_until_ready()
    if unroll:
        params = model.init(jax.random.PRNGKey(0), inputs, policy, actions)
        jit_model = jax.jit(model.apply, static_argnums=(3,))
    else:
        params = model.init(jax.random.PRNGKey(0), inputs, policy)
        jit_model = jax.jit(model.apply)
    for i in range(0, len_dataset, batch_size):
        if i >= n_warmup * batch_size: break
        if unroll:
            out = jit_model(params, inputs, policy, actions)
        else:
            out = jit_model(params, inputs, policy)
        out.block_until_ready()
    timer = Timer("ms")
    profile_start(platform)
    for i in range(0, len_dataset, batch_size):
        if i >= n_run * batch_size: break
        # test time
        timer.start()
        if unroll:
            out = jit_model(params, inputs, policy, actions)
        else:
            out = jit_model(params, inputs, policy)
        out.block_until_ready()
        timer.log()
    profile_stop(platform)
    timer.report()


if __name__ == "__main__":
    if not arguments.overhead_test:
        run(arguments.bs)
    else:
        if arguments.rate == -1:
            actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        elif arguments.rate == 0:
            actions = [0] * 15
        elif arguments.rate == 25:
            actions = [
                0, 0, 1, 0, 0,
                0, 1, 0, 1, 0,
                0, 0, 1, 0, 0,
            ]
        elif arguments.rate == 50:
            actions = [
                0, 1, 0, 1, 0,
                1, 0, 1, 0, 1,
                0, 1, 0, 1, 0,
            ]
        elif arguments.rate == 75:
            actions = [
                1, 1, 0, 1, 1,
                1, 0, 1, 0, 1,
                1, 1, 0, 1, 1,
            ]
        elif arguments.rate == 100:
            actions = [1] * 15
        else: assert False
        test_fix_policy(1, arguments.unroll, actions)
