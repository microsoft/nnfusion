# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
According to the ResNeXt paper (https://arxiv.org/pdf/1611.05431.pdf, Table 7), ResNeXt-29-16x64d achieves the best performance among different model parameters on the CIFAR dataset.
So, we follow the ResNeXt-29-16x64d model parameters based on the widely used open-source implementation (https://github.com/taki0112/ResNeXt-Tensorflow/blob/master/ResNeXt.py).
"""

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
# from cifar10 import *
import numpy as np

weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1
cardinality = 16  # how many split ?
blocks = 3  # res_block ! (split + transition)

"""
So, the total number of layers is (3*blocks)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""

depth = 64  # out channel

batch_size = 128
iteration = 391
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 300


def conv_layer(input, filter, kernel, stride, padding='SAME', data_format='NCHW', layer_name="conv"):
    with tf.name_scope(layer_name):
        if data_format == 'NCHW':
            network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter,
                                       kernel_size=kernel, strides=stride, padding=padding, data_format='channels_first')
        else:
            network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter,
                                       kernel_size=kernel, strides=stride, padding=padding, data_format='channels_last')
        return network


def Global_Average_Pooling(x, data_format='NCHW'):
    if data_format == 'NCHW':
        return tf.reduce_mean(x, axis=[2, 3])
    else:
        return tf.reduce_mean(x, axis=[1, 2])


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='SAME', data_format='NCHW'):
    if data_format == 'NCHW':
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding, data_format='channels_first')
    else:
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding, data_format='channels_last')


def Batch_Normalization(x, training, scope, data_format='NCHW'):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return batch_norm(inputs=x, is_training=training, reuse=tf.AUTO_REUSE, data_format=data_format)


def Relu(x):
    return tf.nn.relu(x)


def Concatenation(layers, data_format='NCHW'):
    if data_format == 'NCHW':
        return tf.concat(layers, axis=1)
    else:
        return tf.concat(layers, axis=3)


def Linear(x, class_num):
    return tf.layers.dense(inputs=x, use_bias=False, units=class_num, name='linear')


class ResNeXt():
    def __init__(self, x, class_num, num_layer, data_format='NCHW', training=False):
        self.class_num = class_num
        self.training = training
        self.num_layer = num_layer
        self.data_format = data_format
        self.model = self.Build_ResNext(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=64, kernel=[
                           3, 3], stride=1, data_format=self.data_format, layer_name=scope+'_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope+'_batch1', data_format=self.data_format)
            x = Relu(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=depth, kernel=[
                           1, 1], stride=stride, data_format=self.data_format, layer_name=scope+'_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope+'_batch1', data_format=self.data_format)
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[
                           3, 3], stride=1, data_format=self.data_format, layer_name=scope+'_conv2')
            x = Batch_Normalization(
                x, training=self.training, scope=scope+'_batch2', data_format=self.data_format)
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[
                           1, 1], stride=1, data_format=self.data_format, layer_name=scope+'_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope+'_batch1', data_format=self.data_format)
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = self.transform_layer(
                    input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split, data_format=self.data_format)

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge

        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            if self.data_format == 'NCHW':
                input_dim = int(np.shape(input_x)[1])
            else:
                input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride=stride,
                                 layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(
                x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))

            if flag is True:
                pad_input_x = Average_pooling(
                    input_x, data_format=self.data_format)
                # [?, height, width, channel]
                if self.data_format == 'NCHW':
                    pad_input_x = tf.pad(
                        pad_input_x, [[0, 0], [channel, channel], [0, 0], [0, 0]])
                else:
                    pad_input_x = tf.pad(
                        pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            else:
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x

    def Build_ResNext(self, input_x):
        # only cifar10 architecture

        x = self.first_layer(input_x, scope='first_layer')

        assert((self.num_layer - 2) % 9 == 0)
        self.num_residual = int((self.num_layer - 2) / 9)

        for i in range(self.num_residual):
            x = self.residual_layer(
                x, out_dim=64, layer_num='1_' + str(i))
        for i in range(self.num_residual):
            x = self.residual_layer(x, out_dim=128, layer_num='2_'+str(i))
        for i in range(self.num_residual):
            x = self.residual_layer(x, out_dim=256, layer_num='3_'+str(i))

        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x, self.class_num)

        return x
