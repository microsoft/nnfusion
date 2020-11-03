"""
According to the ResNeXt paper (https://arxiv.org/pdf/1611.05431.pdf, Table 7), ResNeXt-101-64x4d achieves the best performance among different model parameters of ResNeXt on the ImageNet dataset.
So, we follow the ResNeXt-101-64x4d model parameters based on the widely used open-source implementation (https://github.com/taki0112/ResNeXt-Tensorflow/blob/master/ResNeXt.py).
"""

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np


cardinality = 64  # how many split ?


def conv_layer_padding(input, filter, kernel, stride, padding='SAME', data_format='NCHW', layer_name="conv"):
    with tf.name_scope(layer_name):
        assert (data_format == 'NCHW' and padding == 'SAME' and len(
            kernel) == 2 and kernel[0] == kernel[1])
        if stride > 1:
            # manual padding to avoid asymmetric padding in conv2d operator
            pad = kernel[0] - 1
            pad_h = pad // 2
            pad_w = pad - pad_h
            input = tf.pad(
                input, [[0, 0], [0, 0], [pad_h, pad_w], [pad_h, pad_w]])

        network = tf.layers.conv2d(
            inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
            padding=('SAME' if stride == 1 else 'VALID'), data_format='channels_first')

        return network


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


def Max_pooling(x, pool_size=[3, 3], stride=2, padding='SAME', data_format='NCHW'):
    if data_format == 'NCHW':
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding, data_format='channels_first')
    else:
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding, data_format='channels_last')


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
            x = conv_layer_padding(x, filter=64, kernel=[
                7, 7], stride=2, data_format=self.data_format, layer_name=scope+'_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope+'_batch1', data_format=self.data_format)
            x = Relu(x)
            x = Max_pooling(x, data_format=self.data_format)

            return x

    def transform_layer(self, x, depth, stride, scope):
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

    def split_layer(self, input_x, n_channel, stride, layer_name):
        i_channel = n_channel / cardinality
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = self.transform_layer(
                    input_x, depth=i_channel, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split, data_format=self.data_format)

    def residual_layer(self, input_x, n_channel, out_dim, stride, layer_num):
        # split + transform(bottleneck) + transition + merge
        assert(self.data_format == 'NCHW')

        if self.data_format == 'NCHW':
            input_dim = int(np.shape(input_x)[1])
        else:
            input_dim = int(np.shape(input_x)[-1])

        x = self.split_layer(input_x, n_channel=n_channel, stride=stride,
                             layer_name='split_layer_'+layer_num)
        x = self.transition_layer(
            x, out_dim=out_dim, scope='trans_layer_'+layer_num)

        short_cut = input_x
        if input_dim != out_dim:
            short_cut = conv_layer(input_x, filter=out_dim, kernel=[
                1, 1], stride=stride, data_format=self.data_format, layer_name='short_cut_'+layer_num)
            short_cut = Batch_Normalization(
                short_cut, training=self.training, scope='short_cut_'+layer_num+'_bn', data_format=self.data_format)

        input_x = Relu(x + short_cut)

        return input_x

    def Build_ResNext(self, input_x):
        # only imagenet architecture

        x = self.first_layer(input_x, scope='first_layer')

        if self.num_layer == 50:
            self.num_residual = [3, 4, 6, 3]
        elif self.num_layer == 101:
            self.num_residual = [3, 4, 23, 3]
        elif self.num_layer == 152:
            self.num_residual = [3, 8, 36, 3]
        else:
            print("default: ResNeXt-101")
            self.num_residual = [3, 4, 23, 3]

        for i in range(self.num_residual[0]):
            x = self.residual_layer(
                x, n_channel=256, out_dim=256, stride=1, layer_num='1_'+str(i))

        x = self.residual_layer(
            x, n_channel=512, out_dim=512, stride=2, layer_num='2_'+str(0))
        for i in range(1, self.num_residual[1]):
            x = self.residual_layer(
                x, n_channel=512, out_dim=512, stride=1, layer_num='2_'+str(i))

        x = self.residual_layer(
            x, n_channel=1024, out_dim=1024, stride=2, layer_num='3_'+str(0))
        for i in range(1, self.num_residual[2]):
            x = self.residual_layer(
                x, n_channel=1024, out_dim=1024, stride=1, layer_num='3_'+str(i))

        x = self.residual_layer(
            x, n_channel=2048, out_dim=2048, stride=2, layer_num='4_'+str(0))
        for i in range(1, self.num_residual[3]):
            x = self.residual_layer(
                x, n_channel=2048, out_dim=2048, stride=1, layer_num='4_'+str(i))

        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x, self.class_num)

        return x
