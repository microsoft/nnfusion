# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import graph_util

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

flags.DEFINE_integer("num_step", 100, "sequence length")
flags.DEFINE_integer("num_layer", 10, "num layer")
flags.DEFINE_integer("hidden_size", 256, "hidden size")
flags.DEFINE_integer("batch_size", 1, "mini batch size")
flags.DEFINE_boolean('profile', False, 'profile kernel runtime')
flags.DEFINE_string('backend', 'tf', 'tf or wolong or ngraph')
flags.DEFINE_integer("num_iter", 10, "mini batch size")
flags.DEFINE_integer("warmup", 5, "mini batch size")
flags.DEFINE_boolean('xla', False, 'enable xla')
flags.DEFINE_string('frozen_file', '', 'output path for the frozen pb file')
flags.DEFINE_integer("parallel", 0, "tf.ConfigProto.inter_op_parallelism_threads")

FLAGS = flags.FLAGS


class LSTMCell(object):
    W = []
    U = []
    b = []

    def __init__(self, hidden_size, scope):
        with tf.variable_scope(scope):
            self.W = []
            self.U = []
            self.b = []
            self.num_unit = hidden_size
            for i in range(4):
                W = tf.get_variable(
                    "W%d" % (i), [self.num_unit, self.num_unit], dtype=tf.float32)
                U = tf.get_variable(
                    "U%d" % (i), [self.num_unit, self.num_unit], dtype=tf.float32)
                b = tf.get_variable("bias%d" % (i), [self.num_unit], dtype=tf.float32,
                                    initializer=init_ops.constant_initializer(0, dtype=tf.float32))
                self.W.append(W)
                self.U.append(U)
                self.b.append(b)

    def call(self, inputs, state):
        c, h = state
        res = []
        for i in range(4):
            res.append(math_ops.matmul(
                inputs, self.W[i]) + math_ops.matmul(h, self.U[i]) + self.b[i])
        i, j, f, o = (res[0], res[1], res[2], res[3])
        new_c = (c * math_ops.sigmoid(f + 1.0) +
                 math_ops.sigmoid(i) * math_ops.tanh(j))
        new_h = math_ops.tanh(new_c) * math_ops.sigmoid(o)
        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        return new_h, new_state


class LSTMModel(object):
    stacked_cells = []

    def __init__(self, num_layer, hidden_size):
        self.stacked_cells = []
        self.num_layer = num_layer
        self.num_unit = hidden_size
        for layer in range(self.num_layer):
            self.stacked_cells.append(
                LSTMCell(self.num_unit, "LSTMLayer%d" % (layer)))

    def run(self, inputs, batch_size, num_step):
        self.batch_size = batch_size
        self.num_step = num_step

        cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.num_unit, forget_bias=1.0, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.state = [self._initial_state for layer in range(self.num_layer)]

        for step in range(self.num_step):
            cur_input = inputs[step, :, :]
            for layer in range(self.num_layer):
                cell_output, self.state[layer] = self.stacked_cells[layer].call(
                    cur_input, self.state[layer])
                cur_input = cell_output

        self.output = cell_output
        return self.output, self.state[-1]



