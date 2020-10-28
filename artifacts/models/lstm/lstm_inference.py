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

import ctypes
_cudart = ctypes.CDLL('libcudart.so')
def profile_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)
def profile_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)

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

def main(_):
    profile_stop()
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=FLAGS.parallel
    )
    if FLAGS.xla:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        batch_size = FLAGS.batch_size

        model = LSTMModel(FLAGS.num_layer, FLAGS.hidden_size)

        eval_inputs = tf.placeholder(
            tf.float32, [FLAGS.num_step, FLAGS.batch_size, FLAGS.hidden_size], 'eval_input')

        lstm_output, lstm_state = model.run(
            eval_inputs, FLAGS.batch_size, FLAGS.num_step)

        lstm_inputs = np.ones(
            (FLAGS.num_step, FLAGS.batch_size, FLAGS.hidden_size))

        session.run(tf.global_variables_initializer())

        if FLAGS.frozen_file != '':
            constant_graph = graph_util.convert_variables_to_constants(
                session, session.graph_def, [lstm_output.name.split(':')[0]])
            with tf.gfile.GFile(FLAGS.frozen_file, "wb") as f:
                f.write(constant_graph.SerializeToString())

        if not FLAGS.profile:
            # warm up
            for i in range(FLAGS.warmup):
                res = session.run(lstm_output, {
                                  eval_inputs: lstm_inputs})
                out_flat = res.flat
                if (len(out_flat) > 0):
                    max_len = min(10, len(out_flat))
                    print(lstm_output.name)
                    print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")

            iter_times = []

            profile_start()
            for i in range(FLAGS.num_iter):
                start_time = time.time()
                res = session.run(lstm_output, {
                                  eval_inputs: lstm_inputs})
                iter_time = (time.time() - start_time) * 1000
                iter_times.append(iter_time)
                print("Iteration time %f ms" % (iter_time))
            profile_stop()

            print("Summary: [min, max, mean] = [%f, %f, %f] ms" % (
                min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

        else:
            profile_start()
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            for i in range(5):
                start_time = time.time()
                res = session.run(lstm_output, {
                                  eval_inputs: lstm_inputs},
                                  options=options,
                                  run_metadata=run_metadata)
                end_time = (time.time() - start_time) * 1000
                print("iteration time %f ms" % (end_time))
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timelines/timeline_step_%d.json' % i, 'w') as f:
                    f.write(chrome_trace)
            profile_stop()


if __name__ == "__main__":
    tf.app.run()
