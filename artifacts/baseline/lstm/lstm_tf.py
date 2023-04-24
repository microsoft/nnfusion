from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

flags.DEFINE_integer("num_step", 64, "sequence length")
flags.DEFINE_integer("num_layer", 10, "num layer")
flags.DEFINE_integer("hidden_size", 256, "hidden size")
flags.DEFINE_boolean('profile', False, 'profile kernel runtime')
flags.DEFINE_string('backend', 'tf', 'tf or wolong or ngraph')
flags.DEFINE_integer("num_iter", 100, "mini batch size")
flags.DEFINE_integer("warmup", 100, "mini batch size")
# flags.DEFINE_boolean('xla', False, 'enable xla')
# flags.DEFINE_string('frozen_file', '', 'output path for the frozen pb file')
flags.DEFINE_integer("parallel", 0, "tf.ConfigProto.inter_op_parallelism_threads")
flags.DEFINE_integer("bs", 1, "mini batch size")
flags.DEFINE_string('platform', 'V100', 'V100 or MI100')
flags.DEFINE_bool('overhead_test', False, 'overhead test')
flags.DEFINE_bool('unroll', False, 'unroll or not')
FLAGS = flags.FLAGS
platform = FLAGS.platform
import sys
sys.path.append('../../ast_analyzer/utils')
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

class LSTMModel(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = [tf.keras.layers.LSTMCell(self.hidden_size) for _ in range(self.num_layers)]

    def run(self, inputs):
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        state_h = [tf.zeros((batch_size, self.hidden_size), dtype=tf.float32) for _ in range(self.num_layers)]
        state_c = [tf.zeros((batch_size, self.hidden_size), dtype=tf.float32) for _ in range(self.num_layers)]
        for step in range(seq_len):
            cur_input = inputs[step]
            for j in range(self.num_layers):
                cur_input, (state_h[j], state_c[j]) = self.cell[j](cur_input, (state_h[j], state_c[j]))
        return cur_input

    
    def op_body(self, i, inputs, state, seq_len):
        cur_input = inputs[i]
        h, c = state
        for j in range(self.num_layers):
            cur_input, (h[j], c[j]) = self.cell[j].apply(cur_input, (h[j], c[j]))
        return i + 1, inputs, (h, c), seq_len

    
    def run_op(self, inputs):
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        cond = lambda i, a, b, seq_len: i < seq_len
        state_h = [tf.zeros((batch_size, self.hidden_size), dtype=tf.float32) for _ in range(self.num_layers)]
        state_c = [tf.zeros((batch_size, self.hidden_size), dtype=tf.float32) for _ in range(self.num_layers)]
        _, _, (new_h, new_c), _ = tf.while_loop(cond, self.op_body, (0, inputs, (state_h, state_c), seq_len))
        return new_h[self.num_layers - 1]


class LSTMCudnn(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_layers):
        self.cell = [tf.keras.layers.CuDNNLSTM(hidden_size, time_major=True, return_sequences=True) for _ in range(num_layers)]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def call(self, inputs):
        batch_size = inputs.shape[1]
        out = inputs
        for i in range(self.num_layers):
            state_c = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
            state_h = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
            out = self.cell[i](out, initial_state = (state_h, state_c))
        return out


def test_model(batch_size, enable_xla, enable_training, loop_style): 
    print("----batch_size={}---xla={}---train={}---loop={}---".format(batch_size, enable_xla, enable_training, loop_style))

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=FLAGS.parallel
    )


    if enable_xla:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    else:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1

    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        if loop_style == 'unroll':
            eval_inputs = tf.placeholder(tf.float32, [FLAGS.num_step, batch_size, FLAGS.hidden_size], 'eval_input')
            model = LSTMModel(FLAGS.hidden_size, FLAGS.num_layer)
            trace_start = time.time()
            lstm_output = model.run(eval_inputs)
            trace_stop = time.time()
            print("trace time: {} ms".format((trace_stop - trace_start) * 1000))
        elif loop_style == 'while':
            eval_inputs = tf.placeholder(tf.float32, [FLAGS.num_step, batch_size, FLAGS.hidden_size], 'eval_input')
            model = LSTMModel(FLAGS.hidden_size, FLAGS.num_layer)
            trace_start = time.time()
            lstm_output = model.run_op(eval_inputs)
            trace_stop = time.time()
            print("trace time: {} ms".format((trace_stop - trace_start) * 1000))
        else:
            model = LSTMCudnn(FLAGS.hidden_size, FLAGS.num_layer)
            eval_inputs = tf.placeholder(tf.float32, [FLAGS.num_step, batch_size, FLAGS.hidden_size], 'eval_input')
            lstm_output = model.call(eval_inputs)

        nodes = [lstm_output]
        if enable_training:
            grad = tf.gradients(tf.math.reduce_sum(lstm_output), tf.trainable_variables())
            nodes.append(grad)

        lstm_inputs = np.ones((FLAGS.num_step, batch_size, FLAGS.hidden_size))

        session.run(tf.global_variables_initializer())

        # warm up
        for i in range(FLAGS.warmup):
            start_time = time.time()
            res = session.run(nodes, {eval_inputs: lstm_inputs})
            iter_time = (time.time() - start_time) * 1000
            print("Iteration time %f ms" % (iter_time))
        iter_times = []

        profile_start(platform)
        for i in range(FLAGS.num_iter):
            start_time = time.time()
            res = session.run(nodes, {eval_inputs: lstm_inputs})
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
            # print("Iteration time %f ms" % (iter_time))
        profile_stop(platform)
        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))


def main(_):
    if not FLAGS.overhead_test:
        test_model(FLAGS.bs, False, False, 'cudnn')
    else:
        if FLAGS.unroll:
            test_model(1, False, False, 'unroll')
        else:
            test_model(1, False, False, 'while')
    # test_model(1, True, False, 'cudnn')

    # test_model(64, False, False, 'cudnn')
    # test_model(64, True, False, 'cudnn')

    # test_model(1, False, False, 'while')
    # test_model(1, False, False, 'unroll')


if __name__ == "__main__":
    tf.app.run()
