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
import seq2seq_model
from tensorflow.python.framework import graph_util

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

flags.DEFINE_integer("encoder_step", 100, "sequence length")
flags.DEFINE_integer("encoder_layer", 8, "num layer")
flags.DEFINE_integer("decoder_step", 30, "sequence length")
flags.DEFINE_integer("decoder_layer", 4, "num layer")
flags.DEFINE_integer("hidden_size", 128, "hidden size")
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
        profile_stop()
        batch_size = FLAGS.batch_size

        model = seq2seq_model.Seq2SeqModel(
            batch_size, FLAGS.hidden_size, FLAGS.encoder_layer, FLAGS.encoder_step, FLAGS.decoder_layer, FLAGS.decoder_step)

        eval_inputs = tf.placeholder(
            tf.float32, [FLAGS.encoder_step, FLAGS.batch_size, FLAGS.hidden_size], 'eval_input')

        eval_inputs_list = tf.split(value=eval_inputs, axis=0, num_or_size_splits=FLAGS.encoder_step)
        for i in range(len(eval_inputs_list)):
            eval_inputs_list[i] = tf.squeeze(eval_inputs_list[i],axis=[0])

        logits = model(eval_inputs_list)

        lstm_inputs = np.ones(
            (FLAGS.encoder_step, FLAGS.batch_size, FLAGS.hidden_size))

        session.run(tf.global_variables_initializer())

        if FLAGS.frozen_file != '':
            constant_graph = graph_util.convert_variables_to_constants(session, session.graph_def, [logits.name.split(':')[0]])
            with tf.gfile.GFile(FLAGS.frozen_file, "wb") as f:
                f.write(constant_graph.SerializeToString())

        if not FLAGS.profile:
            # warm up
            for i in range(FLAGS.warmup):
                res = session.run(logits, {
                                  eval_inputs: lstm_inputs})
                out_flat = res.flat
                if (len(out_flat) > 0):
                    max_len = min(10, len(out_flat))
                    print(logits.name)
                    print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")

            iter_times = []

            profile_start()
            for i in range(FLAGS.num_iter):
                start_time = time.time()
                res = session.run(logits, {
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
                res = session.run(logits, {
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
