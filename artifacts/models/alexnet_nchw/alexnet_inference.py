# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.framework import graph_util

from alexnet_model import AlexnetModel

slim = tf.contrib.slim

flags = tf.flags
flags.DEFINE_integer("batch_size", 1, "mini batch size")
flags.DEFINE_boolean('profile', False, 'profile kernel runtime')
flags.DEFINE_integer("num_iter", 10, "mini batch size")
flags.DEFINE_integer("warmup", 5, "mini batch size")
flags.DEFINE_boolean('xla', False, 'enable xla')
flags.DEFINE_string('frozen_file', '', 'output path for the frozen pb file')
flags.DEFINE_integer("parallel", 0, "tf.ConfigProto.inter_op_parallelism_threads")

FLAGS = flags.FLAGS


def main(_):
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
        height, width = 227, 227
        num_classes = 1000

        eval_inputs = tf.placeholder(
            tf.float32, [batch_size, height, width, 3], 'eval_input')

        model = AlexnetModel()
        logits, _ = model.build_network([eval_inputs],
                                               phase_train=False, nclass=num_classes)

        image_inputs = np.ones((batch_size, height, width, 3))

        session.run(tf.global_variables_initializer())

        if FLAGS.frozen_file != '':
            constant_graph = graph_util.convert_variables_to_constants(session, session.graph_def, [logits.name.split(':')[0]])
            with tf.gfile.GFile(FLAGS.frozen_file, "wb") as f:
                f.write(constant_graph.SerializeToString())

        if not FLAGS.profile:
            # warm up
            for i in range(FLAGS.warmup):
                output = session.run(logits, {eval_inputs: image_inputs})
                out_flat = output.flat
                if (len(out_flat) > 0):
                    max_len = min(10, len(out_flat))
                    print(logits.name)
                    print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")

            iter_times = []

            for i in range(FLAGS.num_iter):
                start_time = time.time()
                output = session.run(logits, {eval_inputs: image_inputs})
                iter_time = (time.time() - start_time) * 1000
                iter_times.append(iter_time)
                print("Iteration time %f ms" % (iter_time))

            print("Summary: [min, max, mean] = [%f, %f, %f] ms" % (
                min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

        else:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            for i in range(5):
                start_time = time.time()
                output = session.run(logits, {eval_inputs: image_inputs},
                                     options=options,
                                     run_metadata=run_metadata)
                end_time = (time.time() - start_time) * 1000
                print("iteration time %f ms" % (end_time))
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timelines/timeline_step_%d.json' % i, 'w') as f:
                    f.write(chrome_trace)


if __name__ == "__main__":
    tf.app.run()
