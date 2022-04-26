from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.framework import graph_util

# from nets import inception
import resnet

flags = tf.flags
flags.DEFINE_integer("batch_size", 64, "mini batch size")
flags.DEFINE_boolean('profile', False, 'profile kernel runtime')
flags.DEFINE_integer("num_iter", 10, "num of iterations")
flags.DEFINE_integer("warmup", 5, "warmup")
flags.DEFINE_integer(
    "model_layer", 50, "num of model layers, 'depth should be one of 18, 34, 50, 101, 152, 200'")
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

def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.

    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.

    Args:
      resnet_size: The number of convolutional layers needed in the model.

    Returns:
      A list of block sizes to use in building the model.

    Raises:
      KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(
                   resnet_size, choices.keys()))
        raise ValueError(err)


def main(_):
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=0
    )
    if FLAGS.xla == True:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        profile_stop()
        batch_size = FLAGS.batch_size
        height, width = 224, 224
        num_classes = 1000
        resnet_size = FLAGS.model_layer
        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True

        model = resnet.Model(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            resnet_version=1,
            data_format='channels_first',
            dtype=tf.float32
        )

        eval_inputs = tf.placeholder(
            tf.float32, [batch_size, 3, height, width], 'eval_input')

        logits = model(eval_inputs, False)

        image_inputs = np.ones((batch_size, 3, height, width))

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

            profile_start()
            for i in range(FLAGS.num_iter):
                start_time = time.time()
                output = session.run(logits, {eval_inputs: image_inputs})
                iter_time = (time.time() - start_time) * 1000
                iter_times.append(iter_time)
                print("Iteration time %f ms" % (iter_time))
            profile_stop()

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
