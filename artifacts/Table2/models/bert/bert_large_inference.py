from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import random
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.framework import graph_util

# from nets import inception
import bert_model

slim = tf.contrib.slim

flags = tf.flags
flags.DEFINE_integer("batch_size", 64, "mini batch size")
flags.DEFINE_integer("seq_length", 512, "mini batch size")
flags.DEFINE_boolean('profile', False, 'profile kernel runtime')
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


def bert_large_inference_model(input_ids, input_mask, token_type_ids):
    batch_size = FLAGS.batch_size
    seq_length = FLAGS.seq_length
    is_training = False
    use_input_mask = True
    use_token_type_ids = True
    vocab_size = 30522
    hidden_size = 1024
    num_hidden_layers = 24
    num_attention_heads = 16
    intermediate_size = 4096
    hidden_act = "gelu"
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    max_position_embeddings = 512
    type_vocab_size = 2
    initializer_range = 0.02
    scope = None

    # def ids_tensor(shape, vocab_size, rng=None, name=None):
    #     """Creates a random int32 tensor of the shape within the vocab size."""
    #     # return tf.placeholder(dtype=tf.int32, shape=shape, name=name)
    #     if rng is None:
    #         rng = random.Random()

    #     total_dims = 1
    #     for dim in shape:
    #         total_dims *= dim

    #     values = []
    #     for _ in range(total_dims):
    #         values.append(rng.randint(0, vocab_size - 1))

    #     return tf.constant(value=values, dtype=tf.int32, shape=shape, name=name)

    # input_ids = ids_tensor([batch_size, seq_length],
    #                        vocab_size)

    # input_mask = None
    # if use_input_mask:
    #     input_mask = ids_tensor(
    #         [batch_size, seq_length], vocab_size=2)

    # token_type_ids = None
    # if use_token_type_ids:
    #     token_type_ids = ids_tensor(
    #         [batch_size, seq_length], type_vocab_size)

    config = bert_model.BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        initializer_range=initializer_range)

    model = bert_model.BertModel(
        config=config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=token_type_ids,
        scope=scope)

    return model.get_sequence_output()


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

        input_ids = tf.placeholder(
            tf.int32, [FLAGS.batch_size, FLAGS.seq_length], 'input_ids')
        input_mask = tf.placeholder(
            tf.int32, [FLAGS.batch_size, FLAGS.seq_length], 'input_mask')
        token_type_ids = tf.placeholder(
            tf.int32, [FLAGS.batch_size, FLAGS.seq_length], 'token_type_ids')

        # eval_inputs = tf.random_uniform((batch_size, height, width, 3))

        logits = bert_large_inference_model(input_ids, input_mask, token_type_ids)

        input_ids_np=np.ones((FLAGS.batch_size, FLAGS.seq_length), np.int32)
        input_mask_np=np.ones((FLAGS.batch_size, FLAGS.seq_length), np.int32)
        token_type_ids_np=np.ones((FLAGS.batch_size, FLAGS.seq_length), np.int32)

        session.run(tf.global_variables_initializer())

        if FLAGS.frozen_file != '':
            constant_graph = graph_util.convert_variables_to_constants(
                session, session.graph_def, [logits.name.split(':')[0]])
            with tf.gfile.GFile(FLAGS.frozen_file, "wb") as f:
                f.write(constant_graph.SerializeToString())

        if not FLAGS.profile:
            # warm up
            for i in range(5):
                res = session.run(logits, {
                                  input_ids: input_ids_np, input_mask: input_mask_np, token_type_ids: token_type_ids_np})
                out_flat = res.flat
                if (len(out_flat) > 0):
                    max_len = min(10, len(out_flat))
                    print(logits.name)
                    print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")

            iter_times = []

            profile_start()
            for i in range(FLAGS.num_iter):
                start_time = time.time()
                output = session.run(logits, {
                                  input_ids: input_ids_np, input_mask: input_mask_np, token_type_ids: token_type_ids_np})
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
                output = session.run(logits, {
                                  input_ids: input_ids_np, input_mask: input_mask_np, token_type_ids: token_type_ids_np},
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
