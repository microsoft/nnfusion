import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import os
import time

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.WARN)
flags.DEFINE_boolean('profile', False, 'profile kernel runtime')
flags.DEFINE_integer("num_iter", 100, "mini batch size")
flags.DEFINE_integer("warmup", 100, "mini batch size")
flags.DEFINE_integer("parallel", 0, "tf.ConfigProto.inter_op_parallelism_threads")
flags.DEFINE_integer("bs", 1, "mini batch size")
flags.DEFINE_string('platform', 'V100', 'V100 or MI100')
FLAGS = flags.FLAGS
platform = FLAGS.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

def load_model(batch_size):
    model_path = f"resnet101.b{batch_size}.onnx"
    model = onnx.load(model_path)
    # op = onnx.OperatorSetIdProto()
    # Sigmoid version 13 is not implemented.
    # op.version = 12
    # update_model = onnx.helper.make_model(model.graph, opset_imports=[op])
    tf_model = prepare(model)
    # tf_model.save(f"blockdrop.b{batch_size}.tf", save_format='tf')
    return tf_model


# load_model(1)
# load_model(64)
# exit(0)

# def read_bin(s):
#     with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
#     tensor = np.fromfile(s + ".bin", dtype=np.float32).reshape(shape)
#     return tensor


# prefix = "../../data/blockdrop/"
# inputs_all = read_bin(os.path.join(prefix, "inputs"))
# probs_all = read_bin(os.path.join(prefix, "probs"))
# outputs_all = read_bin(os.path.join(prefix, "outputs"))
# len_dataset = 10000


def test_model(batch_size, enable_xla): 
    print("----batch_size={}---xla={}----".format(batch_size, enable_xla))

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

    # exit(0)
    model = load_model(batch_size)
    # print("inputs:")
    # for onnx_name in model.inputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # print("outputs:")
    # for onnx_name in model.outputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # inputs:
    # x.1 Tensor("x.1:0", shape=(1, 3, 224, 224), dtype=float32)
    # outputs:
    # x.20 Tensor("add_105:0", shape=(1, 1000), dtype=float32)
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model.graph.as_graph_def(), name="")
        # warm up
        inputs = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
        for i in range(FLAGS.warmup):
            outputs = session.run('add_105:0', {
                'x.1:0': inputs,
            })
        # run
        profile_start(platform)
        iter_times = []
        inputs = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
        for i in range(FLAGS.num_iter):
            start_time = time.time()
            outputs = session.run('add_105:0', {
                'x.1:0': inputs,
            })
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

test_model(FLAGS.bs, False)
# test_model(1, True)
# test_model(64, False)
# test_model(64, True)
