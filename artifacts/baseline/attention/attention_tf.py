import tensorflow as tf
import numpy as np
import os
import time

num_iter = 100
warmup = 100

START_LEN = 32
SEQ_LEN = 64
NUM_HEAD = 12
SIZE_PER_HEAD = 64

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.add_argument('--platform', type=str)
parser.add_argument('--bs', type=int, default=1)
parser.set_defaults(unroll=False)
arguments = parser.parse_args()
platform = arguments.platform
import sys
sys.path.append('../../ast_analyzer/utils')
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

import ctypes

def load_model(batch_size, platform, unroll=False):
    if platform == 'V100':
        import onnx
        from onnx_tf.backend import prepare
        if unroll:
            model_path = f"onnx/attention.b{batch_size}.unroll.onnx"
        else:
            model_path = f"onnx/attention.b{batch_size}.onnx"
        model = onnx.load(model_path)
        tf_model = prepare(model)
        return tf_model.graph.as_graph_def()
    elif platform == 'MI100':
        with tf.gfile.GFile(f'attention.b{batch_size}.tfgraph', "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
    else:
        raise NotImplementedError


def export_model(batch_size, unroll):
    from tensorflow.python.framework import graph_util
    model = load_model(batch_size, platform, unroll)
    output_names = tuple([model.tensor_dict[onnx_name].name for onnx_name in model.outputs])
    print(output_names)
    output_names = [o.split(":")[0] for o in output_names]
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=0
    )
    # not enable xla
    session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model.graph.as_graph_def(), name="")

        constant_graph = graph_util.convert_variables_to_constants(
                session, session.graph_def, output_names)
        with tf.gfile.GFile(f"attention.b{batch_size}.tfgraph", "wb") as f:
            f.write(constant_graph.SerializeToString())


def test_model(batch_size, enable_xla):
    print("----batch_size={}---xla={}----".format(batch_size, enable_xla))

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=0
    )

    if enable_xla:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    else:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1

    model = load_model(batch_size, platform, False)
    # print("inputs:")
    # for onnx_name in model.inputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # print("outputs:")
    # for onnx_name in model.outputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # inputs:
    # x.1 Tensor("x.1:0", shape=(1, 12, 1, 64), dtype=float32)
    # k.1 Tensor("k.1:0", shape=(1, 12, 64, 64), dtype=float32)
    # v.1 Tensor("v.1:0", shape=(1, 12, 64, 64), dtype=float32)
    # outputs:
    # onnx::MatMul_4235 Tensor("TensorScatterUpdate_62:0", shape=(1, 12, 64, 64), dtype=float32)
    # onnx::MatMul_4293 Tensor("TensorScatterUpdate_63:0", shape=(1, 12, 64, 64), dtype=float32)
    # x.252 Tensor("onnx_tf_prefix_MatMul_3267:0", shape=(1, 12, 1, 64), dtype=float32)
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model, name="")
        x = np.random.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD)
        k = np.zeros((batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD))
        k[:, :, :START_LEN, :] = np.random.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD)
        v = np.zeros((batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD))
        v[:, :, :START_LEN, :] = np.random.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD)

        for i in range(warmup):
            _ = session.run(
                ['TensorScatterUpdate_62:0', 'TensorScatterUpdate_63:0', 'onnx_tf_prefix_MatMul_3267:0'],
                feed_dict={
                    'x.1:0': x,
                    'k.1:0': k,
                    'v.1:0': v,
                }
            )
            
        profile_start(platform)
        iter_times = []
        for i in range(num_iter):
            start_time = time.time()
            _ = session.run(
                ['TensorScatterUpdate_62:0', 'TensorScatterUpdate_63:0', 'onnx_tf_prefix_MatMul_3267:0'],
                feed_dict={
                    'x.1:0': x,
                    'k.1:0': k,
                    'v.1:0': v,
                }
            )
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

def test_model_unroll(batch_size, enable_xla):
    print("----batch_size={}---xla={}----unroll----".format(batch_size, enable_xla))

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=0
    )

    if enable_xla:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    else:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1

    model = load_model(batch_size, platform, True)
    # print("inputs:")
    # for onnx_name in model.inputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # print("outputs:")
    # for onnx_name in model.outputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # inputs:
    # x.1 Tensor("x.1:0", shape=(1, 12, 1, 64), dtype=float32)
    # k.1 Tensor("k.1:0", shape=(1, 12, 64, 64), dtype=float32)
    # v.1 Tensor("v.1:0", shape=(1, 12, 64, 64), dtype=float32)
    # outputs:
    # onnx::MatMul_4235 Tensor("TensorScatterUpdate_62:0", shape=(1, 12, 64, 64), dtype=float32)
    # onnx::MatMul_4293 Tensor("TensorScatterUpdate_63:0", shape=(1, 12, 64, 64), dtype=float32)
    # x.252 Tensor("onnx_tf_prefix_MatMul_3267:0", shape=(1, 12, 1, 64), dtype=float32)
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model, name="")
        x = np.random.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD)
        k = np.zeros((batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD))
        k[:, :, :START_LEN, :] = np.random.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD)
        v = np.zeros((batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD))
        v[:, :, :START_LEN, :] = np.random.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD)

        for i in range(warmup):
            _ = session.run(
                ['TensorScatterUpdate_62:0', 'TensorScatterUpdate_63:0', 'onnx_tf_prefix_MatMul_3267:0'],
                feed_dict={
                    'x.1:0': x,
                    'k.1:0': k,
                    'v.1:0': v,
                }
            )
            
        profile_start(platform)
        iter_times = []
        for i in range(num_iter):
            start_time = time.time()
            _ = session.run(
                ['TensorScatterUpdate_62:0', 'TensorScatterUpdate_63:0', 'onnx_tf_prefix_MatMul_3267:0'],
                feed_dict={
                    'x.1:0': x,
                    'k.1:0': k,
                    'v.1:0': v,
                }
            )
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

if __name__ == '__main__':
    if not arguments.overhead_test:
        test_model(arguments.bs, False)
    else:
        if arguments.unroll:
            test_model_unroll(1, False)
        else:
            test_model(1, False)