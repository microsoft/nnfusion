# import onnx
# from onnx_tf.backend import prepare
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os
import time

num_iter = 100
warmup = 100
prefix = "../../data/seq2seq"
MAX_LENGTH = 50
OUTPUT_SIZE = 3797
HIDDEN_SIZE = 256

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.add_argument('--platform', type=str)
parser.set_defaults(unroll=False)
parser.add_argument('--rate', type=int, default=-1)
parser.add_argument('--bs', type=int, default=1)
arguments = parser.parse_args()
platform = arguments.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

def load_model(batch_size, backend, unroll):
    if platform == 'V100':
        import onnx
        from onnx_tf.backend import prepare
        if unroll:
            model_path = f"onnx/seq2seq.b{batch_size}.unroll.onnx"
        else:
            model_path = f"onnx/seq2seq.b{batch_size}.onnx"
        model = onnx.load(model_path)
        # op = onnx.OperatorSetIdProto()
        # Sigmoid version 13 is not implemented.
        # op.version = 12
        # update_model = onnx.helper.make_model(model.graph, opset_imports=[op])
        tf_model = prepare(model)
        # tf_model.save(f"blockdrop.b{batch_size}.tf", save_format='tf')
        return tf_model.graph.as_graph_def()
    elif backend == 'MI100':
        with tf.gfile.GFile(f'seq2seq.b{batch_size}.tfgraph', "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
    else:
        raise NotImplementedError

# load_model(1)
# load_model(64)

def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = np.fromfile(s + ".bin", dtype=dtype).reshape(shape)
    return tensor

def gen_mask_from_sequence(std):
    bs = std.shape[0]
    padded_std = np.zeros((bs, MAX_LENGTH), dtype=std.dtype)
    padded_std[:, :std.shape[1]] = std
    mask = np.zeros((bs, MAX_LENGTH, OUTPUT_SIZE))
    mask[np.expand_dims(np.arange(bs), 1), np.expand_dims(np.arange(MAX_LENGTH), 0), padded_std] = 1000000.0
    mask = np.transpose(mask, axes=(1, 0, 2))
    return mask

tokens = read_bin('../../data/tatoeba-eng-fra/tokens', dtype=np.int64)
masks = gen_mask_from_sequence(tokens)

def export_model(batch_size, unroll):
    model = load_model(batch_size, platform, unroll)
    output_names = tuple([model.tensor_dict[onnx_name].name for onnx_name in model.outputs])
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
        with tf.gfile.GFile(f"seq2seq.b{batch_size}.tfgraph", "wb") as f:
            f.write(constant_graph.SerializeToString())

# export_model(1, arguments.unroll)
# export_model(64, arguments.unroll)

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

    # exit(0)
    model = load_model(batch_size, platform, False)
    encoder_output = np.random.randn(MAX_LENGTH, batch_size, HIDDEN_SIZE)
    # print("inputs:")
    # for onnx_name in model.inputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # print("outputs:")
    # for onnx_name in model.outputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])

    # inputs:
    # std.1 Tensor("std.1:0", shape=(50, 1, 3797), dtype=float32)
    # h.1 Tensor("h.1:0", shape=(1, 256), dtype=float32)
    # c.1 Tensor("c.1:0", shape=(1, 256), dtype=float32)
    # outputs:
    # 29 Tensor("while/Exit_6:0", shape=(50, 1), dtype=int64)
    # h.4 Tensor("while/Exit_2:0", shape=(1, 256), dtype=float32)
    h = np.random.randn(batch_size, HIDDEN_SIZE)
    c = np.random.randn(batch_size, HIDDEN_SIZE)
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model, name="")
        # warm up
        for i in range(0, 6400, batch_size):
            if i >= warmup * batch_size: break
            mask = masks[:, i: i + batch_size].copy()
            outputs = session.run(['while/Exit_6:0', 'while/Exit_2:0'] , {
                'std.1:0': mask,
                'h.1:0': h,
                'c.1:0': c,
            })
        # run
        profile_start(platform)
        iter_times = []
        for i in range(0, 6400, batch_size):
            if i >= num_iter * batch_size: break
            mask = masks[:, i: i + batch_size].copy()
            start_time = time.time()
            outputs = session.run(['while/Exit_6:0', 'while/Exit_2:0'] , {
                'std.1:0': mask,
                'h.1:0': h,
                'c.1:0': c,
            })
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))


def test_fix_policy(batch_size, unroll):
    print("----batch_size={}---unroll={}----".format(batch_size, unroll))
    import onnx
    from onnx_tf.backend import prepare
    if unroll:
        model = onnx.load(f'onnx/seq2seq.b1.unroll.onnx')
    else:
        model = onnx.load(f'onnx/seq2seq.b1.fix.onnx')
    
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=0
    )

    std = []
    MAX_LENGTH = 50
    for i in range(batch_size):
        l = 10
        lst = list(range(1, l))
        lst.append(0)
        assert(len(lst) <= MAX_LENGTH)
        # pad to MAX_LENGTH
        lst = lst + [0] * (MAX_LENGTH - len(lst))
        std.append(lst)
    std = np.array(std)
    mask = gen_mask_from_sequence(std)
    encoder_output = np.random.randn(MAX_LENGTH, batch_size, HIDDEN_SIZE)

    session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1
    model = prepare(model)
    # print("inputs:")
    # for onnx_name in model.inputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # print("outputs:")
    # for onnx_name in model.outputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # fix:
    # inputs:
    # std.1 Tensor("std.1:0", shape=(50, 1, 3797), dtype=float32)
    # h.1 Tensor("h.1:0", shape=(1, 256), dtype=float32)
    # c.1 Tensor("c.1:0", shape=(1, 256), dtype=float32)
    # outputs:
    # 29 Tensor("while/Exit_6:0", shape=(50, 1), dtype=int64)
    # h.4 Tensor("while/Exit_2:0", shape=(1, 256), dtype=float32)
    # unroll:
    # inputs:
    # std.1 Tensor("std.1:0", shape=(50, 1, 3797), dtype=float32)
    # h.1 Tensor("h.1:0", shape=(1, 256), dtype=float32)
    # c.1 Tensor("c.1:0", shape=(1, 256), dtype=float32)
    # output.1 Tensor("output.1:0", shape=(1,), dtype=int64)
    # outputs:
    # 533 Tensor("TensorScatterUpdate_9:0", shape=(50, 1), dtype=int64)
    # h.36 Tensor("onnx_tf_prefix_Mul_462:0", shape=(1, 256), dtype=float32)
    # cond.3 Tensor("onnx_tf_prefix_And_481:0", shape=(), dtype=bool)

    h = np.random.randn(batch_size, HIDDEN_SIZE)
    c = np.random.randn(batch_size, HIDDEN_SIZE)
    sos = np.full((batch_size,), 1, dtype=np.int64)
    output_names = tuple([model.tensor_dict[onnx_name].name for onnx_name in model.outputs])
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model.graph.as_graph_def(), name="")
        # warm up
        for i in range(warmup):
            if unroll:
                outputs = session.run(output_names, {
                    'std.1:0': mask,
                    'h.1:0': h,
                    'c.1:0': c,
                    'output.1:0': sos,
                })
            else:
                outputs = session.run(output_names, {
                    'std.1:0': mask,
                    'h.1:0': h,
                    'c.1:0': c,
                })
        # run
        profile_start(platform)
        iter_times = []
        for i in range(num_iter):
            start_time = time.time()
            if unroll:
                outputs = session.run(output_names, {
                    'std.1:0': mask,
                    'h.1:0': h,
                    'c.1:0': c,
                    'output.1:0': sos,
                })
            else:
                outputs = session.run(output_names, {
                    'std.1:0': mask,
                    'h.1:0': h,
                    'c.1:0': c,
                })
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))




if not arguments.overhead_test:
    test_model(arguments.bs, False)
else:
    test_fix_policy(1, arguments.unroll)