import tensorflow as tf
import numpy as np
import os
import time

num_iter = 100
warmup = 100

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
from nvprof import enable_profile, profile_start, profile_stop
enable_profile(platform)

import ctypes

def load_model(batch_size, platform, unroll=False):
    if platform == 'V100':
        import onnx
        from onnx_tf.backend import prepare
        if unroll:
            model_path = f"onnx/blockdrop.b{batch_size}.unroll.onnx"
        else:
            model_path = f"onnx/blockdrop.b{batch_size}.onnx"
        model = onnx.load(model_path)
        tf_model = prepare(model)
        return tf_model.graph.as_graph_def()
    elif platform == 'MI100':
        with tf.gfile.GFile(f'blockdrop.b{batch_size}.tfgraph', "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
    else:
        raise NotImplementedError


def read_bin(s):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = np.fromfile(s + ".bin", dtype=np.float32).reshape(shape)
    return tensor


prefix = "../../artifacts/data/blockdrop/"
inputs_all = read_bin(os.path.join(prefix, "inputs"))
probs_all = read_bin(os.path.join(prefix, "probs"))
outputs_all = read_bin(os.path.join(prefix, "outputs"))
len_dataset = 10000

def export_model(batch_size, unroll):
    from tensorflow.python.framework import graph_util
    model = load_model(batch_size, unroll)
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
        with tf.gfile.GFile(f"blockdrop.b{batch_size}.tfgraph", "wb") as f:
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

    # exit(0)
    model = load_model(batch_size, platform)
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model, name="")
        # warm up
        for i in range(0, len_dataset, batch_size):
            if i >= warmup * batch_size: break
            inputs = inputs_all[i: i + batch_size]
            probs = probs_all[i: i + batch_size]
            outputs = session.run('add_18:0', {
                'inputs.1:0': inputs,
                'probs.1:0': probs
            })
        # run
        profile_start(platform)
        iter_times = []
        for i in range(0, len_dataset, batch_size):
            if i >= num_iter * batch_size: break
            inputs = inputs_all[i: i + batch_size]
            probs = probs_all[i: i + batch_size]
            start_time = time.time()
            outputs = session.run('add_18:0', {
                'inputs.1:0': inputs,
                'probs.1:0': probs
            })
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

def test_fix_policy(batch_size, unroll, probs, rate): 
    print("----batch_size={}---unroll={}----".format(batch_size, unroll))
    import onnx
    from onnx_tf.backend import prepare

    rate_tag = "skip" if rate == -1 else f"{rate}"
    if unroll:
        model = onnx.load(f'onnx/blockdrop.b1.unroll.{rate_tag}.onnx')
    else:
        model = onnx.load(f'onnx/blockdrop.b1.fix.{rate_tag}.onnx')

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=0
    )
    inputs = np.random.rand(batch_size, 3, 32, 32).astype(np.float32)

    session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1
    
    model = prepare(model)
    # print("inputs:")
    # for onnx_name in model.inputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # print("outputs:")
    # for onnx_name in model.outputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # if unroll:
    #     out_name = 'add_30:0'
    # else:
    #     out_name = 'add_18:0'
    output_names = tuple([model.tensor_dict[onnx_name].name for onnx_name in model.outputs])

    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model.graph.as_graph_def(), name="")
        # warm up
        for i in range(0, len_dataset, batch_size):
            if i >= warmup * batch_size: break
            outputs = session.run(output_names, {
                'inputs.1:0': inputs,
                'probs.1:0': probs
            })
        # run
        profile_start(platform)
        iter_times = []
        for i in range(0, len_dataset, batch_size):
            if i >= num_iter * batch_size: break
            start_time = time.time()
            outputs = session.run(output_names, {
                'inputs.1:0': inputs,
                'probs.1:0': probs
            })
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

if not arguments.overhead_test:
    test_model(arguments.bs, False)
else:
    if arguments.rate == -1:
        actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    elif arguments.rate == 0:
        actions = [0] * 15
    elif arguments.rate == 25:
        actions = [
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
        ] 
    elif arguments.rate == 50:
        actions = [
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
        ]
    elif arguments.rate == 75:
        actions = [
            1, 1, 0, 1, 1,
            1, 0, 1, 0, 1,
            1, 1, 0, 1, 1,
        ]
    elif arguments.rate == 100:
        actions = [1] * 15
    else:
        raise NotImplementedError

    actions = np.array(actions, dtype=np.float32).reshape(-1, 15)
    test_fix_policy(1, arguments.unroll, actions, arguments.rate)
