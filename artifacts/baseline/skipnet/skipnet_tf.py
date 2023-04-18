# import onnx
# from onnx_tf.backend import prepare
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
parser.set_defaults(unroll=False)
parser.add_argument('--rate', type=int, default=-1)
parser.add_argument('--platform', type=str)
parser.add_argument('--bs', type=int, default=1)
arguments = parser.parse_args()
platform = arguments.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)


def load_model(batch_size, platform):
    if platform == 'V100':
        import onnx
        from onnx_tf.backend import prepare
        model_path = f"onnx/skipnet.b{batch_size}.onnx"
        model = onnx.load(model_path)
        tf_model = prepare(model)
        return tf_model.graph.as_graph_def()
    elif platform == 'MI100':
        with tf.gfile.GFile(f'skipnet.b{batch_size}.tfgraph', "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
    else:
        raise NotImplementedError


def read_bin(s):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = np.fromfile(s + ".bin", dtype=np.float32).reshape(shape)
    return tensor


prefix = "../../artifacts/data/skipnet/"
inputs_all = read_bin(os.path.join(prefix, "inputs"))
outputs_all = read_bin(os.path.join(prefix, "outputs"))
ch_all = read_bin(os.path.join(prefix, "ch"))
cc_all = read_bin(os.path.join(prefix, "cc"))
len_dataset = 6400

def export_model(batch_size):
    model = load_model(batch_size, platform)
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
        with tf.gfile.GFile(f"skipnet.b{batch_size}.tfgraph", "wb") as f:
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
    # print("inputs:")
    # for onnx_name in model.inputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # print("outputs:")
    # for onnx_name in model.outputs:
    #     print(onnx_name, model.tensor_dict[onnx_name])
    # inputs:
    # x.1 Tensor("x.1:0", shape=(1, 3, 224, 224), dtype=float32)
    # ch.1 Tensor("ch.1:0", shape=(1, 1, 10), dtype=float32)
    # cc.1 Tensor("cc.1:0", shape=(1, 1, 10), dtype=float32)
    # outputs:
    # x.276 Tensor("add_264:0", shape=(1, 1000), dtype=float32)
    # state_h.123 Tensor("onnx_tf_prefix_Mul_2384:0", shape=(1, 1, 10), dtype=float32)
    # state_c.123 Tensor("onnx_tf_prefix_Add_2382:0", shape=(1, 1, 10), dtype=float32)
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model, name="")
        # warm up
        for i in range(0, len_dataset, batch_size):
            if i >= warmup * batch_size: break
            inputs = inputs_all[i: i + batch_size]
            ch = np.zeros((1, batch_size, 10))
            cc = np.zeros((1, batch_size, 10))
            outputs = session.run(('add_264:0', 'onnx_tf_prefix_Mul_2384:0', 'onnx_tf_prefix_Add_2382:0'), {
                'x.1:0': inputs,
                'ch.1:0': ch,
                'cc.1:0': cc
            })
        # run
        profile_start(platform)
        iter_times = []
        for i in range(0, len_dataset, batch_size):
            if i >= num_iter * batch_size: break
            inputs = inputs_all[i: i + batch_size]
            ch = np.zeros((1, batch_size, 10))
            cc = np.zeros((1, batch_size, 10))
            start_time = time.time()
            outputs = session.run(('add_264:0', 'onnx_tf_prefix_Mul_2384:0', 'onnx_tf_prefix_Add_2382:0'), {
                'x.1:0': inputs,
                'ch.1:0': ch,
                'cc.1:0': cc
            })
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))


def test_fix_model(unroll, cond_control, rate):
    print("----unroll={}---rate={}----".format(unroll, rate))
    import onnx
    from onnx_tf.backend import prepare
    rate_tag = "skip" if rate == -1 else f"{rate}"
    if unroll:
        model = onnx.load(f'onnx/skipnet.b1.unroll.{rate_tag}.onnx')
    else:
        model = onnx.load(f'onnx/skipnet.b1.fix.{rate_tag}.onnx')

    model = prepare(model)
    output_names = tuple([model.tensor_dict[onnx_name].name for onnx_name in model.outputs])
    # print("inputs:")
    # for onnx_name in model.inputs:
    #     print(onnx_name, model.tensor_dict[onnx_name].name)
    # print("outputs:")
    # for onnx_name in model.outputs:
    #     print(onnx_name, model.tensor_dict[onnx_name].name)
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=0
    )
    batch_size = 1
    inputs = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    ch = np.zeros((1, batch_size, 10))
    cc = np.zeros((1, batch_size, 10))
    print("cond_control", cond_control.shape, cond_control.dtype)
    cond_control = cond_control * 1000000 - 500000
    session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        tf.import_graph_def(model.graph.as_graph_def(), name="")
        # warm up
        for i in range(warmup):
            if unroll:
                outputs = session.run(output_names, {
                    'x.1:0': inputs,
                    'ch.1:0': ch,
                    'cc.1:0': cc
                })
            else:
                outputs = session.run(output_names, {
                    'x.1:0': inputs,
                    'ch.1:0': ch,
                    'cc.1:0': cc,
                    'cond_control.1:0': cond_control
                })
        # run
        profile_start(platform)
        iter_times = []
        for i in range(0, len_dataset, batch_size):
            if i >= num_iter * batch_size: break
            start_time = time.time()
            if unroll:
                outputs = session.run(output_names, {
                    'x.1:0': inputs,
                    'ch.1:0': ch,
                    'cc.1:0': cc
                })
            else:
                outputs = session.run(output_names, {
                    'x.1:0': inputs,
                    'ch.1:0': ch,
                    'cc.1:0': cc,
                    'cond_control.1:0': cond_control
                })
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))


if not arguments.overhead_test:
    test_model(arguments.bs, False)
else:
    if arguments.rate == -1: # real case
        actions = [
            1, 0,
            1, 0, 1, 1,
            1, 0, 1, 1, 0,
            1, 1, 0, 1, 1,
            0, 1, 1, 0, 1,
            1, 0, 1, 1, 0,
            1, 1, 1,
            1, 0, 1, 1,
        ]
    elif arguments.rate == 0:
            actions = [0] * 32
    elif arguments.rate == 25:
        actions = [
            0, 1,
            0, 1, 0, 0,
            0, 0, 0, 0, 1,
            0, 0, 1, 0, 0,
            1, 0, 0, 1, 0,
            0, 1, 0, 0, 0,
            0, 0, 0,
            0, 1, 0
        ] 
    elif arguments.rate == 50:
        actions = [
            0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1,
        ]
    elif arguments.rate == 75:
        actions = [
            1, 0,
            1, 0, 1, 1,
            1, 1, 1, 1, 0,
            1, 1, 0, 1, 1,
            0, 1, 1, 0, 1,
            1, 0, 1, 1, 1,
            1, 1, 1,
            1, 0, 1,
        ]
    elif arguments.rate == 100:
        actions = [1] * 32
    else:
        raise NotImplementedError

    cond_control = np.array(actions, dtype=np.float32)
    test_fix_model(arguments.unroll, cond_control, arguments.rate)

