from ast import arg
import tensorflow as tf
import numpy as np
import argparse
import os
import time
import os.path as osp
import onnx
from onnx_tf.backend import prepare

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_graph(onnx_file):
    onnx_model = onnx.load(onnx_file)
    tf_rep = prepare(onnx_model, device="cuda")
    outputs = [tf_rep.tensor_dict[output] for output in tf_rep.outputs]
    return tf_rep.graph, outputs

def get_default_sess_config(compiler=None):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    if compiler == "xla":
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    elif compiler == "astitch":
        os.environ["TF_ENABLE_TAO"] = "1"
        os.environ["TAO_ENABLE_FALLBACK"] = "false"
        root_path = os.path.dirname(os.environ['TAO_COMPILER_PATH'])
        tf.load_op_library(os.path.join(root_path, 'libtao_ops.so'))
    elif compiler == "":
        pass
    else:
        raise ValueError(compiler)
    return config

def run_tf(prefix, compiler=None):
    graph, outputs = load_graph(osp.join(prefix, "model.onnx"))
    print("Graph loaded.")
    feed_dict = dict(np.load(osp.join(prefix, "inputs.npz"), allow_pickle=True))
    feed_dict_ = {}
    for k, v in feed_dict.items():
        feed_dict_[k + ':0'] = v
    feed_dict = feed_dict_
    sess = tf.Session(graph=graph, config=get_default_sess_config(compiler))
    def get_runtime():
        tic = time.time()
        _ = sess.run(outputs, feed_dict=feed_dict)
        return (time.time() - tic) * 1000
    _ = [get_runtime() for i in range(50)] # warmup
    times = [get_runtime() for i in range(100)]
    print(np.mean(times), np.min(times), np.max(times))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=str, default="")
    parser.add_argument('--prefix', type=str, default="temp")
    args = parser.parse_args()
    run_tf(args.prefix, compiler=args.compiler)
