import tensorflow as tf
import numpy as np
import argparse
import os
import time
import os.path as osp

# Load protobuf as graph, given filepath
def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def get_default_sess_config(compiler=None):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    if compiler == "xla":
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    elif compiler == "astitch":
        os.environ["TF_ENABLE_TAO"] = "1"
        root_path = os.path.dirname(os.environ['TAO_COMPILER_PATH'])
        tf.load_op_library(os.path.join(root_path, 'libtao_ops.so'))
    elif compiler is None:
        pass
    else:
        raise ValueError(compiler)
    return config

def run_tf(prefix, compiler=None):
    graph = load_pb(osp.join(prefix, "model.pb"))
    feed_dict = dict(np.load(osp.join(prefix, "inputs.npz"), allow_pickle=True))
    with open(osp.join(prefix, "output_names.txt")) as f:
        fetch_list = eval(f.read())
    feed_dict_ = {}
    for k, v in feed_dict.items():
        feed_dict_[k + ':0'] = v
    feed_dict = feed_dict_
    for i in range(len(fetch_list)):
        fetch_list[i] = fetch_list[i] + ":0"
    sess = tf.Session(graph=graph, config=get_default_sess_config(compiler))
    def get_runtime():
        tic = time.time()
        _ = sess.run(fetch_list, feed_dict=feed_dict)
        return time.time() - tic
    _ = [get_runtime() for i in range(50)] # warmup
    times = [get_runtime() for i in range(30)]
    print(np.mean(times), np.min(times), np.max(times))

if __name__ == "__main__":
    run_tf("temp", compiler="xla")
