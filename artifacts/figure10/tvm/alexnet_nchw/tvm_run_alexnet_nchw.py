import tensorflow as tf

import tvm
from tvm import relay
from tvm import autotvm
import numpy as np
import os.path

import tvm.relay.testing.tf as tf_testing

import time

from tvm.contrib import graph_runtime

flags = tf.flags
flags.DEFINE_integer("num_iter", 10, "num of iterations")
flags.DEFINE_string("model_path", "", "path of frozen model pb")
flags.DEFINE_string("autotvm_log", "", "autotvm kernel tuning log")

FLAGS = flags.FLAGS

print("Import Graph to TVM Relay...")

with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(
            sess, "cg/affine2/xw_plus_b")

shape_dict = {'eval_input': (1, 227, 227, 3)}
mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout="NCHW",
                                             shape=shape_dict, outputs=["cg/affine2/xw_plus_b"])

log_file = FLAGS.autotvm_log

print("Compile...")
with autotvm.apply_history_best(log_file):
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                         target="cuda",
                                         target_host="llvm",
                                         params=params)

rt = graph_runtime.create(graph, lib, tvm.gpu(0))

eval_input = np.ones((1, 227, 227, 3))
outputs = tvm.nd.empty([1, 1000])

# warm up
for i in range(5):
    rt.set_input('eval_input', tvm.nd.array(eval_input.astype("float32")))
    rt.run()
    rt.get_output(0, outputs)
    print(outputs)

iter_times = []

for i in range(FLAGS.num_iter):
    start_time = time.time()
    rt.set_input('eval_input', tvm.nd.array(eval_input.astype("float32")))
    rt.run()
    rt.get_output(0, outputs)
    iter_time = (time.time() - start_time) * 1000
    iter_times.append(iter_time)
    print("Iteration time %f ms" % (iter_time))

print("Summary: [min, max, mean] = [%f, %f, %f] ms" % (
    min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))