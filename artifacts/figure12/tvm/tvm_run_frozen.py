import tensorflow as tf

import tvm
from tvm import relay
from tvm import autotvm
import numpy as np
import os.path
import tvm.relay.testing.tf as tf_testing
import time
from tvm.contrib import graph_runtime
import logging
# logging.basicConfig(level=logging.ERROR)

flags = tf.flags
flags.DEFINE_integer("num_iter", 10, "num of iterations")
flags.DEFINE_string("model_path", "", "path of frozen model pb")
flags.DEFINE_string("autotvm_log", "", "autotvm kernel tuning log")
flags.DEFINE_string("model", "", "model name")

FLAGS = flags.FLAGS


class BaseNet:
    input_node = "eval_input"
    input_size = (1, 3, 32, 32)
    name = "basenet"
    output_node = ["output"]
    output_size = (1, 10)

    def execute(self):

        print("Import Graph to TVM Relay...")

        with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
            with tf.Session() as sess:
                graph_def = tf_testing.AddShapesToGraphDef(
                    sess, self.output_node[0])

        shape_dict = {self.input_node: self.input_size}
        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                     layout="NCHW",
                                                     shape=shape_dict, outputs=self.output_node)

        log_file = ""
        if FLAGS.autotvm_log != "":
            print("Import AutoTVM logs")
            log_file = FLAGS.autotvm_log.rstrip('.log') + '.best.log'
            autotvm.record.pick_best(FLAGS.autotvm_log, log_file)
        else:
            logging.log(logging.ERROR, "autotvm_log not provided, use tvm default kernels")

        print("Compile...")
        with autotvm.apply_history_best(log_file):
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(mod,
                                                 target="cuda",
                                                 target_host="llvm",
                                                 params=params)

        rt = graph_runtime.create(graph, lib, tvm.gpu(0))

        rt.set_input(**params)

        eval_input = np.ones(self.input_size)
        outputs = tvm.nd.empty(self.output_size)

        # warm up
        for i in range(5):
            rt.set_input(self.input_node, tvm.nd.array(
                eval_input.astype("float32")))
            rt.run()
            rt.get_output(0, outputs)
            out_flat = outputs.asnumpy().flat
            if (len(out_flat) > 0):
                max_len = min(10, len(out_flat))
                print(self.output_node[0])
                print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")

        iter_times = []

        for i in range(FLAGS.num_iter):
            start_time = time.time()
            rt.set_input(self.input_node, tvm.nd.array(
                eval_input.astype("float32")))
            rt.run()
            rt.get_output(0, outputs)
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
            print("Iteration time %f ms" % (iter_time))

        print("Summary: [min, max, mean] = [%f, %f, %f] ms" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))


class ResNextNchw(BaseNet):
    def __init__(self):
        self.name = "resnext_nchw"
        self.input_size = (1, 3, 32, 32)
        self.input_node = "input"
        self.output_node = ["linear/MatMul"]
        self.output_size = (1, 10)


class ResNextNchwBS4(BaseNet):
    def __init__(self):
        self.name = "resnext_nchw_bs4"
        self.input_size = (4, 3, 32, 32)
        self.input_node = "input"
        self.output_node = ["linear/MatMul"]
        self.output_size = (4, 10)


class ResNextNchwBS16(BaseNet):
    def __init__(self):
        self.name = "resnext_nchw_bs16"
        self.input_size = (16, 3, 32, 32)
        self.input_node = "input"
        self.output_node = ["linear/MatMul"]
        self.output_size = (16, 10)


class NasnetCifarNchw(BaseNet):
    def __init__(self):
        self.name = "nasnet_cifar_nchw"
        self.input_size = (1, 32, 32, 3)
        self.input_node = "eval_input"
        self.output_node = ["final_layer/FC/BiasAdd"]
        self.output_size = (1, 10)


class AlexnetNchw(BaseNet):
    def __init__(self):
        self.name = "alexnet_nchw"
        self.input_size = (1, 227, 227, 3)
        self.input_node = "eval_input"
        self.output_node = ["cg/affine2/xw_plus_b"]
        self.output_size = (1, 1000)


class DeepSpeech(BaseNet):
    def __init__(self):
        self.name = "deepspeech2"
        self.input_size = (1, 300, 171, 1)
        self.input_node = "eval_input"
        self.output_node = ["dense/BiasAdd"]
        self.output_size = (75, 29)


class LSTM(BaseNet):
    def __init__(self):
        self.name = "lstm"
        self.input_size = (100, 1, 256)
        self.input_node = "eval_input"
        self.output_node = ["mul_2999"]
        self.output_size = (1, 256)


class LSTMBS4(BaseNet):
    def __init__(self):
        self.name = "lstm"
        self.input_size = (100, 4, 256)
        self.input_node = "eval_input"
        self.output_node = ["mul_2999"]
        self.output_size = (4, 256)


class LSTMBS16(BaseNet):
    def __init__(self):
        self.name = "lstm"
        self.input_size = (100, 16, 256)
        self.input_node = "eval_input"
        self.output_node = ["mul_2999"]
        self.output_size = (16, 256)


class Seq2seq(LSTM):
    def __init__(self):
        self.name = "seq2seq"
        self.input_size = (100, 1, 128)
        self.input_node = "eval_input"
        self.output_node = ["extended_multi_rnn_cell/decoder_rnn_fw_3/Mul_89"]
        self.output_size = (1, 128)


if __name__ == "__main__":
    net = globals()[FLAGS.model]()
    net.execute()
