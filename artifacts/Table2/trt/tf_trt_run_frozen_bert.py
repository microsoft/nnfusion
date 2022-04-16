import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import time
import sys
flags = tf.flags
flags.DEFINE_integer("num_iter", 10, "num of iterations")
flags.DEFINE_string("model_path", "", "path of frozen model")
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_string("model", "", "model name")
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
class BaseNet:
    input_node = "eval_input"
    image_size = (1, 3, 32, 32)
    name = "basenet"
    output_node = ["output"]
    def parse(self, sess):
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open(FLAGS.model_path, "rb") as f:
            data = f.read()
            input_graph_def.ParseFromString(data)
        return input_graph_def
    def execute(self, sess):
        profile_stop()
        trt_graph = trt.create_inference_graph(
            input_graph_def=self.parse(sess),
            outputs=self.output_node,
            precision_mode='FP32',
            max_batch_size=FLAGS.batch_size
        )
        output_node = tf.import_graph_def(trt_graph, return_elements=self.output_node)
        rand_image = np.ones(shape=self.image_size)
        input_image = sess.graph.get_tensor_by_name("import/" + self.input_node + ":0")
        output_tensor = sess.graph.get_tensor_by_name("import/" + self.output_node[0] + ":0")
        # warm up
        for i in range(5):
            outputs = sess.run(output_tensor, feed_dict={input_image: rand_image})
            out_flat = outputs.flat
            if (len(out_flat) > 0):
                max_len = min(10, len(out_flat))
                print(output_tensor.name)
                print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")
        iter_times = []
        profile_start()
        for i in range(FLAGS.num_iter):
            start_time = time.time()
            sess.run(output_tensor, feed_dict={input_image: rand_image})
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
            print("Iteration time %f ms" % (iter_time))
        profile_stop()
        print("Summary: [min, max, mean] = [%f, %f, %f] ms" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))
class LSTMBS128(BaseNet):
    def __init__(self):
        self.name = "lstm"
        self.image_size = (100, 128, 256)
        self.output_node = ["mul_2999"]
class NasnetLargeNchwBS128(BaseNet):
    def __init__(self):
        self.name = "nasnet_large_nchw"
        self.image_size = (128, 3, 331, 331)
        self.output_node = ["final_layer/FC/BiasAdd"]
class ResNetNchwBS128(BaseNet):
    def __init__(self):
        self.name = "resnet_nchw"
        self.image_size = (128, 3, 224, 224)
        self.input_node = "eval_input"
        self.output_node = ["resnet_model/final_dense"]
# class LSTM(BaseNet):
#     def __init__(self):
#         self.name = "lstm"
#         self.image_size = (100, 1, 256)
#         self.output_node = ["mul_2999"]
# class Seq2seq(LSTM):
#     def __init__(self):
#         self.name = "seq2seq"
#         self.image_size = (100, 1, 128)
#         self.output_node = ["extended_multi_rnn_cell/decoder_rnn_fw_3/Mul_89"]
# class DeepSpeech(BaseNet):
#     def __init__(self):
#         self.name = "deepspeech2"
#         self.image_size = (1, 300, 171, 1)
#         self.output_node = ["dense/BiasAdd"]
# class NasnetCifarNchw(BaseNet):
#     def __init__(self):
#         self.name = "nasnet_cifar_nchw"
#         self.image_size = (1, 32, 32, 3)
#         self.output_node = ["final_layer/FC/BiasAdd"]
# class ResNextNchw(BaseNet):
#     def __init__(self):
#         self.name = "resnext_nchw"
#         self.image_size = (1, 3, 32, 32)
#         self.input_node = "input"
#         self.output_node = ["linear/MatMul"]
# class ResNextNchwBS4(BaseNet):
#     def __init__(self):
#         self.name = "resnext_nchw_bs4"
#         self.image_size = (4, 3, 32, 32)
#         self.input_node = "input"
#         self.output_node = ["linear/MatMul"]
# class ResNextNchwBS16(BaseNet):
#     def __init__(self):
#         self.name = "resnext_nchw_bs16"
#         self.image_size = (16, 3, 32, 32)
#         self.input_node = "input"
#         self.output_node = ["linear/MatMul"]
# class AlexnetNchw(BaseNet):
#     def __init__(self):
#         self.name = "alexnet_nchw"
#         self.image_size = (1, 227, 227, 3)
#         self.input_node = "eval_input"
#         self.output_node = ["cg/affine2/xw_plus_b"]
class BertLarge:
    # input_node = "eval_input"
    # image_size = (1, 3, 32, 32)
    name = "bert"
    output_node = ["bert/encoder/Reshape_25"]
    def parse(self, sess):
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open(FLAGS.model_path, "rb") as f:
            data = f.read()
            input_graph_def.ParseFromString(data)
        return input_graph_def
    def execute(self, sess):
        profile_stop()
        trt_graph = trt.create_inference_graph(
            input_graph_def=self.parse(sess),
            outputs=self.output_node,
            precision_mode='FP32',
            max_batch_size=FLAGS.batch_size
        )
        output_node = tf.import_graph_def(trt_graph, return_elements=self.output_node)
        input_ids_tensor = np.ones(shape=[128, 512])
        input_mask_tensor = np.ones(shape=[128, 512])
        token_type_ids_tensor = np.ones(shape=[128, 512])
        input_ids = sess.graph.get_tensor_by_name("import/input_ids:0")
        input_mask = sess.graph.get_tensor_by_name("import/input_mask:0")
        token_type_ids = sess.graph.get_tensor_by_name("import/token_type_ids:0")
        output_tensor = sess.graph.get_tensor_by_name("import/" + self.output_node[0] + ":0")
        # warm up
        for i in range(5):
            outputs = sess.run(output_tensor, feed_dict={input_ids: input_ids_tensor, input_mask: input_mask_tensor, token_type_ids: token_type_ids_tensor})
            out_flat = outputs.flat
            if (len(out_flat) > 0):
                max_len = min(10, len(out_flat))
                print(output_tensor.name)
                print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")
        iter_times = []
        profile_start()
        for i in range(FLAGS.num_iter):
            start_time = time.time()
            sess.run(output_tensor, feed_dict={input_ids: input_ids_tensor, input_mask: input_mask_tensor, token_type_ids: token_type_ids_tensor})
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
            print("Iteration time %f ms" % (iter_time))
        profile_stop()
        print("Summary: [min, max, mean] = [%f, %f, %f] ms" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))
if __name__ == "__main__":
    net = globals()[FLAGS.model]()
    sess = tf.Session()
    net.execute(sess)