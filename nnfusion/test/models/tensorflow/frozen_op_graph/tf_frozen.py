# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
tf.reset_default_graph()
		
# input tensor
x0 =  tf.placeholder(tf.float32, shape=[1, 3, 2, 2])
x1 =  tf.placeholder(tf.float32, shape=[3, 3, 2, 1])
		
y = tf.nn.depthwise_conv2d(x0, x1, strides=[1,1,1,1], padding='SAME', name='depthwise_conv2d')
with tf.Session() as s:
    resdata = s.run(y, feed_dict={x0:[[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]], x1:[[[[1], [2]], [[7], [8]], [[13], [14]]], [[[3], [4]], [[9], [10]], [[15], [16]]], [[[5], [6]], [[11], [12]], [[17], [18]]]]})
		    
    print("result=", list(resdata))
		    
    g = s.graph
    g_def = g.as_graph_def()
    g_def_const = graph_util.convert_variables_to_constants(input_graph_def=g_def, output_node_names=["depthwise_conv2d"], sess=s)
    graph_io.write_graph(as_text=False, name="depthwise_conv2d.pb", logdir="./",graph_or_graph_def=g_def_const)
