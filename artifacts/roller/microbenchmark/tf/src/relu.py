import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import graph_util
import sys

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)
C = 1024

if __name__ == "__main__":
    if len(sys.argv) == 3:
        N = int(sys.argv[1])
        repeat_time = int(sys.argv[2])
    print("N, C, repeat_time:", N, C, repeat_time)
    flags.DEFINE_integer("N", N, "N")
    flags.DEFINE_integer("C", C, "C")
    FLAGS = flags.FLAGS
    tf.enable_eager_execution()
    print('is eager mode: ',tf.executing_eagerly())
    a = tf.ones([FLAGS.N, FLAGS.C], tf.float32)
    t = tf.reduce_sum(a).numpy()
    st = time.time()
    for i in range(repeat_time):
        c = tf.nn.relu(a)
    x = tf.reduce_sum(c)
    _ = x.numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    pass
