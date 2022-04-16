import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import graph_util
import sys

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        M = int(sys.argv[1])
        K = int(sys.argv[2])
        N = int(sys.argv[3])
        repeat_time = int(sys.argv[4])
    print("M, K, N, repeat_time:", M, K, N, repeat_time)
    flags.DEFINE_integer("m", M, "m")
    flags.DEFINE_integer("k", K, "k")
    flags.DEFINE_integer("n", N, "n")
    flags.DEFINE_integer("iter", 10, "num of iterations")
    FLAGS = flags.FLAGS
    tf.enable_eager_execution()
    print('is eager mode: ',tf.executing_eagerly())
    a = tf.ones([FLAGS.m, FLAGS.k], tf.float32)
    b = tf.ones([FLAGS.k, FLAGS.n], tf.float32)
    t = tf.reduce_sum(b).numpy()
    st = time.time()
    for i in range(repeat_time):
        c = tf.matmul(a, b)
    x = tf.reduce_sum(c)
    _ = x.numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    pass
