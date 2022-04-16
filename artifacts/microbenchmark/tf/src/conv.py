import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import graph_util
import sys

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

if __name__ == "__main__":
    if len(sys.argv) == 12:
        N = int(sys.argv[1])
        C = int(sys.argv[2])
        H = int(sys.argv[3])
        W = int(sys.argv[4])
        F = int(sys.argv[5])
        K = int(sys.argv[6])
        K = int(sys.argv[7])
        if '(' in sys.argv[8]:
            S = tuple(list(map(int,str(sys.argv[7])[1:-1].split(';'))))
        else:
            S = int(sys.argv[8])
        D = int(sys.argv[9])
        P = str(sys.argv[10])
        repeat_time = int(sys.argv[11])
    print("N, C, H, W, F, K, S, D, P, repeat_time:", N, C, H, W, F, K, S, D, P, repeat_time)
    flags.DEFINE_integer("N", N, "N")
    flags.DEFINE_integer("C", C, "C")
    flags.DEFINE_integer("H", H, "H")
    flags.DEFINE_integer("W", W, "W")
    flags.DEFINE_integer("F", F, "F")
    flags.DEFINE_integer("K", K, "K")
    flags.DEFINE_integer("S", S, "S")
    flags.DEFINE_integer("D", D, "D")
    flags.DEFINE_string("P", P, "P")
    FLAGS = flags.FLAGS
    tf.enable_eager_execution()
    print('is eager mode: ',tf.executing_eagerly())
    a = tf.ones([FLAGS.N, FLAGS.C, FLAGS.H, FLAGS.W], tf.float32)
    b = tf.ones([FLAGS.K, FLAGS.K, FLAGS.C, FLAGS.F], tf.float32)
    t = tf.reduce_sum(b).numpy()
    st = time.time()
    for i in range(repeat_time):
        c = tf.nn.conv2d(input=a, filter=b, strides=FLAGS.S, padding=FLAGS.P, data_format='NCHW', dilations=FLAGS.D)
    x = tf.reduce_sum(c)
    _ = x.numpy()
    ed = time.time()
    print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
