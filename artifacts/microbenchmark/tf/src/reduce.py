import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import graph_util
import sys

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

S_list = [[128, 512, 1024], [65536, 1024], [128, 4032, 11, 11], [128, 2048, 7, 7]]
A_list = [[2], [1], [2,3], [2,3]]
K_list = [True, True, False, True]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        id = int(sys.argv[1])
        N = S_list[id][0]
        C = S_list[id][1]
        if id != 1:
            H = S_list[id][2]
            if id > 1:
                W = S_list[id][3]
        K = K_list[id]
        repeat_time = int(sys.argv[2])
    print("id, repeat_time:", id, repeat_time)
    
    if id == 0:
        flags.DEFINE_integer("N", N, "N")
        flags.DEFINE_integer("C", C, "C")
        flags.DEFINE_integer("H", H, "H")
        flags.DEFINE_boolean("K", K, "K")
        flags.DEFINE_integer("A1", A_list[id][0], "A1")
        FLAGS = flags.FLAGS
        tf.enable_eager_execution()
        print('is eager mode: ',tf.executing_eagerly())
        a = tf.ones([FLAGS.N, FLAGS.C, FLAGS.H], tf.float32)
        t = tf.reduce_sum(a).numpy()
        st = time.time()
        for i in range(repeat_time):
            c = tf.math.reduce_sum(input_tensor=a, axis=[FLAGS.A1], keepdims=FLAGS.K)
        x = tf.reduce_sum(c)
        _ = x.numpy()
        ed = time.time()
        print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    elif id == 1:
        flags.DEFINE_integer("N", N, "N")
        flags.DEFINE_integer("C", C, "C")
        flags.DEFINE_boolean("K", K, "K")
        flags.DEFINE_integer("A1", A_list[id][0], "A1")
        FLAGS = flags.FLAGS
        tf.enable_eager_execution()
        print('is eager mode: ',tf.executing_eagerly())
        a = tf.ones([FLAGS.N, FLAGS.C], tf.float32)
        t = tf.reduce_sum(a).numpy()
        st = time.time()
        for i in range(repeat_time):
            c = tf.math.reduce_sum(input_tensor=a, axis=[FLAGS.A1], keepdims=FLAGS.K)
        x = tf.reduce_sum(c)
        _ = x.numpy()
        ed = time.time()
        print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    else:
        flags.DEFINE_integer("N", N, "N")
        flags.DEFINE_integer("C", C, "C")
        flags.DEFINE_integer("H", H, "H")
        flags.DEFINE_integer("W", W, "W")
        flags.DEFINE_boolean("K", K, "K")
        flags.DEFINE_integer("A1", A_list[id][0], "A1")
        flags.DEFINE_integer("A2", A_list[id][1], "A2")
        FLAGS = flags.FLAGS
        tf.enable_eager_execution()
        print('is eager mode: ',tf.executing_eagerly())
        a = tf.ones([FLAGS.N, FLAGS.C, FLAGS.H, FLAGS.W], tf.float32)
        t = tf.reduce_sum(a).numpy()
        st = time.time()
        for i in range(repeat_time):
            c = tf.math.reduce_sum(input_tensor=a, axis=[FLAGS.A1, FLAGS.A2], keepdims=FLAGS.K)
        x = tf.reduce_sum(c)
        _ = x.numpy()
        ed = time.time()
        print("{} ms on avg".format((ed-st)*1000.0/repeat_time))
    pass
