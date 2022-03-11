import tensorflow as tf
import numpy as np

with tf.device('/gpu:0'):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 3, 224, 224), name='x')
    layer = tf.keras.layers.LayerNormalization(axis=-1)
    y = layer(x)
    with tf.Session() as sess:
        a = sess.run([y], feed_dict={x: np.zeros([64, 3, 224, 224])})
