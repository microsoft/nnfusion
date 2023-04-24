# refer to https://github.com/tensorflow/addons/blob/v0.15.0/tensorflow_addons/rnn/nas_cell.py#L30-L236
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.gen_linalg_ops import batch_self_adjoint_eig


flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.ERROR)

flags.DEFINE_integer("num_step", 1000, "sequence length")
flags.DEFINE_integer("hidden_size", 256, "hidden size")
flags.DEFINE_boolean('profile', False, 'profile kernel runtime')
flags.DEFINE_string('backend', 'tf', 'tf or wolong or ngraph')
flags.DEFINE_integer("num_iter", 100, "mini batch size")
flags.DEFINE_integer("warmup", 100, "mini batch size")
flags.DEFINE_integer("parallel", 0, "tf.ConfigProto.inter_op_parallelism_threads")
flags.DEFINE_integer("bs", 1, "mini batch size")
flags.DEFINE_string('platform', 'V100', 'V100 or MI100')
flags.DEFINE_bool('overhead_test', False, 'overhead test')
flags.DEFINE_bool('unroll', False, 'unroll or not')

FLAGS = flags.FLAGS
platform = FLAGS.platform
import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

class NASCell(tf.keras.layers.AbstractRNNCell):
    """Neural Architecture Search (NAS) recurrent network cell.
    This implements the recurrent cell from the paper:
      https://arxiv.org/abs/1611.01578
    Barret Zoph and Quoc V. Le.
    "Neural Architecture Search with Reinforcement Learning" Proc. ICLR 2017.
    The class uses an optional projection layer.
    Example:
    >>> inputs = np.random.random([30,23,9]).astype(np.float32)
    >>> NASCell = tfa.rnn.NASCell(4)
    >>> rnn = tf.keras.layers.RNN(NASCell, return_sequences=True, return_state=True)
    >>> outputs, memory_state, carry_state = rnn(inputs)
    >>> outputs.shape
    TensorShape([30, 23, 4])
    >>> memory_state.shape
    TensorShape([30, 4])
    >>> carry_state.shape
    TensorShape([30, 4])
    """

    # NAS cell's architecture base.
    _NAS_BASE = 8

    def __init__(
        self,
        units,
        projection = None,
        use_bias: bool = False,
        kernel_initializer = "glorot_uniform",
        recurrent_initializer = "glorot_uniform",
        projection_initializer = "glorot_uniform",
        bias_initializer = "zeros",
        **kwargs,
    ):
        """Initialize the parameters for a NAS cell.
        Args:
          units: int, The number of units in the NAS cell.
          projection: (optional) int, The output dimensionality for the
            projection matrices.  If None, no projection is performed.
          use_bias: (optional) bool, If True then use biases within the cell.
            This is False by default.
          kernel_initializer: Initializer for kernel weight.
          recurrent_initializer: Initializer for recurrent kernel weight.
          projection_initializer: Initializer for projection weight, used when
            projection is not None.
          bias_initializer: Initializer for bias, used when use_bias is True.
          **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.units = units
        self.projection = projection
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.projection_initializer = projection_initializer
        self.bias_initializer = bias_initializer

        if projection is not None:
            self._state_size = [units, projection]
            self._output_size = projection
        else:
            self._state_size = [units, units]
            self._output_size = units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        input_size = tf.compat.dimension_value(
            tf.TensorShape(inputs_shape).with_rank(2)[1]
        )

        if input_size is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # Variables for the NAS cell. `recurrent_kernel` is all matrices
        # multiplying the hidden state and `kernel` is all matrices multiplying
        # the inputs.
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.output_size, self._NAS_BASE * self.units],
            initializer=self.recurrent_initializer,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_size, self._NAS_BASE * self.units],
            initializer=self.kernel_initializer,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self._NAS_BASE * self.units],
                initializer=self.bias_initializer,
            )
        # Projection layer if specified
        if self.projection is not None:
            self.projection_weights = self.add_weight(
                name="projection_weights",
                shape=[self.units, self.projection],
                initializer=self.projection_initializer,
            )

        self.built = True

    def call(self, inputs, state):
        """Run one step of NAS Cell.
        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: This must be a list of state Tensors, both `2-D`, with column
            sizes `c_state` and `m_state`.
        Returns:
          A tuple containing:
          - A `2-D, [batch x output_dim]`, Tensor representing the output of
            the NAS Cell after reading `inputs` when previous state was
            `state`.
            Here output_dim is:
               projection if projection was set, units otherwise.
          - Tensor(s) representing the new state of NAS Cell after reading
            `inputs` when the previous state was `state`.  Same type and
            shape(s) as `state`.
        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        sigmoid = tf.math.sigmoid
        tanh = tf.math.tanh
        relu = tf.nn.relu

        c_prev, m_prev = state

        m_matrix = tf.matmul(m_prev, self.recurrent_kernel)
        inputs_matrix = tf.matmul(inputs, self.kernel)

        if self.use_bias:
            m_matrix = tf.nn.bias_add(m_matrix, self.bias)

        # The NAS cell branches into 8 different splits for both the hidden
        # state and the input
        m_matrix_splits = tf.split(
            axis=1, num_or_size_splits=self._NAS_BASE, value=m_matrix
        )
        inputs_matrix_splits = tf.split(
            axis=1, num_or_size_splits=self._NAS_BASE, value=inputs_matrix
        )

        # First layer
        layer1_0 = sigmoid(inputs_matrix_splits[0] + m_matrix_splits[0])
        layer1_1 = relu(inputs_matrix_splits[1] + m_matrix_splits[1])
        layer1_2 = sigmoid(inputs_matrix_splits[2] + m_matrix_splits[2])
        layer1_3 = relu(inputs_matrix_splits[3] * m_matrix_splits[3])
        layer1_4 = tanh(inputs_matrix_splits[4] + m_matrix_splits[4])
        layer1_5 = sigmoid(inputs_matrix_splits[5] + m_matrix_splits[5])
        layer1_6 = tanh(inputs_matrix_splits[6] + m_matrix_splits[6])
        layer1_7 = sigmoid(inputs_matrix_splits[7] + m_matrix_splits[7])

        # Second layer
        l2_0 = tanh(layer1_0 * layer1_1)
        l2_1 = tanh(layer1_2 + layer1_3)
        l2_2 = tanh(layer1_4 * layer1_5)
        l2_3 = sigmoid(layer1_6 + layer1_7)

        # Inject the cell
        l2_0 = tanh(l2_0 + c_prev)

        # Third layer
        l3_0_pre = l2_0 * l2_1
        new_c = l3_0_pre  # create new cell
        l3_0 = l3_0_pre
        l3_1 = tanh(l2_2 + l2_3)

        # Final layer
        new_m = tanh(l3_0 * l3_1)

        # Projection layer if specified
        if self.projection is not None:
            new_m = tf.matmul(new_m, self.projection_weights)

        return new_m, (new_c, new_m)

    def get_config(self):
        config = {
            "units": self.units,
            "projection": self.projection,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer,
            "projection_initializer": self.projection_initializer,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class NasRNNNet(tf.keras.layers.Layer):
    def __init__(self, batch_size, hidden_size):
        self.cell = NASCell(hidden_size)
        self.cell.build((batch_size, hidden_size))
        self.hidden_size = hidden_size

    def op_body(self, i, inputs, state, seq_len):
        inp = inputs[i]
        _, state = self.cell.call(inp, state)
        return i + 1, inputs, state, seq_len

    def call_loop(self, inputs):
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        state_c = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
        state_m = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
        for step in range(seq_len):
            _, (state_c, state_m) = self.cell(inputs[step], (state_c, state_m))
        return state_m            

    def call_op(self, inputs):
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        cond = lambda i, a, b, seq_len: i < seq_len
        state_c = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
        state_m = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
        _, _, (_, new_m), _ = tf.while_loop(cond, self.op_body, (0, inputs, (state_c, state_m), seq_len))
        return new_m


def test_model(batch_size, enable_xla, enable_while, enable_training):
    print("----batch_size={}---xla={}---while={}---train={}----".format(batch_size,
          enable_xla, enable_while, enable_training))

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        graph_options=tf.GraphOptions(infer_shapes=True),
        inter_op_parallelism_threads=FLAGS.parallel
    )

    if enable_xla:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    else:
        session_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.L1

    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        profile_stop(platform)

        model = NasRNNNet(batch_size, FLAGS.hidden_size)

        eval_inputs = tf.placeholder(
            tf.float32, [FLAGS.num_step, batch_size, FLAGS.hidden_size], 'eval_input')

        nas_inputs = np.ones((FLAGS.num_step, batch_size, FLAGS.hidden_size))


        if enable_while:
            nas_output = model.call_op(eval_inputs)
            feed_dict = {eval_inputs: nas_inputs}
        else:
            nas_output = model.call_loop(eval_inputs)
            feed_dict = {eval_inputs: nas_inputs}
        
        nodes = [nas_output]
        if enable_training:
            grad = tf.gradients(tf.math.reduce_sum(nas_output), tf.trainable_variables())
            nodes.append(grad)

        session.run(tf.global_variables_initializer())

        # warm up
        for i in range(FLAGS.warmup):
            start_time = time.time()
            res = session.run(nodes, feed_dict)
            iter_time = (time.time() - start_time) * 1000
            print("Iteration time %f ms" % (iter_time))
            if i <= 1 and not enable_training:
                out_flat = res[0].reshape(-1)
                max_len = min(10, out_flat.shape[0])
                print(out_flat[:max_len], "...(size=",
                        res[0].shape, "end with", out_flat[-1], ")")

        iter_times = []
        profile_start(platform)
        for i in range(FLAGS.num_iter):
            start_time = time.time()
            res = session.run(nodes, feed_dict)
            iter_time = (time.time() - start_time) * 1000
            iter_times.append(iter_time)
            # print("Iteration time %f ms" % (iter_time))
        profile_stop(platform)

        print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))


def main(_):
    if not FLAGS.overhead_test:
        test_model(FLAGS.bs, False, True, False)
    else:
        if FLAGS.unroll:
            test_model(1, False, False, False)
        else:
            test_model(1, False, True, False)
    # test_model(1, False, True, False)
    # test_model(1, True, True, False)
    # test_model(64, False, True, False)
    # test_model(64, True, True, False)
    # test_model(1, False, False, False)
    

if __name__ == "__main__":
    tf.app.run()
