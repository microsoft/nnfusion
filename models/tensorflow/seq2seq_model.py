# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf


class ExtendedMultiRnnCell(object):

    def __init__(self, cells):
        self.cells = cells

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "extended_multi_rnn_cell"):
            cur_input = inputs
            new_states = []
            for i in range(len(self.cells)):
                cell = self.cells[i]
                cur_state = state[i]
                next_input, new_state = cell(cur_input, cur_state)
                cur_input = next_input
                new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_input, new_states


class Seq2SeqModel(object):

    def __init__(self, batch_size, hidden_size, num_encoder_layer, encoder_step, num_decoder_layer, decoder_step):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layer = num_decoder_layer
        self.encoder_step = encoder_step
        self.decoder_step = decoder_step

    def _rnn_layer(self, inputs, rnn_hidden_size, seq_length, layer_id):
        """Defines a rnn layer.
        Args:
            inputs: input tensors for the current layer.
            rnn_hidden_size: an integer for the dimensionality of the rnn output space.
            seq_length: input sequence length
            layer_id: an integer for the index of current layer.
        Returns:
            tensor output for the current layer.
        """
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell

        # Construct forward/backward RNN cells.
        fw_cell = rnn_cell(num_units=rnn_hidden_size,
                           name="encoder_rnn_fw_{}".format(layer_id))

        rnn_outputs, _ = tf.nn.static_rnn(
            fw_cell, inputs, dtype=tf.float32)

        return rnn_outputs, _

    def _build_encoder(self, inputs):
        for layer_counter in range(self.num_encoder_layer):
            inputs, encoder_state = self._rnn_layer(
                inputs, self.hidden_size, self.encoder_step, layer_counter + 1)

        return encoder_state

    def _build_decoder(self, inputs):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
        cells = []
        for layer_counter in range(self.num_decoder_layer):
            cell = rnn_cell(num_units=self.hidden_size,
                            name="decoder_rnn_fw_{}".format(layer_counter))
            cells.append(cell)
        if len(cells) > 1:
            final_cell = ExtendedMultiRnnCell(cells)
        else:
            final_cell = cells[0]

        cur_states = []
        for i in range(len(cells)):
            cur_states.append(cells[i].zero_state(self.batch_size, dtype=tf.float32))
        if len(cur_states) == 1:
            cur_states=cur_states[0]

        cur_input = inputs[0]

        for step in range(self.decoder_step):
            next_input, next_states = final_cell(cur_input, cur_states)
            cur_input = next_input
            cur_states = next_states

        return cur_input

    def __call__(self, inputs):
        encoder_state = self._build_encoder(inputs)
        decoder_state = self._build_decoder(encoder_state)

        return decoder_state


