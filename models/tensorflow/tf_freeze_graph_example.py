#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from tensorflow_freezer import tf_freezer
import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--frozen_graph', type=str, default='frozen_graph.pb',
                    help='The file name of the frozen graph.')
parser.add_argument('--const_folding', action='store_true',
                    help='Run tf_run_const_folding.')
parser.add_argument('--run_graph', action='store_true',
                    help='Run tf_run_frozen_graph.')
parser.add_argument('--xla', action='store_true',
                    help='Run tf_run_frozen_graph with xla enabled.')
parser.add_argument('--parallel', type=int, default=0,
                    help='tf.ConfigProto.inter_op_parallelism_threads.')
parser.add_argument('--warmup', type=int, default=5,
                    help='Warmup steps when runing tf_run_frozen_graph.')
parser.add_argument('--num_iter', type=int, default=10,
                    help='Iteration steps when runing tf_run_frozen_graph.') 
parser.add_argument('--profile', action='store_true',
                    help='Profile kernel run time.')  
parser.add_argument('--run_const_folded_graph', action='store_true',
                    help='Run tf_run_graph_graph with const_folded graph.')                    
parser.add_argument('--debug', action='store_true',
                    help='Print log.') 

args = parser.parse_args()

tf.random.set_random_seed(1)

# construct inputs and outputs, take seq2seq model as an example
from seq2seq_model import Seq2SeqModel
print('>> Converting graph seq2seq')
batch_size = 1
encoder_step = 1
encoder_layer = 1
decoder_step = 1
decoder_layer = 1
hidden_size = 128

cur_model = Seq2SeqModel(
            batch_size, hidden_size, encoder_layer, encoder_step, decoder_layer, decoder_step)
eval_inputs = tf.placeholder(
            tf.float32, [encoder_step, batch_size, hidden_size], 'eval_input')

eval_inputs_list = tf.split(value=eval_inputs, axis=0, num_or_size_splits=encoder_step)
for i in range(len(eval_inputs_list)):
    eval_inputs_list[i] = tf.squeeze(eval_inputs_list[i],axis=[0])
logits = cur_model(eval_inputs_list)

inputs = [eval_inputs]
outputs = [tf.identity(logits, name="logits")]

if __name__ == "__main__":
    freezer = tf_freezer(args.frozen_graph, args.const_folding, args.run_graph, args.xla, args.parallel, 
        args.warmup, args.num_iter, args.profile, args.run_const_folded_graph, args.debug)
    freezer.execute(inputs, outputs)


