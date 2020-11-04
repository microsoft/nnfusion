#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from nnf_tf_freezer import nnf_tf_freezer
import argparse
import tensorflow as tf
import sys
sys.path.append('..')
from seq2seq_model import Seq2SeqModel
from google_bert.modeling import BertConfig, BertModel
from nasnet_cifar_nchw import nasnet
from alexnet_model import AlexnetModel
import resnet_model
from lstm import LSTMModel
from deep_speech_model import DeepSpeech2
from inception_model import Inceptionv3Model
from vgg_model import Vgg11Model

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='seq2seq',
                    help='Model name.')
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
parser.add_argument('--run_const_folded_graph', action='store_true',
                    help='Run tf_run_graph_graph with const_folded graph.')                    
parser.add_argument('--debug', action='store_true',
                    help='Print log.') 
parser.add_argument('--is_training', action='store_true',
                    help='Is training graph.')
parser.add_argument('--to_fp16', action='store_true',
                    help='whether save frozen_graph in fp16 format')

args = parser.parse_args()

tf.random.set_random_seed(1)

inputs = []
outputs = []
optimizer = None

if args.model_name == 'alexnet':
    print('>> Converting graph alexnet')
    batch_size = 1
    model = AlexnetModel()
    model.batch_size = batch_size
    input_shapes = model.get_input_shapes('validation')
    dtype = model.get_input_data_types('validation')
    graph_in = tf.placeholder(dtype[0], shape=input_shapes[0])
    logits, _ = model.build_network([graph_in], phase_train=False, nclass=1001)

    inputs = [graph_in]
    if args.const_folding:
        outputs = [tf.identity(tf.identity(logits, name="logits"), name="logits_identity")]
    else:
        outputs = [tf.identity(logits, name="logits")]
elif args.model_name == 'resnet50':
    print('>> Converting graph resnet50')
    batch_size = 1
    model = resnet_model.create_resnet50_model(None)
    model.batch_size = batch_size
    input_shapes = model.get_input_shapes('validation')
    dtype = model.get_input_data_types('validation')
    graph_in = tf.placeholder(dtype[0], shape=input_shapes[0])
    logits, _ = model.build_network([graph_in], phase_train=False, nclass=1001)

    inputs = [graph_in]
    if args.const_folding:
        outputs = [tf.identity(tf.identity(logits, name="logits"), name="logits_identity")]
    else:
        outputs = [tf.identity(logits, name="logits")]
elif args.model_name == 'inception3':
    print('>> Converting graph inception3')
    batch_size = 1
    model = Inceptionv3Model()
    model.batch_size = batch_size
    input_shapes = model.get_input_shapes('validation')
    dtype = model.get_input_data_types('validation')
    graph_in = tf.placeholder(dtype[0], shape=input_shapes[0])
    logits, _ = model.build_network([graph_in], phase_train=False, nclass=1001)

    inputs = [graph_in]
    if args.const_folding:
        outputs = [tf.identity(tf.identity(logits, name="logits"), name="logits_identity")]
    else:
        outputs = [tf.identity(logits, name="logits")]
elif args.model_name == 'vgg11':
    print('>> Converting graph vgg11')
    batch_size = 1
    model = Vgg11Model()
    model.batch_size = batch_size
    input_shapes = model.get_input_shapes('validation')
    dtype = model.get_input_data_types('validation')
    graph_in = tf.placeholder(dtype[0], shape=input_shapes[0])
    logits, _ = model.build_network([graph_in], phase_train=False, nclass=1001)

    inputs = [graph_in]
    if args.const_folding:
        outputs = [tf.identity(tf.identity(logits, name="logits"), name="logits_identity")]
    else:
        outputs = [tf.identity(logits, name="logits")]
elif args.model_name == 'seq2seq':
    print('>> Converting graph seq2seq')
    batch_size = 1
    encoder_step = 1
    encoder_layer = 1
    decoder_step = 1
    decoder_layer = 1
    hidden_size = 128
    model = Seq2SeqModel(
            batch_size, hidden_size, encoder_layer, encoder_step, decoder_layer, decoder_step)
    eval_inputs = tf.placeholder(
                tf.float32, [encoder_step, batch_size, hidden_size], 'eval_input')

    eval_inputs_list = tf.split(value=eval_inputs, axis=0, num_or_size_splits=encoder_step)
    for i in range(len(eval_inputs_list)):
        eval_inputs_list[i] = tf.squeeze(eval_inputs_list[i],axis=[0])
    logits = model(eval_inputs_list)

    inputs = [eval_inputs]
    if args.const_folding:
        outputs = [tf.identity(tf.identity(logits, name="logits"), name="logits_identity")]
    else:
        outputs = [tf.identity(logits, name="logits")]
elif args.model_name == 'deepspeech2':
    print('>> Converting graph deepspeech2')
    batch_size = 1
    hidden_size = 256
    num_classes = 29
    height = 300 # voice length
    eval_inputs = tf.placeholder(
                tf.float32, [batch_size, height, 171, 1], 'eval_input')
    model = DeepSpeech2(num_rnn_layers=7, rnn_type='lstm', is_bidirectional=False,
                                                rnn_hidden_size=hidden_size, num_classes=num_classes, use_bias=True)

    logits = model(eval_inputs, False)    

    inputs = [eval_inputs]
    if args.const_folding:
        outputs = [tf.identity(tf.identity(logits, name="logits"), name="logits_identity")]
    else:
        outputs = [tf.identity(logits, name="logits")]
elif args.model_name == 'lstm':
    print('>> Converting graph lstm')  
    batch_size = 1
    hidden_size = 256
    num_layer = 10
    num_step = 100 #sequence length
    model = LSTMModel(num_layer, hidden_size)
    eval_inputs = tf.placeholder(
                tf.float32, [num_step, batch_size, hidden_size], 'eval_input')
    lstm_output, lstm_state = model.run(
                eval_inputs, batch_size, num_step)

    inputs = [eval_inputs]
    if args.const_folding:
        outputs = [tf.identity(tf.identity(lstm_output, name="logits"), name="logits_identity")]
    else:
        outputs = [tf.identity(lstm_output, name="logits")]
elif args.model_name == 'nasnet_cifar':
    print('>> Converting graph nasnet_cifar')  
    arg_scope = tf.contrib.framework.arg_scope
    slim = tf.contrib.slim
    batch_size = 1
    height, width = 32, 32
    num_classes = 10
    eval_inputs = tf.placeholder(
            tf.float32, [batch_size, height, width, 3], 'eval_input')
    with slim.arg_scope(nasnet.nasnet_cifar_arg_scope()):
        logits, end_points = nasnet.build_nasnet_cifar(eval_inputs, num_classes, is_training=False)

    inputs = [eval_inputs]
    if args.const_folding:
        outputs = [tf.identity(tf.identity(logits, name="logits"), name="logits_identity")]
    else:
        outputs = [tf.identity(logits, name="logits")]
elif args.model_name == 'bert':
    print('>> Converting graph bert')
    batch_size = 1
    seq_len = 128
    num_layers = 2
    bert_config = BertConfig(
      vocab_size=30522,
      hidden_size=1024, # 768,
      num_hidden_layers=num_layers, # 12,
      num_attention_heads=16, #12,
      intermediate_size=4096, #3072,
      type_vocab_size=2,
    )
    input_ids = tf.placeholder(tf.int32, shape=(batch_size, seq_len))
    input_mask = tf.placeholder(tf.int32, shape=(batch_size, seq_len))
    segment_ids = tf.placeholder(tf.int32, shape=(batch_size, seq_len))

    model = BertModel(
      config=bert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

    output_layer = model.get_pooled_output()
    logits = tf.layers.dense(output_layer, units=1001, activation=tf.nn.softmax)

    inputs = [input_ids, input_mask, segment_ids]
    if args.const_folding:
        outputs = [tf.identity(tf.identity(logits, name="logits"), name="logits_identity")]
    else:
        outputs = [tf.identity(logits, name="logits")]
else:
    raise Exception("Model `%s` not recognized!" % args.model_name)

if __name__ == "__main__":
    freezer = nnf_tf_freezer(args.frozen_graph, args.const_folding, args.run_graph, args.xla, args.parallel, 
        args.warmup, args.num_iter, args.run_const_folded_graph, args.debug, args.is_training, args.to_fp16)
    freezer.execute(inputs, outputs, optimizer)

