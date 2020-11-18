# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import sys
from typing import Iterable
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./frozen_graph.pb', help='The file name of the frozen graph.')
parser.add_argument('--warmup', type=int, default=5, help='The number of warmup iterations.')
parser.add_argument('--iters', type=int, default=100, help='The number of execution iterations.')
parser.add_argument('--xla', type=bool, default=False, help='Enable XLA?')
args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

graph_def = None
graph = None

print('Loading graph definition ...', file=sys.stderr)
try:
    with tf.gfile.GFile(args.file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
except BaseException as e:
    parser.exit(2, 'Error loading the graph definition: {}'.format(str(e)))

print('Importing graph ...', file=sys.stderr)
try:
    assert graph_def is not None
    with tf.Graph().as_default() as graph:  # type: tf.Graph
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='',
            op_dict=None,
            producer_op_list=None
        )
except BaseException as e:
    parser.exit(2, 'Error importing the graph: {}'.format(str(e)))

print()
print('Operations:')
assert graph is not None
ops = graph.get_operations()  # type: Iterable[tf.Operation]
input_nodes = []
last_nodes = []
for op in ops:
    print('- {0:20s} "{1}" ({2} outputs)'.format(op.type, op.name, len(op.outputs)))
    last_nodes = op.outputs
    if op.type == 'Placeholder':
        for node in op.outputs:
            input_nodes.append(node)

print()
print('Sinks (operations without outputs):')
last_outputs = []
num_nodes = len(ops)
name2nodeIdx_map = {}
for i in range(num_nodes):
    name2nodeIdx_map[ops[i].name] = i
node_outputs_ = [[] for i in range(num_nodes)]
for n in range(num_nodes):
#    if len(ops[n].outputs) > 0:
#        last_outputs.append(ops[n].outputs[0])
    op = ops[n]
    pending_count = len(op.inputs)
    for i in range(pending_count):
        input_name_id = op.inputs[i].name.split(':')
        node_outputs_[name2nodeIdx_map[input_name_id[0]]].append(n)
for n in range(num_nodes):
    if len(node_outputs_[n]) == 0 and ops[n].type != 'NoOp' and ops[n].type != 'Assert':
        print('- {0:20s} {1}'.format(ops[n].type, ops[n].name))
        last_outputs.append(ops[n].outputs[0])
        



print()
print('Sources (operations without inputs):')
for op in ops:
    if len(op.inputs) > 0:
        continue
    print('- {0:20s} {1}'.format(op.type, op.name))

print()
print('Operation inputs:')
for op in ops:
    if len(op.inputs) == 0:
        continue
    print('- {0:20s}  {1:20}'.format(op.type, op.name))
    print('  {0}'.format(', '.join(i.name for i in op.inputs)))
    # print('  {0}'.format(', '.join(i.type for i in op.inputs)))

print()
print('Tensors:')
for op in ops:
    for out in op.outputs:
        print('- {0:20} {1:10} "{2}"'.format(str(out.shape), out.dtype.name, out.name))

config = tf.ConfigProto()
if args.xla == True:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

with tf.Session(graph=graph, config=config) as sess:
    if len(last_nodes) != 1:
        raise Exception("Output tensor should be exactly one, while received number = %d" % len(last_nodes))
    # logits = graph.get_tensor_by_name('logits:0')
    logits = last_nodes[-1]
    import numpy as np
    import time
    feed_dict = {}
    for node in input_nodes:
        feed_dict[node] = np.ones(node.shape, dtype=node.dtype.as_numpy_dtype())
        #feed_dict[node] = np.ones((20,64), dtype=node.dtype.as_numpy_dtype())
    #print('>> Output Shape =', logits.shape)
    #print('>> Output Value =', sess.run(last_outputs, feed_dict=feed_dict))
    for warmup in range(args.warmup):
        outputs = sess.run(last_outputs, feed_dict=feed_dict)
        for i in range(len(outputs)):
            out_flat = outputs[i].flat
            if (len(out_flat) > 0):
                max_len = min(10, len(out_flat))
                print(last_outputs[i].name)
                print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")
                # print_offset = int(len(out_flat) / 3)
                # max_len = min(10, len(out_flat) - print_offset)
                # print(out_flat[print_offset:max_len + print_offset], "offset=", print_offset)

    print('>> Evalutating Benchmark ...')
    num_steps = args.iters
    t_start = time.time()
    for step in range(num_steps):
        sess.run(last_outputs, feed_dict=feed_dict)
    t_end = time.time()
    print('>> Average time for each run: %.4f ms;' % ((t_end - t_start) * 1e3 / num_steps))