# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf
import os, sys
import argparse
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.tools import graph_transforms
tf.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./frozen_graph.pb', help='The file name of the frozen graph.')
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
print('Placeholders:')
assert graph is not None
ops = graph.get_operations()  # type: Iterable[tf.Operation]
input_nodes = []
last_nodes = []
for op in ops:
    if op.type == 'Placeholder':
        for tensor in op.outputs:
            print('- {0:20s} {1}'.format("Tensor", tensor.name))
            input_nodes.append(tensor.name)
            
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
        last_outputs.append(ops[n].outputs[0].name)
    
g_def_const = tf.import_graph_def(graph_def, name="")
g_def_const = graph_transforms.TransformGraph(graph_def, input_nodes, last_outputs, ["fold_constants", "strip_unused_nodes", "merge_duplicate_nodes", "sort_by_execution_order"])

print()
folded_graph = args.file[:-3] + ".const_folded.pb"
print("Saving Const-folded Graph... as " + folded_graph)
graph_io.write_graph(as_text=False, name=folded_graph, logdir="./",graph_or_graph_def=g_def_const)
print("Finished.")