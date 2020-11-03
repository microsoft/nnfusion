import tensorflow as tf
from tensorflow.core.framework import types_pb2, graph_pb2, attr_value_pb2
from tensorflow.tools.graph_transforms import TransformGraph
from google.protobuf import text_format
import numpy as np


def convert_graph_to_fp16(source_graph_def, target_type='fp16', input_name=None, output_names=None, keep_fp32_node_name=[]):
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp64':
        dtype = types_pb2.DT_DOUBLE
    else:
        dtype = types_pb2.DT_FLOAT

    target_graph_def = graph_pb2.GraphDef()
    target_graph_def.versions.CopyFrom(source_graph_def.versions)
    for node in source_graph_def.node:
        # replicate node
        new_node = target_graph_def.node.add()
        new_node.op = node.op
        new_node.name = node.name
        new_node.input.extend(node.input)
        attrs = list(node.attr.keys())
        # replace dtype in node attr with target dtype
        for attr in attrs:
            # keep special node in fp32
            new_node.attr[attr].CopyFrom(node.attr[attr])
            if node.name in keep_fp32_node_name:
                continue
            if node.attr[attr].type == types_pb2.DT_FLOAT:
                # modify node dtype
                new_node.attr[attr].type = dtype
            if attr == "value":
                tensor = node.attr[attr].tensor
                if tensor.dtype == types_pb2.DT_FLOAT:
                    # if float_val exists
                    if tensor.float_val:
                        float_val = tf.make_ndarray(node.attr[attr].tensor)
                        new_node.attr[attr].tensor.CopyFrom(tf.make_tensor_proto(float_val, dtype=dtype))
                        continue
                    # if tensor content exists
                    if tensor.tensor_content:
                        tensor_shape = [x.size for x in tensor.tensor_shape.dim]
                        tensor_weights = tf.make_ndarray(tensor)
                        # reshape tensor
                        tensor_weights = np.reshape(tensor_weights, tensor_shape)
                        new_node.attr[attr].tensor.CopyFrom(tf.make_tensor_proto(tensor_weights, dtype=dtype))
                        continue
    # transform graph
    if output_names:
        if not input_name:
            input_name = []
        transforms = ["strip_unused_nodes"]
        target_graph_def = TransformGraph(target_graph_def, input_name, output_names, transforms)
    # write graph_def to model
    print("Converting done ...")
    return target_graph_def