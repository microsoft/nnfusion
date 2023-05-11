import os
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# dst_dir = "/home/zimiao/project/NNFusion/test/models/onnx/"

dst_dir = "."


def save_model(model_def, file_name):
    print('The model is:\n{}'.format(model_def))
    try:
        # Some op is not standardized
        onnx.checker.check_model(model_def)
    except Exception as e:
        print("Warning: ", e)
    else:
        print('The model is checked!')
    onnx.save(model_def, os.path.join(dst_dir, file_name))


def gather_op_axis_0():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 4096, 2560])
    indices = helper.make_tensor_value_info('indices', TensorProto.INT32, [1])

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Gather',  # node name
        ['data', 'indices'],  # inputs
        ['output'],  # outputs
        domain=None,
        axis=0,  # attributes
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'gather',
        [data, indices],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "gather_axis_0.onnx")


def softmax_axis_1():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 3])

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 3])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Softmax',  # node name
        ['data'],  # inputs
        ['output'],  # outputs
        domain=None,
    axis=1)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'softmax',
        [data],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "softmax_axis_1.onnx")

def batch_mat_mul_1():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, [3, 3, 2])
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [3, 2, 1])
    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 3, 1])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'MatMul',  # node name
        ['input0', 'input1'],  # inputs
        ['output'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'MatMul',
        [input0, input1],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "batch_mat_mul_1.onnx")

def softmax_cross_entropy():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 5])
    labels = helper.make_tensor_value_info('labels', TensorProto.INT32, [3])

    # Create one output (ValueInfoProto)
    sce = helper.make_tensor_value_info('sce', TensorProto.FLOAT, [3])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'labels'],
                             outputs=['sce'],
                            domain=None,
                             reduction='mean')

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'SoftmaxCrossEntropyLoss',
        [x, labels],
        [sce],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "softmax_cross_entropy.onnx")

def softmax_grad_axis_1():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    dy = helper.make_tensor_value_info('dy', TensorProto.FLOAT, [3, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5])

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 5])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'SoftmaxGrad',  # node name
        ['dy', 'y'],  # inputs
        ['output'],  # outputs
        domain=None,
    axis=1)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'SoftmaxGrad',
        [dy, y],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "softmax_grad_axis_1.onnx")


def gelu():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 3])

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 3])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Gelu',  # node name
        ['data'],  # inputs
        ['output'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'Gelu',
        [data],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "gelu.onnx")


def depth2space():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 48, 2, 3])

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 12, 4, 6])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'DepthToSpace',  # node name
        ['data'],  # inputs
        ['output'],  # outputs
        domain=None,
        blocksize=2,
        mode='CRD')

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'DepthToSpace',
        [data],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "depth2space_crd1.onnx")

def roll():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2, 3, 4])
    shifts = helper.make_tensor_value_info('shifts', TensorProto.INT64, [2])
    dims = helper.make_tensor_value_info('dims', TensorProto.INT64, [2])
    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2, 3, 4])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'roll',  # node name
        ['input', 'shifts', 'dims'],  # inputs
        ['output'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'roll',
        [input, shifts, dims],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "roll.onnx")

def slice():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [20,10,5])
    starts = helper.make_tensor_value_info('starts', TensorProto.INT64, [3])
    ends = helper.make_tensor_value_info('ends', TensorProto.INT64, [3])
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [20,10,1])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Slice',  # node name
        ['x', 'starts', 'ends'],  # inputs
        ['y'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'Slice',
        [x, starts, ends],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "slice.onnx")

def mat_mul_1():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT16, [64, 196, 4096])
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT16, [1024, 4096])
    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT16, [64, 196, 1024])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def1 = helper.make_node(
        'Transpose',  # node name
        ['input1'],  # inputs
        ['output1'],  # outputs
        perm=[1,0],
        domain=None)
    node_def2 = helper.make_node(
        'MatMul',  # node name
        ['input0', 'output1'],  # inputs
        ['output'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def1, node_def2],
        'TransMatMul',
        [input0, input1],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "mat_mul_2.onnx")


def identity():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 3])

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 3])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Identity',  # node name
        ['data'],  # inputs
        ['output'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'identity',
        [data],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "identity.onnx")

def conv1d():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 5])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [4, 3, 2])
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 4, 6])

    node_def = helper.make_node(
        'Conv',  # node name
        ['x', 'w'],  # inputs
        ['y'],  # outputs
        kernel_shape = [2],
        pads = [1, 1],
        strides = [1],
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'Conv',
        [x, w],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "conv1d.onnx")

def globalavgpool():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 2, 3])

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 1])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'GlobalAveragePool',  # node name
        ['data'],  # inputs
        ['output'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'GlobalAveragePool',
        [data],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "globalavgpool.onnx")

def clip():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    min = helper.make_tensor_value_info('min', TensorProto.FLOAT, [])
    max = helper.make_tensor_value_info('max', TensorProto.FLOAT, [])

    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Clip',  # node name
        ['x', 'min', 'max'],  # inputs
        ['y'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'GlobalAveragePool',
        [x, min, max],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "clip.onnx")

def concat():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    x1 = helper.make_tensor_value_info('x1', TensorProto.STRING, [2, 3])
    x2 = helper.make_tensor_value_info('x2', TensorProto.STRING, [2, 3])
  
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.STRING, [2, 6])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Concat',  # node name
        ['x1', 'x2'],  # inputs
        ['y'],
        axis=1,  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'Concat',
        [x1, x2],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "concat.onnx")


def constantofshape():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.INT64, [2])
   
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3])

    tensor_value = onnx.helper.make_tensor(
    "value", onnx.TensorProto.FLOAT, [1], [0])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'ConstantOfShape',  # node name
        ['x'],  # inputs
        ['y'],  # outputs
        # value=tensor_value,
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'ConstantOfShape',
        [x],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "constantofshape.onnx")

def constant():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

 # Create one output (ValueInfoProto)

    values = np.random.randn(5).astype(np.float32)
    node_def = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['values'],
        value_floats=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=onnx.TensorProto.FLOAT,
            dims=values.shape,
            vals=values.flatten().astype(float),
        ),
    )

    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5])

    # node_def = onnx.helper.make_node(
    #         "Constant",
    #         inputs=[],
    #         outputs=["y"],
    #         value_int=onnx.helper.make_tensor(
    #             name="const_tensor1",
    #             data_type=onnx.TensorProto.INT64,
    #             dims=[],
    #             vals=[2],
    #         ),
    #     )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'Constant',
        [],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "constant.onnx")

def convtrans():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 1, 2, 2])
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])

    node_def = helper.make_node(
        'ConvTranspose',  # node name
        ['x', 'w'],  # inputs
        ['y'],  # outputs
        # pads = [1, 2, 1, 2],
        # strides = [3, 2],
        dilations=[2, 2],
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "convtrans_d.onnx")

def convtrans1d():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 3])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 2, 3])
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 5])

    node_def = helper.make_node(
        'ConvTranspose',  # node name
        ['x', 'w'],  # inputs
        ['y'],  # outputs
        pads = [0, 0],
        strides = [1],
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'ConvTranspose',
        [x, w],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "convtrans1d1.onnx")

def divb():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5])

    # Create one output (ValueInfoProto)
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [3, 4, 5])
    node = onnx.helper.make_node(
    "Div",
    inputs=["x", "y"],
    outputs=["z"],
)

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Div',  # node name
        ['x', 'y'],  # inputs
        ['z'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'Div',
        [x, y],
        [z],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "divb.onnx")

def cumsum():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    axis = helper.make_tensor_value_info('axis', TensorProto.INT32, [])
    # Create one output (ValueInfoProto)
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 3])
    node = onnx.helper.make_node(
    "CumSum",
    inputs=["x", "axis"],
    outputs=["z"],
)

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'CumSum',  # node name
        ['x', 'axis'],  # inputs
        ['z'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'CumSum',
        [x, axis],
        [z],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "cumsum.onnx")


import numpy as np
def tile():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
       # Create one input (ValueInfoProto)
   
    # Create one output (ValueInfoProto)
    n1 = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["n1"],
            value=onnx.helper.make_tensor(
                name="const_tensor1",
                data_type=onnx.TensorProto.INT64,
                dims=[2],
                vals=[1, 2],
            ),
        )

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2])
    # y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])
    # Create one output (ValueInfoProto)
    # y = helper.make_tensor_value_info('y', TensorProto.INT64, [2])

    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 4])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Tile',  # node name
        ['x', 'n1'],  # inputs
        ['z'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [n1, node_def],
        'Tile',
        [x],
        [z],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "tile.onnx")

def conv():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 960, 64, 64])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT16, [320, 960, 1, 1])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT16, [320])
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 320, 64, 64])

    node_def = helper.make_node(
        'Conv',  # node name
        ['x', 'w', 'b'],  # inputs
        ['y'],  # outputs
        kernel_shape = [1, 1],
        pads = [0, 0, 0, 0],
        strides = [1, 1],
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'Conv',
        [x, w, b],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "conv.onnx")



def instancenorm():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 20480, 32])
    w = helper.make_tensor_value_info('w', TensorProto.FLOAT16, [32])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT16, [32])
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 20480, 32])

    node_def = helper.make_node(
        'InstanceNormalization',  # node name
        ['x', 'w', 'b'],  # inputs
        ['y'],  # outputs
        epsilon=0.000009999999747378752,
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'InstanceNormalization',
        [x, w, b],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "instancenormnhwc.onnx")

def groupnorm():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 32, 32, 640])
    g = helper.make_tensor_value_info('g', TensorProto.FLOAT16, [640])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT16, [640])
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 32, 32, 640])

    node_def = helper.make_node(
        'GroupNorm',  # node name
        ['x', 'g', 'b'],  # inputs
        ['y'],  # outputs
        epsilon=0.000009999999747378752,
        activation=1,
        groups=32,
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'GroupNorm',
        [x, g, b],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "groupnorm.onnx")


def memeffattn():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    q = helper.make_tensor_value_info('q', TensorProto.FLOAT16, [2, 8, 128, 64])
    k = helper.make_tensor_value_info('k', TensorProto.FLOAT16, [2, 8, 128, 64])
    v = helper.make_tensor_value_info('v', TensorProto.FLOAT16, [2, 8, 128, 64])
    lse = helper.make_tensor_value_info('lse', TensorProto.FLOAT16, [2, 8, 128])
    m = helper.make_tensor_value_info('m', TensorProto.FLOAT16, [2, 8, 128])
    acco = helper.make_tensor_value_info('acco', TensorProto.FLOAT16, [2, 8, 128, 64])
    # Create one output (ValueInfoProto)
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 8, 128, 64])

    node_def = helper.make_node(
        'MemEffAttn',  # node name
        ['q', 'k', 'v', 'lse', 'm', 'acco'],  # inputs
        ['y'],  # outputs
        is_causal=False,
        softmax_scale=0.125,
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'MemEffAttn',
        [q, k, v, lse, m, acco],
        [y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "memeffattn128.s.onnx")


def max():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 3])
    data1 = helper.make_tensor_value_info('data1', TensorProto.FLOAT, [3, 3])

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 3])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Max',  # node name
        ['data', 'data1'],  # inputs
        ['output'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'Max',
        [data, data1],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "max.onnx")

def msa0():
    BNBL, NQ, BLQ, KD, NQ, NV, BLK, D = 512, 4, 128, 256, 4, 1, 128, 256
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    q = helper.make_tensor_value_info('q', TensorProto.FLOAT16, [BNBL, NQ, BLQ, KD])
    k = helper.make_tensor_value_info('k', TensorProto.FLOAT16, [BNBL, NQ, BLK, KD])
    v = helper.make_tensor_value_info('v', TensorProto.FLOAT16, [BNBL, NQ, NV, BLK, D])
    mask = helper.make_tensor_value_info('mask', TensorProto.FLOAT16, [NQ, NV, BLQ, BLK])
    # Create one output (ValueInfoProto)
    attn = helper.make_tensor_value_info('attn', TensorProto.FLOAT16, [BNBL, NQ, NV, BLQ, D])

    node_def = helper.make_node(
        'MultiScaleAttn0',  # node name
        ['q', 'k', 'v', 'mask'],  # inputs
        ['attn'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'MultiScaleAttn0',
        [q, k, v, mask],
        [attn],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "msa0.onnx")

def msa1():
    B, NQ, BLQ, KD, NQ, NV, BLK, D = 4, 4, 128, 256, 4, 1, 128, 256
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    q = helper.make_tensor_value_info('q', TensorProto.FLOAT16, [B, NQ, BLQ, KD])
    k = helper.make_tensor_value_info('k', TensorProto.FLOAT16, [B, NQ, BLK, KD])
    v = helper.make_tensor_value_info('v', TensorProto.FLOAT16, [B, NQ, NV, BLK, D])
    mask = helper.make_tensor_value_info('mask', TensorProto.FLOAT16, [NQ, NV, BLQ, BLK])

    cross_decay = helper.make_tensor_value_info('cross_decay', TensorProto.FLOAT16, [NQ, NV])
    inner_decay = helper.make_tensor_value_info('inner_decay', TensorProto.FLOAT16, [NQ, NV, BLQ])
    kv_state = helper.make_tensor_value_info('kv_state', TensorProto.FLOAT16, [B, NQ, NV, D, KD])

    # Create one output (ValueInfoProto)
    crossattn = helper.make_tensor_value_info('crossattn', TensorProto.FLOAT16, [B, NQ, NV, BLQ, D])
    new_kv_state = helper.make_tensor_value_info('new_kv_state', TensorProto.FLOAT16, [B, NQ, NV, KD, D])

    node_def = helper.make_node(
        'MultiScaleAttn1',  # node name
        ['q', 'k', 'v', 'mask', 'cross_decay', 'inner_decay', 'kv_state'],  # inputs
        ['crossattn', 'new_kv_state'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'MultiScaleAttn1',
        [q, k, v, mask, cross_decay, inner_decay, kv_state],
        [crossattn, new_kv_state],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "msa1.onnx")

def msa():
    B, NQ, BLQ, KD, NQ, NV, BLK, D = 4, 4, 128, 256, 4, 1, 128, 256
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create input (ValueInfoProto)
    q = helper.make_tensor_value_info('q', TensorProto.FLOAT16, [B, NQ, BLQ, KD])
    k = helper.make_tensor_value_info('k', TensorProto.FLOAT16, [B, NQ, BLK, KD])
    v = helper.make_tensor_value_info('v', TensorProto.FLOAT16, [B, NQ, NV, BLK, D])
    mask = helper.make_tensor_value_info('mask', TensorProto.FLOAT16, [NQ, NV, BLQ, BLK])

    cross_decay = helper.make_tensor_value_info('cross_decay', TensorProto.FLOAT16, [NQ, NV])
    inner_decay = helper.make_tensor_value_info('inner_decay', TensorProto.FLOAT16, [NQ, NV, BLQ])
    kv_state = helper.make_tensor_value_info('kv_state', TensorProto.FLOAT16, [B, NQ, NV, KD, D])

    # Create one output (ValueInfoProto)
    out = helper.make_tensor_value_info('out', TensorProto.FLOAT16, [B, NQ, NV, BLQ, D])

    node_def = helper.make_node(
        'MultiScaleAttn',  # node name
        ['q', 'k', 'v', 'mask', 'cross_decay', 'inner_decay', 'kv_state'],  # inputs
        ['out'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'MultiScaleAttn',
        [q, k, v, mask, cross_decay, inner_decay, kv_state],
        [out],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "msa.onnx")
def add():
    BNBL, NQ, BLQ, KD, NQ, NV, BLK, D = 4 * 128, 4, 128, 256, 4, 1, 128, 256
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

    # Create one input (ValueInfoProto)
    attn = helper.make_tensor_value_info('attn', TensorProto.FLOAT16, [BNBL, NQ, NV, BLQ, D])
    crossattn = helper.make_tensor_value_info('crossattn', TensorProto.FLOAT16, [BNBL, NQ, NV, BLQ, D])

    # Create one output (ValueInfoProto)
    out = helper.make_tensor_value_info('out', TensorProto.FLOAT16, [BNBL, NQ, NV, BLQ, D])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'Add',  # node name
        ['attn', 'crossattn'],  # inputs
        ['out'],  # outputs
        domain=None)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'Add',
        [attn, crossattn],
        [out],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    save_model(model_def, "add.onnx")

memeffattn()