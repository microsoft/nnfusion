import torch
from onnx import ModelProto, GraphProto, numpy_helper, load_from_string


def save_model_state(model, filename):
    graph = GraphProto()
    # exporting only once
    exported_ids = set()
    for name, tensor in model.state_dict().items():
        if id(tensor) in exported_ids:
            continue
        else:
            exported_ids.add(id(tensor))
        try:
            numpy_value = tensor.clone().cpu().numpy()
        except:
            raise RuntimeError("Parameter {}, tensor: {} can't be dumped. \
                               Sparse tensors can't be saved".format(name, tensor))
        # cloning to avoid moving tensor from/to GPU
        initializer = numpy_helper.from_array(numpy_value, name=name)
        graph.initializer.extend([initializer])
    with open(filename, mode='wb') as f:
        f.write(ModelProto(graph=graph).SerializeToString())


def load_model_state(model, filename, strict=True):
    with open(filename, mode='rb') as f:
        graph_loaded = load_from_string(f.read()).graph

    # exporting only once
    imported_ids = set()
    own_state = model.state_dict()
    for initializer in graph_loaded.initializer:
        name = initializer.name
        if name in own_state:
            try:
                tensor = own_state[name]
                numpy_value = numpy_helper.to_array(initializer)
                tensor.copy_(tensor.new(numpy_value)[:], broadcast=False)
                imported_ids.add(id(tensor))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), numpy_value.shape))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
    if strict:
        # checking that all tensors were covered
        for name, tensor in own_state.items():
            if id(tensor) not in imported_ids:
                raise KeyError('missing keys in state_dict: "{}"'.format(name))
