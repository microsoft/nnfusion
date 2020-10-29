# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import onnx
import io


class IODescription(object):
    """ A tensor description for PyTorch model input/output.

    Attributes:
        name: A string representing tensor name.
        shape: A sequence of ints representing tensor shape.
        dtype: torch.Dtype representing tensor type
        num_classes: An int if the tensor is a integer and
            in the range of [0, num_classes-1].
    """
    def __init__(self, name, shape, dtype=None, num_classes=None):
        self.name_ = name
        self.shape_ = shape
        self.dtype_ = dtype
        self.num_classes_ = num_classes


class ModelDescription(object):
    """ A model description for PyTorch models.

    Attributes:
        inputs: A sequence of input IODescription.
        outputs: A sequence of output IODescription.
    """
    def __init__(self, inputs, outputs):
        self.inputs_ = inputs
        self.outputs_ = outputs


def generate_sample(desc, device=None):
    size = [s if isinstance(s, (int)) else 1 for s in desc.shape_]
    if desc.num_classes_:
        return torch.randint(0, desc.num_classes_, size,
                             dtype=desc.dtype_).to(device)
    else:
        return torch.randn(size, dtype=desc.dtype_).to(device)


def convert_model_to_onnx(model, model_desc, device, file_name):
    model.to(device)
    model.eval()

    input_names = [input.name_ for input in model_desc.inputs_]
    output_names = [output.name_ for output in model_desc.outputs_]

    sample_inputs = []
    for input_desc in model_desc.inputs_:
        input_sample = generate_sample(input_desc, device)
        sample_inputs.append(input_sample)

    sample_outputs = []
    for output_desc in model_desc.outputs_:
        output_sample = generate_sample(output_desc, device)
        sample_outputs.append(output_sample)

    torch.onnx.export(model,
                      tuple(sample_inputs),
                      file_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=12,
                      _retain_param_name=True,
                      example_outputs=tuple(sample_outputs),
                      do_constant_folding=False)

    return model


class PTFreezer(object):
    """ A class to freeze PyTorch model to ONNX format.

    Attributes:
        model: A torch.nn.Module to freeze.
        model_desc: ModelDescription for this model.
    """
    def __init__(self, model, model_desc):
        self.model = model
        self.model_desc = model_desc
        self.onnx_model = None

    def execute(self, output_path):
        """ Execute the freeze process and dump model to disk.

        Args:
            output_path: freezed ONNX format model path.
        """
        f = io.BytesIO()
        convert_model_to_onnx(self.model, self.model_desc, "cpu", f)
        self.onnx_model = onnx.load_model_from_string(f.getvalue())
        if not self._check_control_flow(self.onnx_model):
            print("Control flow not yet supported")
        if not self._check_op_availability(self.onnx_model):
            print("Model ops not fully supported")
        onnx.save(self.onnx_model, output_path)

    def _check_control_flow(self, model):
        op_types = {node.op_type for node in model.graph.node}
        return "If" not in op_types and "Loop" not in op_types

    def _check_op_availability(self, model):
        return True
