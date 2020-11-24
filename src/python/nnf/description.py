# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


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
        self.shape_ = tuple(shape)
        self.dtype_ = dtype
        self.num_classes_ = num_classes

    def __hash__(self):
        return hash(
            (self.name_, tuple(self.shape_), self.dtype_, self.num_classes_))

    def __eq__(self, other):
        return (self.name_, tuple(self.shape_), self.dtype_,
                self.num_classes_) == (other.name_, tuple(other.shape_),
                                       other.dtype_, other.num_classes_)

    def __ne__(self, other):
        return not (self == other)


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
        return torch.ones(size, dtype=desc.dtype_).to(device)