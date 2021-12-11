# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
cuda_device = torch.device('cuda:0')
from . import dtypes


class IODescription(object):
    """ A hashable description for NNFusion model input/output.

    Attributes:
        name: A string representing name.
        shape: A sequence of ints representing shape.
        dtype: A string representing element type
        num_classes: An int if the element is a integer and
            in the range of [0, num_classes-1].
    """
    def __init__(self, name, shape, dtype=None, num_classes=None):
        self._name = name
        self._shape = tuple(shape)
        self._dtype = dtype
        self._num_classes = num_classes

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_classes(self):
        return self._num_classes

    def __hash__(self):
        return hash(
            (self.name, tuple(self.shape), self.dtype, self.num_classes))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.name, tuple(self.shape), self.dtype,
                    self.num_classes) == (other.name, tuple(other.shape),
                                          other.dtype, other.num_classes)
        return False

    def __ne__(self, other):
        return not (self == other)

    def get_torch_cuda_buffer(self):
        return torch.empty(self.shape, dtype=dtypes.str2type[self._dtype].torch_type, device=cuda_device)


class ModelDescription(object):
    """ A model description for PyTorch models.

    Attributes:
        inputs: A sequence of input IODescription.
        outputs: A sequence of output IODescription.
    """
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs
