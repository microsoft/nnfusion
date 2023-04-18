"""Non-differentiable functions.

Not in the mathematical sense, but in the sense of them providing zero gradient
because they provide meta-information (shape) do integer arithmetic, or are
tensor constructors.

"""
import numpy
import torch

NON_DIFFERENTIABLE = set([
    len,
    numpy.shape, numpy.zeros, numpy.ones, numpy.zeros_like, numpy.ones_like,
    torch.ones, torch.zeros, torch.Tensor.size
])


def register_non_differentiable_functions(*funcs):
  global NON_DIFFERENTIABLE
  NON_DIFFERENTIABLE |= set(funcs)
