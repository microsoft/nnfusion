# Semi-static Type Inference Engine for Chainer/PyTorch

This directory includes the implementation of "Semi-static Type, Shape and Symbolic Shape Inference for Dynamic Computation Graphs" (MAPL 2020).

## File description

* `types.py` defines the type used in our system, unification and subtype relations.
* `type_inference.py` includes the core implementation of type inference engine
* `shape_elem.py` defines _shape element_ which represents the inferred size of each dimensions of tensors.
* `ext` and `std` includes type signatures for external libraries (e.g. Numpy, Chainer and PyTorch) and Python built-in functions respectively.

## How to run

### Requirements
* NumPy
* PyTorch
* Chainer

### Example program

The following program infers the types and shapes of the `forward` function of the `Example` class.
To try this, save the program to a file and execute it with Python3.

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

from   chainer_compiler.elichika.ast_analyzer.shape_inference.types                   import *
from   chainer_compiler.elichika.testtools.type_inference_tools import *

class Example(nn.Module):
    def __init__(self):
        super(Example, self).__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)

    def forward(self, x: TyTorchTensor(np.float32, ('bsize', 3, 'height', 'width'))):
        h1 = self.conv(x)
        h2 = F.relu(h1)
        h3 = F.max_pool2d(h2, 3, stride=2)
        return h3

def main():
    # Prepare example inputs
    model = Example()
    forward_args = (torch.rand(2, 3, 227, 227), )

    id2type, id2node = generate_type_inference_results(model, forward_args)
    print_inference_results(id2type, id2node)

if __name__=="__main__":
    main()
```

## Experiment Results

All the tests of the type inference engine are stored in [tests/elichika\_typing](https://github.com/pfnet-research/chainer-compiler/tree/master/tests/elichika_typing).

For the experiment results introduced in the paper,
see [tests/elichika\_typing/pytorch](https://github.com/pfnet-research/chainer-compiler/tree/master/tests/elichika_typing/pytorch).
