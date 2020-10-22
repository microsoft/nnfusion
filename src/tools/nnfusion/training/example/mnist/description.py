import torch

class IODescription(object):
    def __init__(self, name, shape, dtype=None, num_classes=None):
        self.name_ = name
        self.shape_ = shape
        self.dtype_ = dtype
        self.num_classes_ = num_classes


class ModelDescription(object):
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