import torch


def unbroadcast(x, like):
    if x.shape == like.shape:
        return x

    if len(x.shape) < len(like.shape):
        raise ValueError("unbroadcast: x.shape = {}, like.shape = {}".format())
    if len(x.shape) > len(like.shape):
        x = torch.sum(x, dim = tuple(range(len(x.shape) - len(like.shape))), keepdim = False)

    if x.shape == like.shape:
        return x

    axis = []
    for i, (a, b) in enumerate(zip(x.shape, like.shape)):
        if a != b:
            assert(b == 1)
            axis.append(i)
    ret = torch.sum(x, dim = axis, keepdim = True)
    return ret
    
