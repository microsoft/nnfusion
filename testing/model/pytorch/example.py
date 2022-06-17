import torch

class Example(torch.nn.Module):
    def __init__(self, n, m):
        super(Example, self).__init__()

    def forward(self, x):
        x1 = torch.sum(x, axis=-1, keepdim=True)
        x3 = x - x1
        x4 = torch.exp(x3)
        x5 = torch.sum(x4, axis=-1, keepdim=True)
        x7 = x4 / x5
        return x7
