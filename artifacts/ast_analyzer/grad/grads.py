import math
import types

import gast
import torch

adjoints = {}
primals = {}


def create_register(dict_):
    def register(key):
        def _(f):
            dict_[key] = f
            return f
        return _
    return register


adjoint = create_register(adjoints)
primal = create_register(primals)


# Functions: f => f, df
@adjoint(gast.FunctionDef)
def dfunction_def(adjoint_body, return_dx):
    def df():
        adjoint_body
        return_dx


@primal(gast.For)
def for_(body, i, iter_, target, save_target):
    i = 0
    for target in iter_:
        save_target
        body
        i = i + 1


@adjoint(gast.For)
def dfor_(adjoint_body, n, i2, i_tmp, load_target):
    for i_tmp in range(n):
        i2 = n - i_tmp - 1
        load_target
        adjoint_body


@adjoint(torch.mm)
def mm(y, x1, x2):
    # x1, x2, y need to be 2D tensors
    d[x1] = torch.mm(d[y], torch.transpose(x2, 0, 1))
    d[x2] = torch.mm(torch.transpose(x1, 0, 1), d[y])


@adjoint(torch.tanh)
def tanh(y, x):
    d[x] = d[y] * (1.0 - (y * y))


@adjoint(torch.sigmoid)
def sigmoid(y, x):
    d[x] = d[y] * (1.0 - y) * y


@adjoint(torch.relu)
def relu(y, x):
    cond = torch.gt(x, torch.zeros_like(x))
    d[x] = d[y] * torch.where(cond, torch.ones_like(x), torch.zeros_like(x))


@adjoint(torch.sum)
def sum(y, x):
    d[x] = d[y].expand_as(x)


# Binary ops: z = op(x, y)
@adjoint(gast.Mult)
def mult(z, x, y):
    d[x] = grad.unbroadcast(d[z] * y, x)
    d[y] = grad.unbroadcast(d[z] * x, y)


@adjoint(gast.Add)
def add(z, x, y):
    d[x] = grad.unbroadcast(d[z], x)
    d[y] = grad.unbroadcast(d[z], y)


@adjoint(gast.Sub)
def sub(z, x, y):
    d[x] = grad.unbroadcast(d[z], x)
    d[y] = -grad.unbroadcast(d[z], y)


@adjoint(gast.Div)
def div(z, x, y):
    d[x] = d[z] / y
    d[y] = -d[z] * x / (y * y)


@adjoint(torch.matmul) # TODO: only supports bk,akc->abc now
def matmul(y, x1, x2):
    # x1, x2, y need to be 2D tensors
    _tmp_a, _tmp_b, _tmp_c = y.shape
    _tmp_k = x1.shape[1]
    _tmp_C = d[y].permute((1, 0, 2)).reshape((_tmp_b, _tmp_a * _tmp_c))
    _tmp_B = x2.permute((0, 2, 1)).reshape((_tmp_a * _tmp_c, _tmp_k))
    d[x1] = torch.matmul(_tmp_C, _tmp_B)
    _tmp_A = torch.transpose(x1, 0, 1)
    d[x2] = torch.matmul(_tmp_A, d[y])


@adjoint(torch.reshape)
def reshape(y, x1, x2):
    d[x1] = torch.reshape(d[y], x1.shape)


@adjoint(torch.split)
def split(y, x, split_size_or_sections, dim):
    d[x] = torch.cat(d[y], dim=dim)
