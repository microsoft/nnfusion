import torch
import torch.nn as nn
import torch.nn.functional as F

from   chainer.utils import type_check

from   ast_analyzer.shape_inference.ext.utils          import *
from   ast_analyzer.shape_inference.types              import *
from   ast_analyzer.shape_inference.ext.common         import *
from   ast_analyzer.shape_inference.ext.pytorch.nn     import *
from   ast_analyzer.shape_inference.ext.pytorch.tensor import *

__all__ = [ 'pytorch_attr_ty', 'pytorch_func_ty', 'pytorch_callable_ty' ]


pytorch_attr_ty = {
        'shape' : ty_Shape,
        'dtype' : ty_DType,
        }


pytorch_func_ty = {
        torch.is_tensor  : ty_TorchIsTensor(),

        # https://pytorch.org/docs/stable/torch.html#creation-ops
        torch.tensor     : ty_TorchTensor(),
        torch.zeros      : ty_TorchTensorOfShape(),
        torch.empty      : ty_TorchTensorOfShape(),
        torch.full      : ty_TorchTensorOfShape(),
        torch.zeros_like : ty_TorchIdentical(),
        torch.ones_like : ty_TorchIdentical(),
        torch.full_like : ty_TorchIdentical(),
        torch.ones       : ty_TorchTensorOfShape(),
        torch.rand       : ty_TorchTensorOfShape(),
        torch.randn      : ty_TorchTensorOfShape(),
        torch.randint    : ty_TorchRandint(),
        torch.from_numpy : ty_TorchFromNumpy(),

        # https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
        torch.cat       : ty_TorchCat(),
        torch.chunk     : ty_TorchChunk(),
        torch.reshape   : ty_TorchReshape(),
        torch.split     : ty_TorchSplit(),
        torch.squeeze   : ty_TorchSqueeze(),
        torch.stack     : ty_TorchStack(),
        torch.transpose : ty_TorchTranspose(),
        torch.unsqueeze : ty_TorchUnsqueeze(),
        torch.scatter  : ty_TorchIdentical(),

        # https://pytorch.org/docs/stable/torch.html#random-sampling
        torch.rand_like  : ty_TorchIdentical(),
        torch.randn_like : ty_TorchIdentical(),

        # https://pytorch.org/docs/stable/torch.html#math-operations
        torch.abs     : ty_TorchIdentical(),
        torch.cos     : ty_TorchIdentical(),
        torch.cosh    : ty_TorchIdentical(),
        torch.exp     : ty_TorchIdentical(),
        torch.log     : ty_TorchIdentical(),
        torch.sigmoid : ty_TorchIdentical(),
        torch.sin     : ty_TorchIdentical(),
        torch.sinh    : ty_TorchIdentical(),
        torch.sqrt    : ty_TorchIdentical(),
        torch.tan     : ty_TorchIdentical(),
        torch.tanh    : ty_TorchIdentical(),
        torch.relu    : ty_TorchIdentical(),
        torch.softmax : ty_TorchIdentical(),
        torch.erf     : ty_TorchIdentical(),

        torch.add     : ty_TorchArith(lambda x, y: x + y),
        torch.sub     : ty_TorchArith(lambda x, y: x - y),
        torch.mul     : ty_TorchArith(lambda x, y: x * y),
        torch.pow     : ty_TorchArith(lambda x, y: x ** y),

        torch.flatten : ty_TorchFlatten(),
        torch.matmul  : ty_TorchMatmul(),
        torch.mm      : ty_TorchMatmul(),
        torch.sum     : ty_TorchSum(),
        torch.all     : ty_TorchAll(),
        torch.mean    : ty_TorchSum(),
        torch.max     : ty_TorchMax(),
        torch.clone   : ty_TorchIdentical(),
        torch.eq      : ty_TorchCompare(),
        torch.ge      : ty_TorchCompare(),
        torch.gt      : ty_TorchCompare(),
        torch.lt      : ty_TorchCompare(),
        torch.narrow  : ty_TorchNarrow(),
        torch.softmax : ty_TorchIdentical(),
        torch.log_softmax: ty_TorchIdentical(),

        torch.where   : ty_TorchWhere(),

        # https://pytorch.org/docs/stable/nn.functional.html#pooling-functions
        F.avg_pool1d  : ty_TorchPooling(dim=1),
        F.avg_pool2d  : ty_TorchPooling(dim=2),
        F.avg_pool3d  : ty_TorchPooling(dim=3),
        F.max_pool1d  : ty_TorchPooling(dim=1),
        F.max_pool2d  : ty_TorchPooling(dim=2),
        F.max_pool3d  : ty_TorchPooling(dim=3),

        # https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
        F.relu        : ty_TorchIdentical(),
        F.softmax     : ty_TorchIdentical(),
        F.log_softmax : ty_TorchIdentical(),
        F.sigmoid     : ty_TorchIdentical(),

        # https://pytorch.org/docs/stable/nn.functional.html#dropout-functions
        F.dropout         : ty_TorchIdentical(),
        F.dropout2d       : ty_TorchIdentical(ndim_min=1),
        F.dropout3d       : ty_TorchIdentical(ndim_min=1),
        F.alpha_dropout   : ty_TorchIdentical(),

        # https://pytorch.org/docs/stable/nn.functional.html#sparse-functions
        F.embedding   : ty_TorchEmbed(),

        # https://pytorch.org/docs/stable/nn.functional.html#vision-functions
        F.interpolate : ty_TorchInterpolate(),

        # https://pytorch.org/docs/master/nn.functional.html#pad
        F.pad : ty_TorchPad(),

        F.cross_entropy: ty_TorchNNCrossEntropyLoss(),

        torch.Tensor.add  : ty_TorchArith(lambda x, y: x + y),
        torch.Tensor.add_ : ty_TorchArith(lambda x, y: x + y),
        torch.Tensor.sub  : ty_TorchArith(lambda x, y: x - y),
        torch.Tensor.sub_ : ty_TorchArith(lambda x, y: x - y),
        torch.Tensor.mul  : ty_TorchArith(lambda x, y: x * y),
        torch.Tensor.mul_ : ty_TorchArith(lambda x, y: x * y),

        torch.Tensor.chunk     : ty_TorchChunk(),
        torch.Tensor.contiguous : ty_TorchIdentical(is_float_only=False),
        torch.Tensor.cpu       : ty_TorchIdentical(is_float_only=False),
        torch.Tensor.expand    : ty_TorchExpand(),
        torch.Tensor.expand_as : ty_TorchExpandAs(),
        torch.Tensor.numpy     : ty_TorchNumpy(),
        torch.Tensor.repeat    : ty_TorchRepeat(),
        torch.Tensor.reshape    : ty_TorchReshape(),
        torch.Tensor.size      : ty_TorchSize(),
        torch.Tensor.squeeze   : ty_TorchSqueeze(),
        torch.Tensor.tolist    : ty_TensorToList(),
        torch.Tensor.transpose : ty_TorchTranspose(),
        torch.Tensor.t         : ty_TorchTranspose2D(),
        torch.Tensor.unsqueeze : ty_TorchUnsqueeze(),
        torch.Tensor.view      : ty_TorchView(),
        torch.Tensor.permute   : ty_TorchPermute(),
        torch.Tensor.argmax    : ty_TorchArgmax(),

        torch.Tensor.detach    : ty_TorchIdentical(is_float_only=False),
        torch.Tensor.float     : ty_TorchIdentical(is_float_only=False, dtype=np.dtype('float32')),
        torch.Tensor.item      : ty_TorchItem(),
        torch.Tensor.fill_     : ty_TorchIdentical(is_float_only=False),
        torch.Tensor.copy_     : ty_TorchIdentical(is_float_only=False),
        torch.Tensor.masked_fill:ty_TorchIdentical(is_float_only=False),
        torch.Tensor.topk      : ty_TorchTopk(),
        }


pytorch_callable_ty = {
        # https://pytorch.org/docs/stable/nn.html#convolution-layers
        nn.Conv1d            : ty_TorchConv(dim=1).nn,
        nn.Conv2d            : ty_TorchConv(dim=2).nn,
        nn.Conv3d            : ty_TorchConv(dim=3).nn,
        nn.ConvTranspose1d   : ty_TorchConv(dim=1, transpose=True).nn,
        nn.ConvTranspose2d   : ty_TorchConv(dim=2, transpose=True).nn,
        nn.ConvTranspose3d   : ty_TorchConv(dim=3, transpose=True).nn,

        # https://pytorch.org/docs/stable/nn.html#pooling-layers
        nn.AvgPool1d         : ty_TorchPooling(dim=1).nn,
        nn.AvgPool2d         : ty_TorchPooling(dim=2).nn,
        nn.AvgPool3d         : ty_TorchPooling(dim=3).nn,
        nn.MaxPool1d         : ty_TorchPooling(dim=1).nn,
        nn.MaxPool2d         : ty_TorchPooling(dim=2).nn,
        nn.MaxPool3d         : ty_TorchPooling(dim=3).nn,
        nn.AdaptiveAvgPool1d : ty_TorchAdaptivePooling(dim=1).nn,
        nn.AdaptiveAvgPool2d : ty_TorchAdaptivePooling(dim=2).nn,
        nn.AdaptiveAvgPool3d : ty_TorchAdaptivePooling(dim=3).nn,

        # https://pytorch.org/docs/stable/nn.html#padding-layers
        nn.ReflectionPad1d   : ty_TorchDimPad(dim=1).nn,
        nn.ReflectionPad2d   : ty_TorchDimPad(dim=2).nn,
        nn.ReplicationPad1d  : ty_TorchDimPad(dim=1).nn,
        nn.ReplicationPad2d  : ty_TorchDimPad(dim=2).nn,
        nn.ReplicationPad3d  : ty_TorchDimPad(dim=3).nn,
        nn.ZeroPad2d         : ty_TorchDimPad(dim=2).nn,
        nn.ConstantPad1d     : ty_TorchDimPad(dim=1, is_const=True).nn,
        nn.ConstantPad2d     : ty_TorchDimPad(dim=2, is_const=True).nn,
        nn.ConstantPad3d     : ty_TorchDimPad(dim=3, is_const=True).nn,

        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        nn.LeakyReLU        : ty_TorchIdentical().nn,
        nn.ReLU             : ty_TorchIdentical().nn,
        nn.Sigmoid          : ty_TorchIdentical().nn,
        nn.Tanh             : ty_TorchIdentical().nn,
        nn.Hardtanh         : ty_TorchIdentical().nn,
        nn.GELU             : ty_TorchIdentical().nn,


        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-other

        # https://pytorch.org/docs/stable/nn.html#normalization-layers
        nn.BatchNorm1d      : ty_TorchBatchNorm(dim=1).nn,
        nn.BatchNorm2d      : ty_TorchBatchNorm(dim=2).nn,
        nn.BatchNorm3d      : ty_TorchBatchNorm(dim=3).nn,
        nn.InstanceNorm1d   : ty_TorchInstanceNorm(dim=1).nn,
        nn.InstanceNorm2d   : ty_TorchInstanceNorm(dim=2).nn,
        nn.InstanceNorm3d   : ty_TorchInstanceNorm(dim=3).nn,

        # https://pytorch.org/docs/stable/nn.html#recurrent-layers
        nn.LSTMCell         : ty_TorchLSTMCell().nn,

        # https://pytorch.org/docs/stable/nn.html#linear-layers
        nn.Linear           : ty_TorchLinear().nn,

        # https://pytorch.org/docs/stable/nn.html#dropout-layers
        nn.Dropout          : ty_TorchIdentical().nn,
        nn.Dropout2d        : ty_TorchIdentical(ndim_min=1).nn,
        nn.Dropout3d        : ty_TorchIdentical(ndim_min=1).nn,
        nn.AlphaDropout     : ty_TorchIdentical().nn,

        # https://pytorch.org/docs/stable/nn.html#sparse-layers
        nn.Embedding        : ty_TorchEmbed().nn,

        # https://pytorch.org/docs/stable/nn.html#loss-functions
        nn.CrossEntropyLoss : ty_TorchNNCrossEntropyLoss().nn,

        # https://pytorch.org/docs/stable/nn.html#vision-layers
        nn.PixelShuffle     : ty_TorchPixelShuffle().nn,

        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        nn.LayerNorm        : ty_TorchIdentical().nn,

        }
