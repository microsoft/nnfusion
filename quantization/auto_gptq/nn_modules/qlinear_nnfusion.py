import math
import numpy as np
import torch
import torch.nn as nn
from .tvm_untils import cache
from torch.onnx import (
    _constants,
    _type_utils,
    errors,
    symbolic_helper,
    symbolic_opset9 as opset9,
)

class QuantLinearFunction(torch.autograd.Function):  
    @staticmethod  
    def forward(ctx, x, qweight, scales, zeros):  
        outfeatures = qweight.shape[0]
        outshape = x.shape[:-1] + (outfeatures,)
        y = torch.zeros(outshape, dtype=x.dtype, device=x.device)
        return y


    @staticmethod
    def symbolic(g, x, qweight, scales, zeros):
        # get x.dtype
        dtype = x.type().scalarType()
        # get output shape
        # get x.shape
        ret = g.op(
            f'nnfusion::QuantLinear',
            x,
            qweight,
            scales,
            zeros,
            ).setType(x.type())

        return ret

# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class QuantLinear(nn.Module): 
    def __init__(
        self,
        bits,
        groupsize,
        infeatures,
        outfeatures,
        bias,
        ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.register_buffer(
            'qweight', torch.zeros((outfeatures, infeatures // 8 * 3), dtype=torch.int8)
        )
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('zeros', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        self.groupsize = groupsize if groupsize != -1 else infeatures

        if groupsize != -1:
            self.register_buffer(
                'g_idx',
                torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32)
            )
        # self.tvm_handler = cache.get_handler(n=outfeatures, k=infeatures, bits=bits, group_size=groupsize)
    
    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        # print("scales shape", scales.shape, zeros.shape, self.g_idx.shape)
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        self.zeros = scale_zeros.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (
                        W[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        qweight = np.ascontiguousarray(qweight.T)
        qweight = qweight.view(dtype=np.int8)
        self.qweight = torch.from_numpy(qweight) 


    def forward(self, x):
        y = QuantLinearFunction.apply(x, self.qweight, self.scales, self.zeros)
        y = y + self.bias if self.bias is not None else y 
        return y
    