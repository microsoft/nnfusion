import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from custom_op import CustomOp

class CustomLinear(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)

    def reset_parameters(self):
        self.fc2.reset_parameters()

    def forward(self, x):
        x = F.gelu(x.float()).type_as(x)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x
    
M_SQRT1_2 = 0.70710678118654752440
M_2_SQRTPI = 1.12837916709551257390

class FusedLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, p):
        mask = (torch.rand_like(x) >= p)
#         fused_op = CustomOp(ir=f'''
# output0[N0, N2] +=! input0[N0, N1] * input1[N2, N1];
# ''', input_orders={'input0': x, 'input1': weight}, tags="tensorCoreConfig=(0, 1)", device=device)

        fused_op = CustomOp(ir=f'''
m0[N0, N1] = input0[N0, N1].cast(`float32`); 
m1[N0, N1] = m0[N0, N1] * const(0.5).cast(`float32`) * (const(1.0).cast(`float32`) + (m0[N0, N1] * const({M_SQRT1_2}).cast(`float32`)).call(`erf`)); 
m2[N0, N1] = m1[N0, N1].cast(`float16`); 
m3[N0, N1] = m2[N0, N1] * input3[N0, N1] / const({1-p}).cast(`float16`); 
m4[N0, N2] +=! m3[N0, N1] * input1[N2, N1];
output0[N0, N2] = m4[N0, N2] + input2[N0];
''', input_orders={'input0': x, 'input1': weight, 'input2': bias, 'input3': mask}, tags="tensorCoreConfig=(0, 1)", device=device)
        y = fused_op([x, weight, bias, mask])
        ctx.save_for_backward(x, weight, mask)
        ctx.p = p
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, mask = ctx.saved_tensors
        p = ctx.p
        dbias = torch.sum(dy, dim=0)
        dw_op = CustomOp(ir=f'''
m0[N0, N1] = input0[N0, N1].cast(`float32`); 
m1[N0, N1] = m0[N0, N1] * const(0.5).cast(`float32`) * (const(1.0).cast(`float32`) + (m0[N0, N1] * const({M_SQRT1_2}).cast(`float32`)).call(`erf`)); 
m2[N0, N1] = m1[N0, N1].cast(`float16`); 
m3[N0, N1] = m2[N0, N1] * input2[N0, N1] / const({1-p}).cast(`float16`); 
output0[N2, N1] +=! input1[N0, N2] * m3[N0, N1];
''', input_orders={'input0': x, 'input1': dy, 'input2': mask}, tags="tensorCoreConfig=(0, 1)", device=device)
        dw = dw_op([x, dy, mask])

        dx_op = CustomOp(ir=f'''
m0[N0, N1] +=! input3[N0, N2] * input1[N2, N1];
m1[N0, N1] = m0[N0, N1] * input2[N0, N1] * const({1-p}).cast(`float16`);
m2[N0, N1] = m1[N0, N1].cast(`float32`); 
m3[N0, N1] = const(0.5).cast(`float32`) * (const(1.0).cast(`float32`) + (input0[N0, N1] * const({M_SQRT1_2}).cast(`float32`)).call(`erf`));
m4[N0, N1] = (const(-0.5).cast(`float32`) * input0[N0, N1] * input0[N0, N1]).call(`exp`) * const({M_2_SQRTPI * M_SQRT1_2 * 0.5}).cast(`float32`);
output0[N0, N1] = m2[N0, N1] * (m3[N0, N1] + input0[N0, N1] * m4[N0, N1]);
''', input_orders={'input0': x, 'input1': weight, 'input2': mask, 'input3': dy}, tags="tensorCoreConfig=(0, 1)", device=device)
        dx = dx_op([x, weight, mask, dy])
        return dx, dw, dbias, None

class FusedCustomLinear(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_dropout = activation_dropout
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, dtype=torch.float16)

    def reset_parameters(self):
        self.fc2.reset_parameters()

    def forward(self, x):
        return FusedLinearFunc.apply(x, self.fc2.weight, self.fc2.bias, self.activation_dropout)
    

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float16)
    x = torch.randn(2048, 16384, requires_grad = True, device=device)
    ref = CustomLinear(4096, 16384, 0).to(device)
    fused = FusedCustomLinear(4096, 16384, 0).to(device)
    
    y_ref = ref(x)
    y_fused = fused(x)

    y_grad = torch.ones_like(y, device=device)
    y.backward(y_grad)

    # start = time.time()
    # for i in range(100):
    #     y = layer.forward(x)
    #     y.backward(y_grad)
    #     #print(x, x.grad, layer.fc2.weight.grad, layer.fc2.bias.grad)
    # end = time.time()
    # print(end-start)
    
    



