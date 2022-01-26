import torch
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)

import time


# grid size
N = 1024 * 2
x = torch.linspace(0, 1, N)
y = torch.linspace(0, 1, N)
x, y = torch.meshgrid(x, y)
# dxdy = (1. / (N-1)) ** 2

# stencil computing as a convolution
filter = torch.tensor(
    [[0.,  1., 0.],
     [1., -4., 1.],
     [0.,  1., 0.]]
).view(1, 1, 3, 3).cuda()

# known function rho
rho = F.conv2d(torch.ones((1, 1, N, N)).cuda(), filter, padding=1)
# random initial guess for function phi
# phi = torch.randn((1, 1, N, N)).cuda()
phi = torch.zeros((1, 1, N, N)).cuda()

# conjugate gradient iterations
# padding=1 for zero Dirichlet boundary condition
torch.cuda.synchronize()
start = time.time()
r0 = rho - F.conv2d(phi, filter, padding=1)
p = r0
counter = 0

# ============== expected usage 1 ===========
# import nnfusion
# while True:
#     counter += 1
#     with nnfusion.jit():
#         conv_out = F.conv2d(p, filter, padding=1)
#         alpha = torch.mul(r0, r0).sum() / \
#             torch.mul(p, conv_out).sum()
#         phi += alpha * p
#         r1 = r0 - alpha * conv_out

# ============== expected usage 2 ===========
# @nnfusion.jit
# def poisson(r0):
#     p = r0
#     conv_out = F.conv2d(p, self.filter, padding=1)
#     alpha = torch.mul(r0, r0).sum() \
#         / torch.mul(p, conv_out).sum()
#     phi = alpha * p
#     r1 = r0 - alpha * conv_out
#     r1_sum = torch.mul(r1, r1).sum()
#     beta = r1_sum / torch.mul(r0, r0).sum()
#     p = r1 + beta * p
#     return r1_sum, phi, p, r1

while True:
    counter += 1
    conv_out = F.conv2d(p, filter, padding=1)
    alpha = torch.mul(r0, r0).sum() / \
        torch.mul(p, conv_out).sum()
    phi += alpha * p
    r1 = r0 - alpha * conv_out

    # exit loop if converged
    r1_sum = torch.mul(r1, r1).sum()
    if counter % 100 == 0:
        print('iters:\t', counter)
        print('rnorm:\t', torch.sqrt(r1_sum))
    if torch.sqrt(r1_sum) < 1e-10:
        # print grid-wise residual
        print('**************** Converged ****************')
        print('iters:\t', counter)
        torch.cuda.synchronize()
        print('time:\t', time.time() - start)
        print('error:\t', torch.norm(phi - torch.ones((1, 1, N, N)).cuda()))
        break

    beta = r1_sum / torch.mul(r0, r0).sum()
    p = r1 + beta * p
    r0 = r1
