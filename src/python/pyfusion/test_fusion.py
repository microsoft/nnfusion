import torch
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)

from nnfusion.runtime import NNFusionRT

class Stencil(torch.nn.Module):
    def __init__(self):
        super(Stencil, self).__init__()
        self.filter = torch.tensor(
            [[0.,  1., 0.],
            [1., -4., 1.],
            [0.,  1., 0.]], device="cuda"
        ).view(1, 1, 3, 3)
    def forward(self, X):
        X = F.conv2d(X, self.filter, padding=1)
        return X

class FusedOp(torch.nn.Module):
    def forward(self, r0, alpha, conv):
        r1 = r0 - alpha * conv
        r1_sum = torch.mul(r1, r1)
        return r1_sum

class FusedMulSubMulOp(torch.nn.Module):
    def forward(self, alpha, r0, conv):
        r1 = r0 - alpha * conv
        r1_sum = torch.mul(r1, r1)
        return r1_sum

class FusedMulAddOp(torch.nn.Module):
    def forward(self, alpha, a, b):
        return alpha * a + b

class FusedStencilGraph(torch.nn.Module):
    def __init__(self):
        super(FusedStencilGraph, self).__init__()
        self.filter = torch.tensor(
            [[0.,  1., 0.],
            [1., -4., 1.],
            [0.,  1., 0.]], device="cuda"
        ).view(1, 1, 3, 3)
    def forward(self, r0):
        p = r0
        conv_out = F.conv2d(p, self.filter, padding=1)
        alpha = torch.mul(r0, r0).sum() / torch.mul(p, conv_out).sum()
        phi = alpha * p
        r1 = r0 - alpha * conv_out
        r1_sum = torch.mul(r1, r1).sum()
        beta = r1_sum / torch.mul(r0, r0).sum()
        p = r1 + beta * p
        return r1_sum, phi, p, r1


if __name__ == "__main__":
    M = 1024 * 4
    N = 1024 * 8

    torch.cuda.set_device(1)
    
    X = torch.randn(1, 1, M, N, device="cuda")
    
    alpha = torch.randn([1], device="cuda")
    A = torch.randn(1, 1, M, N, device="cuda")
    B = torch.randn(1, 1, M, N, device="cuda")

    stencil = Stencil()
    fused_op = FusedOp()
    fused_op = FusedMulSubMulOp()
    fused_op = FusedMulAddOp()
    fused_graph = FusedStencilGraph()

    #nnf = NNFusionRT(stencil, X, server = "52.166.15.182:8880", steps=1000)
    #nnf = NNFusionRT(fused_op, (alpha, A, B), server = "127.0.0.1:8880", steps=0)
    nnf = NNFusionRT(fused_graph, (A), (alpha, A, A, B), server = "127.0.0.1:8880", steps=2000)
    
    nnf.compile(buildall=True, rebuild=True)
    #exit()

    STEP = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    r0 = A
    phi = B

    r1_sum_n = torch.randn([], device="cuda")
    phi_n = torch.randn(1, 1, M, N, device="cuda")
    p_n = torch.randn(1, 1, M, N, device="cuda")
    r1_n = torch.randn(1, 1, M, N, device="cuda")

    # NNFusion evaluation
    nnf.run([r0], [r1_sum_n, phi_n, p_n, r1_n])
    start.record()
    for i in range(STEP):
        nnf.run([r0], [r1_sum_n, phi_n, p_n, r1_n])
    end.record()
    torch.cuda.synchronize()
    print("NNFusion step time:", start.elapsed_time(end) / STEP, "ms")


    # PyTorch evaluation
    r1_sum_p, phi_p, p_p, r1_p = fused_graph(r0, phi)
    start.record()
    for i in range(STEP):
        r1_sum_p, phi_p, p_p, r1_p = fused_graph(r0, phi)
    end.record()
    torch.cuda.synchronize()
    print("PyTorch step time:", start.elapsed_time(end) / STEP, "ms")

    assert torch.allclose(r1_sum_p.cpu(), r1_sum_n.cpu())
    assert torch.allclose(phi_p.cpu(), phi_n.cpu())
    assert torch.allclose(p_p.cpu(), p_n.cpu())
    assert torch.allclose(r1_p.cpu(), r1_n.cpu())
