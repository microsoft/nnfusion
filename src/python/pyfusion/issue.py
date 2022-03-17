import torch
import os
import sys
os.environ["PATH"] = os.path.abspath("/data/zimiao/project/NNFusion_github/build/src/tools/nnfusion/") + ":" + os.environ["PATH"]

from nnfusion.runtime import NNFusionRT

# torch.set_default_tensor_type(torch.DoubleTensor)


if __name__ == "__main__":
    N = 1024

    torch.cuda.set_device(0)

    input = torch.randn(1, N, device="cuda")
    output = torch.randn(1, N, device="cuda")
    print(output[:5])
    # model = torch.nn.Identity()
    model = torch.nn.Linear(N, N).cuda()  # Fails too

    nnf = NNFusionRT(model, (input, ), [output])
    nnf.compile(buildall=True, rebuild=True)
    nnf.run([input], [output])
    print(output[:5])