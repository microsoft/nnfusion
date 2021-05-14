# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
os.environ["PATH"] = os.path.abspath(
    "./build/src/tools/nnfusion") + ":" + os.environ["PATH"]
import time
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import json
import tempfile

from nnfusion.data_format import cast_numpy_array, cast_pytorch_tensor
from nnfusion.executor import Executor
from nnfusion.session import PTSession as Session, generate_sample
from nnfusion.runner import PTRunner as Runner
from nnfusion.description import IODescription
from nnfusion.trainer import PTTrainer as Trainer
from nnfusion.utils import cd, execute
import data_loader


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output


def test_executor():
    def create_mnist_model(workdir, model, sample_input):

        sample_inputs = [sample_input]
        torch.onnx.export(model,
                          tuple(sample_inputs),
                          os.path.join(workdir, "nnf.onnx"),
                          input_names=["data"],
                          output_names=["output"],
                          opset_version=12,
                          _retain_param_name=True)
        from nnfusion.session import codegen, modify_nnfusion_rt, build
        codegen(os.path.join(workdir, "nnf.onnx"),
                "-f onnx -fextern_result_memory=1", workdir)
        rt_dir = os.path.join(workdir, "nnfusion_rt/cuda_codegen")
        modify_nnfusion_rt(rt_dir)
        build(rt_dir)

    # Execute by PyTorch
    batch_size = 5
    device = "cuda:0"
    sample_input = torch.ones((batch_size, 1, 28, 28))
    model = MLP()
    pt_out = model(sample_input)

    # Prepare NNFusion model
    dir_ctx = tempfile.TemporaryDirectory(prefix="nnf_")
    workdir = dir_ctx.name
    create_mnist_model(workdir, model.to(device), sample_input.to(device))

    # Execute by NNFusion executor
    rt_dir = os.path.join(workdir, "nnfusion_rt/cuda_codegen")
    executor = Executor(rt_dir)
    input_name = executor.get_inputs()[0].name
    output_name = executor.get_outputs()[0].name
    data_desc = cast_pytorch_tensor(sample_input.cuda())
    nnf_out = torch.ones([batch_size, 10]).cuda()
    nnf_out_desc = cast_pytorch_tensor(nnf_out)
    executor({input_name: data_desc}, {output_name: nnf_out_desc})

    # Results from PyTorch and NNFusion should be equal
    assert torch.allclose(pt_out, nnf_out.cpu())


def test_session():
    model = MLP()
    batch_size = 5
    input_desc = [
        IODescription("data", [batch_size, 1, 28, 28], "float32"),
    ]
    device = "cuda:0"

    inputs = {
        desc.name: generate_sample(input_desc[0], device)
        for desc in input_desc
    }
    session = Session(model, input_desc, device)
    print(session(inputs))


def test_runner():
    model = MLP()
    tensor1 = torch.ones([5, 1, 28, 28], dtype=torch.float32, device="cuda:0")
    tensor2 = torch.ones([3, 1, 28, 28], dtype=torch.float32, device="cuda:0")
    tensor3 = torch.ones([5, 1, 28, 28], dtype=torch.float32, device="cuda:0")
    nnf_flags = {"training_mode": 1}
    runner = Runner(model, codegen_flags=nnf_flags)
    point1 = time.time()
    out1 = runner(tensor1)[0].cpu().numpy()
    point2 = time.time()
    runner(tensor2)
    point3 = time.time()
    out3 = runner(tensor3)[0].cpu().numpy()
    point4 = time.time()
    print("Duration1: {}s".format(point2 - point1))
    print("Duration2: {}s".format(point3 - point2))
    print("Duration3: {}s".format(point4 - point3))
    assert np.allclose(out1, out3)


def train_mnist():
    model = MLP()
    loss_func = F.nll_loss
    device = "cuda:0"
    batch_size = 5

    codegen_flags = {
        "autodiff":
        True,  # add backward graph
        "training_mode":
        True,  # move weight external
        "extern_result_memory":
        True,  # move result external
        "training_optimizer":
        '\'' + json.dumps({
            "optimizer": "SGD",
            "learning_rate": 0.001
        }) + '\'',  # training optimizer configs
    }

    trainer = Trainer(model,
                      loss_func,
                      device=device,
                      workdir="./tmp",
                      codegen_flags=codegen_flags)
    train_loader, _ = data_loader.get_mnist_dataloader(batch_size=batch_size,
                                                       shuffle=False)
    print("feeding")
    i = 0
    for i, batch in enumerate(train_loader):
        feed_data = [t.to(device) for t in batch]

        pytorch_loss = trainer.run_by_pytorch(*feed_data)
        nnf_loss = trainer(*feed_data)
        if i % 100 == 0:
            print("iter ", i)
            print('pytorch_loss: ', pytorch_loss)
            print('nnf_loss: ', nnf_loss)

        if i == 10000:
            break

    torch.save(model.state_dict(), "/tmp/mnist.pt")


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def eval():
    device = "cuda:0"
    model = MLP()
    state_dict = torch.load("/tmp/mnist.pt")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _, test_loader = data_loader.get_mnist_dataloader(batch_size=1,
                                                      shuffle=False)

    res = []
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            pred = model(batch[0].to(device))
            prob = np_softmax(pred.cpu().numpy())
            print('Image {} is digit "{}", confidence {}'.format(
                i, np.argmax(prob), np.max(prob)))

        res.append(np.argmax(prob))
        if i == 5:
            break
    assert tuple(res) == (7, 2, 1, 0, 4, 1)


if __name__ == "__main__":
    test_executor()
    test_session()
    test_runner()
    train_mnist()
    eval()