#!D:\project\transfer_xbox\python\tools\python.exe
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
sys.path.insert(1, os.path.abspath("./src/python"))
import time
import struct
import ctypes
import logging
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from nnf.executor import Executor
from nnf.data_format import cast_pytorch_tensor, cast_numpy_array
import data_loader
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level="INFO")
logger = logging.getLogger(__name__)

def test_executor():
    executor = Executor(r"..\nnfusion_rt\dxcompute_codegen\Direct3DWinNN\x64\Debug")
    inputs = [
        torch.ones([154587], dtype=torch.float32), 
        torch.ones([1001], dtype=torch.float32)
        ]
    executor(inputs)
    print(inputs[1])

def load_constants(src_file, size, type):
    if type == "float":
        c_type = ctypes.c_float
        pt_type = torch.float32
        fmt_str = f"{size}f"
    elif type == "int":
        c_type = ctypes.c_int32
        pt_type = torch.int32
        fmt_str = f"{size}i"
    else:
        raise Exception(f"Unsupported type {type}")

    with open(src_file, "rb") as f:
        raw = f.read(size * int(ctypes.sizeof(c_type)))
        data = struct.unpack(fmt_str, raw)
        tensor = torch.Tensor(data).to(pt_type)
    return tensor

def train():

    executor = Executor(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\x64\Release")
    
    # Parameter_0_0_host = torch.ones([200704], dtype=torch.float32)
    # Parameter_1_0_host = torch.ones([256], dtype=torch.float32)
    # Parameter_2_0_host = torch.ones([65536], dtype=torch.float32)
    # Parameter_3_0_host = torch.ones([256], dtype=torch.float32)
    # Parameter_4_0_host = torch.ones([2560], dtype=torch.float32)
    # Parameter_5_0_host = torch.ones([10], dtype=torch.float32)
    Parameter_0_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_0_0", 200704, "float")
    Parameter_1_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_3_0", 256, "float")
    Parameter_2_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_1_0", 65536, "float")
    Parameter_3_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_4_0", 256, "float")
    Parameter_4_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_2_0", 2560, "float")
    Parameter_5_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_5_0", 10, "float")

    Parameter_0_0_host /= 10
    Parameter_1_0_host /= 10
    Parameter_2_0_host /= 10
    Parameter_3_0_host /= 10
    Parameter_4_0_host /= 10
    Parameter_5_0_host /= 10

    Parameter_20_0_host = torch.ones([100], dtype=torch.int32)
    Parameter_21_0_host = torch.ones([78400], dtype=torch.float32)

    Result_81_0_host = torch.ones([1], dtype=torch.float32)
    Result_82_0_host = torch.ones([10], dtype=torch.float32)
    Result_83_0_host = torch.ones([2560], dtype=torch.float32)
    Result_84_0_host = torch.ones([256], dtype=torch.float32)
    Result_85_0_host = torch.ones([65536], dtype=torch.float32)
    Result_86_0_host = torch.ones([256], dtype=torch.float32)
    Result_87_0_host = torch.ones([200704], dtype=torch.float32)

    inputs = [
        Parameter_0_0_host,
        Parameter_1_0_host,
        Parameter_2_0_host,
        Parameter_3_0_host,
        Parameter_4_0_host,
        Parameter_5_0_host,
        Parameter_20_0_host,
        Parameter_21_0_host,
        Result_81_0_host,
        Result_82_0_host,
        Result_83_0_host,
        Result_84_0_host,
        Result_85_0_host,
        Result_86_0_host,
        Result_87_0_host
        ]

    model = MLP()
    loss_func = F.nll_loss
    model = ModelWithLoss(model, loss_func)
    inputs = []
    copy_params = [param.t().detach().clone().contiguous() for param in model.parameters()]
    inputs.extend(copy_params)
    inputs.extend(
        [
            Parameter_20_0_host,
            Parameter_21_0_host,
            Result_81_0_host,
            Result_82_0_host,
            Result_83_0_host,
            Result_84_0_host,
            Result_85_0_host,
            Result_86_0_host,
            Result_87_0_host
        ]
    )
    

    train_loader, _ = data_loader.get_mnist_dataloader(batch_size=100, shuffle=False)
    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    display_interval = 100
    start = time.time()
    for epoch in range(100):
        sum_nnf_loss = 0
        sum_pt_loss = 0
        for i, batch in enumerate(train_loader):
            # nnf training
            inputs[7], inputs[6] = batch[0], batch[1].int()
            executor(inputs)
            sum_nnf_loss += Result_81_0_host

            # pt training
            optim.zero_grad()
            pt_loss = model(*batch)
            pt_loss.backward()
            optim.step()
            sum_pt_loss += pt_loss
            
            # print(f"Iter {iter}, Batch {i}, NNF Loss {Result_81_0_host}, PT Loss {pt_loss}")
            # if i == 0:
            #     sys.exit(0)
            
            if (i + 1) % display_interval == 0:
                logger.info(f"Epoch {epoch}, Batch {i+1}, NNF loss {sum_nnf_loss/display_interval}, PyTorch loss {sum_pt_loss/display_interval}")
                sum_nnf_loss = 0
                sum_pt_loss = 0
    print(f"{time.time()-start}s")


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        # print("Input: ", torch.flatten(x)[:10])
        # print("Layer1 weight: ", torch.flatten(self.fc1.weight.t())[:10])
        # print("Layer1 bias: ", torch.flatten(self.fc1.bias.t())[:10])
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        output = F.log_softmax(x, dim=1)
        # print("Layer1 output:", torch.flatten(output)[:10])
        return output

class ModelWithLoss(nn.Module):
    def __init__(self, model, loss_func):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss_func = loss_func

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss_func(output, target)
        return loss

def train_with_faked_data():
    executor = Executor(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\x64\Release")
    
    # Parameter_0_0_host = torch.ones([200704], dtype=torch.float32)
    # Parameter_1_0_host = torch.ones([256], dtype=torch.float32)
    # Parameter_2_0_host = torch.ones([65536], dtype=torch.float32)
    # Parameter_3_0_host = torch.ones([256], dtype=torch.float32)
    # Parameter_4_0_host = torch.ones([2560], dtype=torch.float32)
    # Parameter_5_0_host = torch.ones([10], dtype=torch.float32)
    Parameter_0_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_0_0", 200704, "float")
    Parameter_1_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_3_0", 256, "float")
    Parameter_2_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_1_0", 65536, "float")
    Parameter_3_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_4_0", 256, "float")
    Parameter_4_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_2_0", 2560, "float")
    Parameter_5_0_host = load_constants(r"..\nnfusion_rt_mlp\dxcompute_codegen\Direct3DWinNN\Constant\Constant_5_0", 10, "float")

    Parameter_0_0_host /= 10
    Parameter_1_0_host /= 10
    Parameter_2_0_host /= 10
    Parameter_3_0_host /= 10
    Parameter_4_0_host /= 10
    Parameter_5_0_host /= 10

    Parameter_20_0_host = torch.ones([100], dtype=torch.int32)
    Parameter_21_0_host = torch.ones([78400], dtype=torch.float32)

    Result_81_0_host = torch.ones([1], dtype=torch.float32)
    Result_82_0_host = torch.ones([10], dtype=torch.float32)
    Result_83_0_host = torch.ones([2560], dtype=torch.float32)
    Result_84_0_host = torch.ones([256], dtype=torch.float32)
    Result_85_0_host = torch.ones([65536], dtype=torch.float32)
    Result_86_0_host = torch.ones([256], dtype=torch.float32)
    Result_87_0_host = torch.ones([200704], dtype=torch.float32)

    inputs = [
        Parameter_0_0_host,
        Parameter_1_0_host,
        Parameter_2_0_host,
        Parameter_3_0_host,
        Parameter_4_0_host,
        Parameter_5_0_host,
        Parameter_20_0_host,
        Parameter_21_0_host,
        Result_81_0_host,
        Result_82_0_host,
        Result_83_0_host,
        Result_84_0_host,
        Result_85_0_host,
        Result_86_0_host,
        Result_87_0_host
        ] 

    display_interval = 100
    start = time.time()
    for epoch in range(100):
        sum_nnf_loss = 0
        sum_pt_loss = 0
        for i in range(600):
            # nnf training
            executor(inputs)
            sum_nnf_loss += Result_81_0_host
            
            if (i + 1) % display_interval == 0:
                logger.info(f"Epoch {epoch}, Batch {i+1}, NNF loss {sum_nnf_loss/display_interval}, PyTorch loss {sum_pt_loss/display_interval}")
                sum_nnf_loss = 0
                sum_pt_loss = 0
        time.sleep(10)
    print(f"{time.time()-start}s")


def inference():
    nnf_model_path = r"D:\project\wsl_codegen\nnfusion_rt\dxcompute_codegen\Direct3DWinNN\build"
    executor = Executor(nnf_model_path)
    input_name = executor.get_inputs()[0].name
    output_name = executor.get_outputs()[0].name

    batch_size = 5
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=batch_size, shuffle=True)
    for i, batch in enumerate(train_loader):
        data_desc = cast_pytorch_tensor(batch[0])
        
        out = torch.zeros([batch_size, 10])
        out_desc = cast_pytorch_tensor(out)

        out2 = np.zeros([batch_size, 10], dtype=np.float32)
        print(out2.nbytes)
        out2_desc = cast_numpy_array(out2)
        
        executor({input_name: data_desc}, {output_name: out_desc})
        executor({input_name: data_desc}, {output_name: out2_desc})
        print(out)
        print(out2)
        if i == 5:
            sys.exit(0)



if __name__ == "__main__":
    # test_executor()
    #train()
    # pytorch_train()
    # train_with_faked_data()
    inference()
