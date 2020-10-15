from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils import cd
import shutil
import json
import numpy as np
import nnf
import dataprc
from description import IODescription, ModelDescription, generate_sample


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


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)
        return loss


def convert_model_to_onnx(model, model_desc, device, file_name):
    model.to(device)

    input_names = [input.name_ for input in model_desc.inputs_]
    output_names = [output.name_ for output in model_desc.outputs_]

    sample_inputs = []
    for input_desc in model_desc.inputs_:
        input_sample = generate_sample(input_desc, device)
        sample_inputs.append(input_sample)

    sample_outputs = []
    for output_desc in model_desc.outputs_:
        output_sample = generate_sample(output_desc, device)
        sample_outputs.append(output_sample)

    torch.onnx.export(model,
                      tuple(sample_inputs),
                      file_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=12,
                      _retain_param_name=True,
                      example_outputs=tuple(sample_outputs),
                      do_constant_folding=False)

    return model


def detect_nnfusion():
    return shutil.which("nnfusion")


def nnfusion_codegen(nnfusion_path, model_path, flags, output_dir):
    nnfusion_path = os.path.abspath(nnfusion_path)
    model_path = os.path.abspath(model_path)
    with cd(output_dir):
        command = ' '.join([nnfusion_path, model_path, flags])
        assert os.system(command) == 0


def modify_nnfusion_rt(rt_dir):
    with cd(rt_dir):
        # static -> shared library
        command = "sed -i '/cuda_add_library(${TARGET_NAME} ${SRC})/s/(${TARGET_NAME} ${SRC})/(${TARGET_NAME} SHARED ${SRC})/g'" + " " + "CMakeLists.txt"
        assert os.system(command) == 0

        # remove culibos
        command = "sed -i '/target_link_libraries(${TARGET_NAME} cudnn culibos cublas)/s/culibos//g'" + " " + "CMakeLists.txt"
        assert os.system(command) == 0

        # remove cudaDevice reset in cuda_init()
        command = "sed -i '/cudaDeviceReset()/s:^://:'" + " " + "nnfusion_rt.cu"
        assert os.system(command) == 0

        # early return in Debug()


def compile_nnfusion_rt(rt_dir):
    with cd(rt_dir):
        command = "cmake ."
        assert os.system(command) == 0

        command = "make -j"
        assert os.system(command) == 0


def parse_nnf_params(param_file):
    with open(param_file) as f:
        nnf_params = json.load(f)

    weights = {}
    for name, desc in nnf_params["weight"].items():
        index = desc["id"].split("inputs[")[1].split("]")[0]
        dtype = desc["id"][2:].split("*")[0]
        shape = desc["shape"]
        weights[name] = {
            "name": name,
            "id": index,
            "dtype": dtype,
            "shape": shape,
            "raw_id": desc["id"],
            "nnf_name": desc["name"]
        }

    inputs = {}
    for name, desc in nnf_params["input"].items():
        index = desc["id"].split("inputs[")[1].split("]")[0]
        dtype = desc["id"][2:].split("*")[0]
        shape = desc["shape"]
        inputs[name] = {
            "name": name,
            "id": index,
            "dtype": dtype,
            "shape": shape,
            "raw_id": desc["id"],
            "nnf_name": desc["name"]
        }

    outputs = {}
    for name, desc in nnf_params["output"].items():
        index = desc["id"].split("outputs[")[1].split("]")[0]
        dtype = desc["id"][2:].split("*")[0]
        shape = desc["shape"]
        outputs[name] = {
            "name": name,
            "id": index,
            "dtype": dtype,
            "shape": shape,
            "raw_id": desc["id"],
            "nnf_name": desc["name"]
        }

    return weights, inputs, outputs


def str2dtype(ss):
    if ss == "float":
        return torch.float
    elif ss == "int64_t":
        return torch.int64
    else:
        assert False, "unsupported type: {}".format(ss)


def validate_nnf_weights(nnf_weights, torch_weights):
    assert set(nnf_weights.keys()) == set(torch_weights.keys())
    for name, desc in nnf_weights.items():
        torch_tensor = torch_weights[name]
        assert desc["shape"] == list(torch_tensor.shape)
        assert str2dtype(desc["dtype"]) == torch_tensor.dtype


def validate_nnf_inputs(nnf_inputs, inputs_desc):
    assert set(nnf_inputs.keys()) == {desc.name_ for desc in inputs_desc}
    for desc in inputs_desc:
        assert desc.shape_ == nnf_inputs[desc.name_]["shape"]
        assert desc.dtype_ == str2dtype(nnf_inputs[desc.name_]["dtype"])


def validate_nnf_outputs(nnf_outputs, outputs_desc):
    for desc in outputs_desc:
        assert desc.shape_ == nnf_outputs[desc.name_]["shape"]
        assert desc.dtype_ == str2dtype(nnf_outputs[desc.name_]["dtype"])


class Trainer(object):
    def __init__(self,
                 model,
                 loss,
                 input_desc,
                 device="cuda:0",
                 nnfusion_path=None,
                 workdir='./tmp'):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.model_with_loss = ModelWithLoss(self.model, self.loss).to(device)
        self.torch_weights = {
            name: param
            for name, param in self.model_with_loss.named_parameters()
        }
        self.input_desc = input_desc
        self.output_desc = [
            IODescription("loss", [], torch.float32),
        ]
        self.model_desc = ModelDescription(self.input_desc, self.output_desc)
        self.device = device
        self.workdir = workdir
        if os.path.exists(os.path.join(workdir, "nnfusion_rt")):
            shutil.rmtree(os.path.join(workdir, "nnfusion_rt"))
        if not os.path.exists(workdir):
            os.mkdir(workdir)
            
        ## convert torch model to onnx
        self.onnx_model = os.path.join(self.workdir, "nnf.onnx")
        convert_model_to_onnx(self.model_with_loss, self.model_desc,
                              self.device, self.onnx_model)

        if nnfusion_path:
            self.nnfusion_path = nnfusion_path
        else:
            self.nnfusion_path = detect_nnfusion()
        self.nnfusion_path = os.path.abspath(nnfusion_path)

        ## codegen
        codegen_flags = "-f onnx -fautodiff -ftraining_mode -fextern_result_memory=True"
        nnfusion_codegen(self.nnfusion_path, self.onnx_model, codegen_flags,
                         self.workdir)
        rt_dir = os.path.join(self.workdir, "nnfusion_rt/cuda_codegen")
        modify_nnfusion_rt(rt_dir)
        compile_nnfusion_rt(rt_dir)
        param_file = os.path.join(rt_dir, "para_info.json")

        ## validate nnf_rt
        nnf_weights, nnf_inputs, nnf_outputs = parse_nnf_params(param_file)
        validate_nnf_weights(nnf_weights, self.torch_weights)
        validate_nnf_inputs(nnf_inputs, self.input_desc)
        validate_nnf_outputs(nnf_outputs, self.output_desc)
        self.nnf_inputs = nnf_inputs
        self.loss_id = nnf_outputs[self.output_desc[0].name_]["id"]

        ## load nnf_rt
        LIB_NNF_RT = os.path.join(rt_dir, "libnnfusion_naive_rt.so")
        os.environ['LIB_NNF_RT'] = LIB_NNF_RT
        self.runtime = nnf.Runtime()
        with cd(os.path.join(rt_dir)):
            self.runtime.init()

        ## attach torch tensors
        self.feed_tensors = [None] * (len(nnf_weights) + len(nnf_inputs) +
                                      len(nnf_outputs))
        for name, desc in nnf_weights.items():
            self.feed_tensors[int(desc["id"])] = self.torch_weights[name]

        output_offset = len(nnf_weights) + len(nnf_inputs)
        for name, desc in nnf_outputs.items():
            self.feed_tensors[int(desc["id"]) + output_offset] = torch.ones(
                desc["shape"], dtype=str2dtype(desc["dtype"]), device=device)
        self.loss_id = int(
            nnf_outputs[self.output_desc[0].name_]["id"]) + output_offset

    def __del__(self):
        self.runtime.free()

    def run_by_pytorch(self, feed_data):
        args = [feed_data[desc.name_] for desc in self.input_desc]
        with torch.no_grad():
            loss = self.model_with_loss(*args)
        return loss

    def run_by_nnf(self, feed_data):
        for key, value in feed_data.items():
            index = int(self.nnf_inputs[key]["id"])
            self.feed_tensors[index] = value
        self.runtime.feed(tensors=self.feed_tensors)
        return self.feed_tensors[self.loss_id]


def train_mnist():
    model = MLP()
    loss = F.nll_loss
    batch_size = 5
    input_desc = [
        IODescription("data", [batch_size, 1, 28, 28], torch.float32),
        IODescription("target", [
            batch_size,
        ], torch.int64, num_classes=10),
    ]
    device = "cuda:0"
    workdir = "./tmp"
    nnfusion_path = "./build/src/tools/nnfusion/nnfusion"

    trainer = Trainer(model, loss, input_desc, device, nnfusion_path, workdir)
    data, _ = dataprc.get_mnist_dataloader(batch_size=batch_size)
    print("feeding")
    i = 0
    for i, batch in enumerate(data):
        batch = {"data": batch[0], "target": batch[1]}
        for k in batch:
            batch[k] = batch[k].to(device)

        pytorch_loss = trainer.run_by_pytorch(batch)
        nnf_loss = trainer.run_by_nnf(batch)
        if i % 100 == 0:
            print("iter ", i)
            print('pytorch_loss: ', trainer.run_by_pytorch(batch))
            print('nnf_loss: ', trainer.run_by_nnf(batch))

        if i == 10000:
            break

    torch.save(model.state_dict(), os.path.join(workdir, "mnist.pt"))


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def eval():
    device = "cuda:0"
    workdir = "./tmp"
    model = MLP()
    state_dict = torch.load(os.path.join(workdir, "mnist.pt"))
    # state_dict = {(k[6:] if k.startswith("model.") else k): v
    #               for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    data, _ = dataprc.get_mnist_dataloader(batch_size=1, shuffle=False)

    for i, batch in enumerate(data):
        with torch.no_grad():
            pred = model(batch[0].to(device))
            prob = np_softmax(pred.cpu().numpy())
            print('Image {} is digit "{}", confidence {}'.format(
                i, np.argmax(prob), np.max(prob)))

        if i == 5:
            break


if __name__ == "__main__":
    train_mnist()
    eval()
