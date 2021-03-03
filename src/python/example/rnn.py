# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from nnf.trainer import Trainer
from nnf.description import IODescription, generate_sample
from nnf.runner import Runner
from nnf.session import Session
import data_loader
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import time
import sys
import os
import argparse
import json
sys.path.insert(1, os.path.abspath("./src/python"))
torch.manual_seed(0)

os.environ["PATH"] = os.path.abspath(
    "./build/src/tools/nnfusion") + ":" + os.environ["PATH"]

parser = argparse.ArgumentParser()
parser.add_argument('--num_layer', default=4, type=int)
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epoch', default=1, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--learning_rate', default=0.05, type=float)
parser.add_argument('--backend', default="nnfusion", type=str)

args = parser.parse_args()


class LSTMCell(nn.Module):

    def __init__(self, hidden_size):
        super(LSTMCell, self).__init__()
        self.W = nn.ParameterList()
        self.U = nn.ParameterList()
        self.b = nn.ParameterList()
        self.num_unit = hidden_size
        for i in range(4):
            W = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(
                self.num_unit, self.num_unit, device=args.device)))
            U = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(
                self.num_unit, self.num_unit, device=args.device)))
            b = nn.Parameter(torch.zeros(self.num_unit, device=args.device))
            self.W.append(W)
            self.U.append(U)
            self.b.append(b)

    def forward(self, inputs, state):
        c, h = state

        i = torch.mm(inputs, self.W[0]) + torch.mm(h, self.U[0]) + self.b[0]
        j = torch.mm(inputs, self.W[1]) + torch.mm(h, self.U[1]) + self.b[1]
        f = torch.mm(inputs, self.W[2]) + torch.mm(h, self.U[2]) + self.b[2]
        o = torch.mm(inputs, self.W[3]) + torch.mm(h, self.U[3]) + self.b[3]

        new_c = (c * torch.sigmoid(f + 1.0) +
                 torch.sigmoid(i) * torch.tanh(j))
        new_h = torch.tanh(new_c) * torch.sigmoid(o)

        return new_h, (new_c, new_h)


class SimpleRNNCell(nn.Module):

    def __init__(self, hidden_size):
        super(SimpleRNNCell, self).__init__()
        self.num_unit = hidden_size
        self.W = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(
            self.num_unit, self.num_unit, device=args.device)))
        self.U = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(
            self.num_unit, self.num_unit, device=args.device)))

    def forward(self, inputs, state):
        c, h = state

        t1 = torch.mm(inputs, self.W)
        t2 = torch.mm(h, self.U)
        y = t1 + t2

        new_c = new_h = torch.tanh(y)

        return new_h, (new_c, new_h)


class StackedRNN(nn.Module):

    def __init__(self, num_layer, cell, hidden_size, batch_size, num_step):
        super(StackedRNN, self).__init__()
        self.num_layer = num_layer
        self.num_unit = hidden_size
        self.batch_size = batch_size
        self.num_step = num_step
        self.initial_state = (torch.zeros(self.batch_size, self.num_unit, device=args.device), torch.zeros(
            self.batch_size, self.num_unit, device=args.device))

        self.stacked_cells = torch.nn.ModuleList(
            [cell(self.num_unit) for layer in range(self.num_layer)])

    def forward(self, inputs):
        inputs = torch.unbind(inputs, dim=0)
        states = [self.initial_state for layer in range(self.num_layer)]

        for step in range(self.num_step):
            cur_input = inputs[step]
            for layer in range(self.num_layer):
                cell_output, states[layer] = self.stacked_cells[layer](
                    cur_input, states[layer])
                cur_input = cell_output

        output = cell_output
        final_state = states[-1]
        return output, final_state


class ImageClassification(nn.Module):
    def __init__(self, input_dimension, num_label, num_layer, cell, hidden_size, batch_size, num_step):
        super(ImageClassification, self).__init__()
        self.batch_size = batch_size
        self.fc_input = nn.Linear(input_dimension, hidden_size, False)
        self.rnn = StackedRNN(num_layer, cell,
                              hidden_size, batch_size, num_step)
        self.fc_output = nn.Linear(hidden_size, num_label, False)

    def forward(self, inputs):
        data = torch.reshape(inputs, (self.batch_size, 28, 28))
        data = data.permute(2, 0, 1)
        data = self.fc_input(data)
        data, rnn_states = self.rnn(data)
        output = self.fc_output(data)
        output = torch.nn.functional.log_softmax(output, dim=1)
        return output


def pytorch_train_rnn():
    backend = "PyTorch"
    if args.backend == "torchscript":
        backend += " TorchScript"
    print("Train RNN model on MNIST dataset with {}".format(backend))

    train_loader, test_loader = data_loader.get_mnist_dataloader(
        batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)

    model = ImageClassification(28, 10, args.num_layer, SimpleRNNCell,
                                args.hidden_size, args.batch_size, 28).to(args.device)
    if args.backend == "torchscript":
        model = torch.jit.trace(
            model, (torch.ones(args.batch_size, 1, 28, 28).to(args.device)))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_func = F.nll_loss

    sum_loss = 0
    sum_iter = 0
    sum_time = 0
    torch.cuda.synchronize()
    start_time = time.time()
    for epoch in range(args.num_epoch):
        for i, batch in enumerate(train_loader):
            if sum_iter == 100:
                print("Epoch {}, batch {}，loss {}, time {} s".format(
                    epoch, i, sum_loss / sum_iter, sum_time / sum_iter))
                sum_loss = 0
                sum_iter = 0
                sum_time = 0

            data, label = (t.to(args.device) for t in batch)

            predict = model(data)

            loss = loss_func(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            end_time = time.time()
            iter_time = end_time - start_time
            start_time = end_time

            sum_loss += loss
            sum_time += iter_time
            sum_iter += 1

    torch.save(model.state_dict(), "/tmp/rnn.pt")


def test_session():
    model = ImageClassification(28, 10, args.num_layer, SimpleRNNCell,
                                args.hidden_size, args.batch_size, 28).to(args.device)
    batch_size = args.batch_size
    input_desc = [IODescription(
        "data", [batch_size, 1, 28, 28], torch.float32), ]
    device = args.device
    inputs = {
        desc.name_: generate_sample(input_desc[0], device)
        for desc in input_desc
    }
    session = Session(model, input_desc, device)
    print(session(inputs))


def test_runner():
    model = ImageClassification(28, 10, args.num_layer, SimpleRNNCell,
                                args.hidden_size, args.batch_size, 28).to(args.device)
    batch_size = args.batch_size
    image_input = torch.ones([batch_size, 1, 28, 28],
                             dtype=torch.float32, device=args.device)
    codegen_flags = {
        "autodiff": False,  # disable backward graph for test_runner
        "training_mode": True,  # move weight external
        "extern_result_memory": True,  # move result external
        "kernels_as_files": True,  # enable parallel compilation for generated model code
        "kernels_files_number": 20,  # number of parallel compilation
        "training_optimizer": '\'' + json.dumps({"optimizer": "SGD", "learning_rate": args.learning_rate}) + '\'', # training optimizer configs
    }
    runner = Runner(model, codegen_flags=codegen_flags)
    point1 = time.time()
    out1 = runner(image_input)[0].cpu().numpy()
    point2 = time.time()
    out2 = runner(image_input)[0].cpu().numpy()
    point3 = time.time()
    print("Duration1: {}s".format(point2 - point1))
    print("Duration2: {}s".format(point3 - point2))
    assert np.allclose(out1, out2)


def train_rnn():
    print("Train RNN model on MNIST dataset with NNFusion")

    train_loader, test_loader = data_loader.get_mnist_dataloader(
        batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
    model = ImageClassification(28, 10, args.num_layer, SimpleRNNCell,
                                args.hidden_size, args.batch_size, 28).to(args.device)
    loss_func = F.nll_loss

    codegen_flags = {
        "autodiff": True,  # add backward graph
        "training_mode": True,  # move weight external
        "extern_result_memory": True,  # move result external
        "kernels_as_files": True,  # enable parallel compilation for generated model code
        "kernels_files_number": 20,  # number of parallel compilation
        "training_optimizer": '\'' + json.dumps({"optimizer": "SGD", "learning_rate": args.learning_rate}) + '\'', # training optimizer configs
    }
    trainer = Trainer(model, loss_func, device=args.device,
                      codegen_flags=codegen_flags)

    sum_loss = 0
    sum_iter = 0
    sum_time = 0
    torch.cuda.synchronize()
    start_time = time.time()
    for epoch in range(args.num_epoch):
        for i, batch in enumerate(train_loader):
            if sum_iter == 100:
                print("Epoch {}, batch {}，loss {}, time {} s".format(
                    epoch, i, sum_loss / sum_iter, sum_time / sum_iter))
                sum_loss = 0
                sum_iter = 0
                sum_time = 0

            data, label = (t.to(args.device) for t in batch)

            nnf_loss = trainer(data, label)

            torch.cuda.synchronize()
            end_time = time.time()
            iter_time = end_time - start_time
            start_time = end_time

            sum_loss += nnf_loss
            sum_time += iter_time
            sum_iter += 1

    torch.save(model.state_dict(), "/tmp/rnn.pt")


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def eval_rnn():
    print("Evaluate RNN model on MNIST dataset")

    train_loader, test_loader = data_loader.get_mnist_dataloader(
        batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
    model = ImageClassification(28, 10, args.num_layer, SimpleRNNCell,
                                args.hidden_size, args.batch_size, 28)
    state_dict = torch.load("/tmp/rnn.pt")
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    eval_acc = 0
    eval_size = 0

    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            data, label = (t.to(args.device) for t in batch)
            predict = model(data)

            eval_acc += (predict.max(dim=1)[1] == label).sum().cpu().numpy()
            eval_size += args.batch_size

    print("Evaluation accuracy:", (eval_acc / eval_size) * 100, "%")


if __name__ == "__main__":
    if args.backend == "pytorch" or args.backend == "torchscript":
        pytorch_train_rnn()
        eval_rnn()
    elif args.backend == "nnfusion":
        train_rnn()
        eval_rnn()
    elif args.backend == "eval":
        eval_rnn()
    elif args.backend == "test":
        test_session()
        test_runner()
    else:
        print("Error: unknown backend:", args.backend)
