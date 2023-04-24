# reference: https://github.com/fawazsammani/mogrifier-lstm-pytorch/blob/master/mog_lstm.py
# changed to be compiled by torchscript


import torch
import torch.nn as nn
from time import time
from nas_pytorch_unroll import NasRNNUnroll
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
arguments = parser.parse_args()
platform = arguments.platform
import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

cuda_device = torch.device("cuda:0")
n_warmup = 100
n_run = 100


class NasRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NasRNN, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(
            8, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh = nn.Parameter(torch.randn(
            8, hidden_size, hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)

    def forward(self, inputs):  # seq_len, batch, input_size
        batch_size = inputs.shape[1]
        state_c = torch.ones(batch_size, self.hidden_size, device='cuda')
        state_m = torch.ones(batch_size, self.hidden_size, device='cuda')
        for i in range(inputs.size()[0]):
            inp = inputs[i]

            ih = torch.matmul(inp, self.weight_ih)
            hh = torch.matmul(state_m, self.weight_hh)

            i0 = ih[0]
            i1 = ih[1]
            i2 = ih[2]
            i3 = ih[3]
            i4 = ih[4]
            i5 = ih[5]
            i6 = ih[6]
            i7 = ih[7]

            h0 = hh[0]
            h1 = hh[1]
            h2 = hh[2]
            h3 = hh[3]
            h4 = hh[4]
            h5 = hh[5]
            h6 = hh[6]
            h7 = hh[7]

            layer1_0 = torch.sigmoid(i0 + h0)
            layer1_1 = torch.relu(i1 + h1)
            layer1_2 = torch.sigmoid(i2 + h2)
            layer1_3 = torch.relu(i3 + h3)
            layer1_4 = torch.tanh(i4 + h4)
            layer1_5 = torch.sigmoid(i5 + h5)
            layer1_6 = torch.tanh(i6 + h6)
            layer1_7 = torch.sigmoid(i7 + h7)

            l2_0 = torch.tanh(layer1_0 * layer1_1)
            l2_1 = torch.tanh(layer1_2 + layer1_3)
            l2_2 = torch.tanh(layer1_4 * layer1_5)
            l2_3 = torch.sigmoid(layer1_6 + layer1_7)

            # Inject the cell
            l2_0_v2 = torch.tanh(l2_0 + state_c)

            # Third layer
            state_c = l2_0_v2 * l2_1
            l3_1 = torch.tanh(l2_2 + l2_3)

            # Final layer
            state_m = torch.tanh(state_c * l3_1)

        return state_m

def test_model(enable_torch, batch_size, unroll, *params):
    input_size, hidden_size, seq_len = params
    if not unroll:
        model = NasRNN(input_size, hidden_size).cuda()
    else:
        model = NasRNNUnroll(input_size, hidden_size).cuda()
    model.eval()
    if enable_torch:
        model = torch.jit.script(model)
    embed = torch.randn([seq_len, batch_size, input_size], device=cuda_device)
    print("----batch_size={}---torchscript={}----".format(batch_size, enable_torch))
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        _ = model(embed)
        torch.cuda.synchronize()
        print("Time {} ms".format((time() - t0) * 1000))

    timer = Timer("ms")
    profile_start(platform)
    torch.cuda.synchronize()
    for i in range(n_run):
        timer.start()
        _ = model(embed)
        torch.cuda.synchronize()
        timer.log()
    profile_stop(platform)
    timer.report()


def test_train(enable_torch, batch_size, *params):
    input_size, hidden_size, seq_len = params
    model = NasRNN(input_size, hidden_size).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if enable_torch:
        model = torch.jit.script(model)
    inp = torch.randn([seq_len, batch_size, input_size], device=cuda_device)
    state_h = torch.randn(1, batch_size, hidden_size, device=cuda_device)
    state_c = torch.randn(1, batch_size, hidden_size, device=cuda_device)
    print("----batch_size={}---torchscript={}----".format(batch_size, enable_torch))
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time()
        output = model(inp)
        s = torch.sum(output)
        s.backward()
        torch.cuda.synchronize()
        print("Time {} ms".format((time() - t0) * 1000))

    timer = Timer("ms")
    torch.cuda.synchronize()
    for i in range(n_run):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        timer.start()
        output = model(inp)
        s = torch.sum(output)
        s.backward()
        torch.cuda.synchronize()
        timer.log()
    timer.report()


def export_model(batch_size, input_size, hidden_size, seq_len, unroll):
    if unroll:
        model = NasRNNUnroll(input_size, hidden_size).cuda()
    else:
        model = NasRNN(input_size, hidden_size).cuda()
    model.eval()
    model = torch.jit.script(model)
    inp = torch.randn([seq_len, batch_size, input_size], device=cuda_device)
    out = model(inp)
    torch.onnx.export(model, (inp), f'nas.b{batch_size}.onnx', verbose=True, example_outputs=out)


if __name__ == '__main__':
    input_size = 256
    hidden_size = 256
    seq_len = 1000

    with torch.no_grad():
        # export_model(1, input_size, hidden_size, seq_len)
        # export_model(64, input_size, hidden_size, seq_len)
        if not arguments.overhead_test:
            test_model(True, arguments.bs, False, input_size, hidden_size, seq_len)
        else:
            if arguments.unroll:
                test_model(True, arguments.bs, True, input_size, hidden_size, seq_len)
            else:
                test_model(True, arguments.bs, False, input_size, hidden_size, seq_len)

        # test_model(False, 1, False, input_size, hidden_size, seq_len)
        # test_model(True, 1, False, input_size, hidden_size, seq_len)
        # test_model(True, 1, False, input_size, hidden_size, seq_len)
        # test_model(False, 64, False, input_size, hidden_size, seq_len)
        # test_model(True, 64, False, input_size, hidden_size, seq_len)
        # test_model(True, 1, True, input_size, hidden_size, seq_len)
