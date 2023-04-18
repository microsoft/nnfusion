from ast_analyzer.shape_inference.types import *
from ast_analyzer import workflow_fix_flag, test_torch_eval, test_torch_train, workflow_train_recursion, workflow_search_flag
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from ast_analyzer.utils.argparser import get_parser
from ast_analyzer.to_onnx import to_torch_func
from ast_analyzer.utils.timer import Timer
from ast_analyzer.utils.nvprof import enable_profile, profile_start, profile_stop

parser = get_parser()

parser.add_argument('--bs', type=int, default=1)
args = parser.parse_args()

device = torch.device('cuda')

prefix = "../data/seq2seq"
MAX_LENGTH = 50
OUTPUT_SIZE = 3797
HIDDEN_SIZE = 256

class LSTMCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        # self.weight_ih_l0 = nn.Parameter(torch.randn(3 * hidden_size, input_size, dtype=torch.float32))
        # self.weight_hh_l0 = nn.Parameter(torch.randn(3 * hidden_size, input_size, dtype=torch.float32))
        # self.bias_ih_l0 = nn.Parameter(torch.randn(3 * hidden_size, dtype=torch.float32))
        # self.bias_hh_l0 = nn.Parameter(torch.randn(3 * hidden_size, dtype=torch.float32))
        self.weight_ih_l0_t = nn.Parameter(torch.randn(4, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh_l0_t = nn.Parameter(torch.randn(4, input_size, hidden_size, dtype=torch.float32))
        # self.bias_ih_l0_t = nn.Parameter(torch.randn(3, 1, hidden_size, dtype=torch.float32))
        # self.bias_hh_l0_t = nn.Parameter(torch.randn(3, 1, hidden_size, dtype=torch.float32))
        self.bias_ih_0 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_0 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_1 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_1 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_2 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_2 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_3 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_3 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        self.input_size = input_size
        nn.init.xavier_uniform_(self.weight_ih_l0_t)
        nn.init.xavier_uniform_(self.weight_hh_l0_t)
    
    # def update_param(self):
    #     self.state_dict()[f"weight_ih_l0_t"][:] = torch.transpose(self.weight_ih_l0.view(3, self.hidden_size, self.input_size), 1, 2)
    #     self.state_dict()[f"bias_ih_l0_t"][:] = self.bias_ih_l0.reshape((3, 1, self.hidden_size))
    #     self.state_dict()[f"weight_hh_l0_t"][:] = torch.transpose(self.weight_hh_l0.view(3, self.hidden_size, self.input_size), 1, 2)
    #     self.state_dict()[f"bias_hh_l0_t"][:] = self.bias_hh_l0.reshape((3, 1, self.hidden_size))
    #     self.state_dict()[f"bias_ih_0"][:] = self.bias_ih_l0[0 * self.hidden_size: 1 * self.hidden_size]
    #     self.state_dict()[f"bias_hh_0"][:] = self.bias_hh_l0[0 * self.hidden_size: 1 * self.hidden_size]
    #     self.state_dict()[f"bias_ih_1"][:] = self.bias_ih_l0[1 * self.hidden_size: 2 * self.hidden_size]
    #     self.state_dict()[f"bias_hh_1"][:] = self.bias_hh_l0[1 * self.hidden_size: 2 * self.hidden_size]
    #     self.state_dict()[f"bias_ih_2"][:] = self.bias_ih_l0[2 * self.hidden_size: 3 * self.hidden_size]
    #     self.state_dict()[f"bias_hh_2"][:] = self.bias_hh_l0[2 * self.hidden_size: 3 * self.hidden_size]


    def forward(self, x, h, c):
        ih = torch.matmul(x, self.weight_ih_l0_t)
        hh = torch.matmul(h, self.weight_hh_l0_t)
        ih0 = ih[0] + self.bias_ih_0
        hh0 = hh[0] + self.bias_hh_0
        ih1 = ih[1] + self.bias_ih_1
        hh1 = hh[1] + self.bias_hh_1
        ih2 = ih[2] + self.bias_ih_2
        hh2 = hh[2] + self.bias_hh_2
        ih3 = ih[3] + self.bias_ih_3
        hh3 = hh[3] + self.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        return h, c

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.gru = LSTMCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # self.EOS_token = nn.Parameter(torch.full((1,), 0, dtype=torch.long, device=device), requires_grad=False)
        self.EOS_token = 0
        self.SOS_token = 1

    def forward(self, encoder_output, std, h, c):
        # hidden: (1, bs, hidden_size)
        # encoder_outputs: (max_length, bs, hidden_size)
        batch_size = encoder_output.size()[1]
        output_all = torch.zeros(self.max_length, batch_size, dtype=torch.int64, device='cuda') + 0 # Hack for bug in ScatterND on Constant
        output = torch.full((batch_size,), self.SOS_token, dtype=torch.int64, device='cuda')
        cond = True
        # when disable cf
        # id = torch.zeros((), dtype=torch.int64, device='cuda')
        id = 0
        while cond:
            x = self.embedding(output)
            h = torch.reshape(h, (batch_size, self.hidden_size))
            # lstm start
            ih = torch.matmul(x, self.gru.weight_ih_l0_t)
            hh = torch.matmul(h, self.gru.weight_hh_l0_t)
            ih0 = ih[0] + self.gru.bias_ih_0
            hh0 = hh[0] + self.gru.bias_hh_0
            ih1 = ih[1] + self.gru.bias_ih_1
            hh1 = hh[1] + self.gru.bias_hh_1
            ih2 = ih[2] + self.gru.bias_ih_2
            hh2 = hh[2] + self.gru.bias_hh_2
            ih3 = ih[3] + self.gru.bias_ih_3
            hh3 = hh[3] + self.gru.bias_hh_3

            ingate = torch.sigmoid(ih0 + hh0)
            forgetgate = torch.sigmoid(ih1 + hh1)
            cellgate = torch.tanh(ih2 + hh2)
            outgate = torch.sigmoid(ih3 + hh3)

            c = (forgetgate * c) + (ingate * cellgate)
            h = outgate * torch.tanh(c)
            # lstm end
            output = self.out(h) + std[id]
            output = output.argmax(1)
            output_all[id] = output
            id = id + 1
            # cond = bool((torch.max(output) > self.EOS_token).item()) & (id < self.max_length) # when testing torchscript
            cond = (torch.max(output) > self.EOS_token) & (id < self.max_length)
        return output_all, h


class AttnDecoderRNNUnroll(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNNUnroll, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.gru = LSTMCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # self.EOS_token = nn.Parameter(torch.full((1,), 0, dtype=torch.long, device=device), requires_grad=False)
        self.EOS_token = 0
        self.SOS_token = 1

    def forward(self, encoder_output, std, h, c):
        # hidden: (1, bs, hidden_size)
        # encoder_outputs: (max_length, bs, hidden_size)
        batch_size = encoder_output.size()[1]
        # output_all = torch.zeros(self.max_length, batch_size, dtype=torch.int64, device='cuda') + 0 # Hack for bug in ScatterND on Constant
        output_all = torch.zeros(self.max_length, batch_size, dtype=torch.int64, device='cuda') + 0 # Hack for bug in ScatterND on Constant
        output = torch.full((batch_size,), self.SOS_token, dtype=torch.int64, device='cuda')
        cond = True
        # when disable cf
        # id = torch.zeros((), dtype=torch.int64, device='cuda')
        id = 0
        # while cond:
        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1
        cond = (torch.max(output) > self.EOS_token) & (id < self.max_length)

        # while cond:
        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1 # id = 2
        cond = cond & (torch.max(output) > self.EOS_token) & (id < self.max_length)

        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1 # id = 3
        cond = cond & (torch.max(output) > self.EOS_token) & (id < self.max_length)

        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1 # id = 4
        cond = cond & (torch.max(output) > self.EOS_token) & (id < self.max_length)

        # while cond:
        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1 # id = 5
        cond = cond & (torch.max(output) > self.EOS_token) & (id < self.max_length)

        # while cond:
        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1 # id = 6
        cond = cond & (torch.max(output) > self.EOS_token) & (id < self.max_length)

        # while cond:
        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1 # id = 7
        cond = cond & (torch.max(output) > self.EOS_token) & (id < self.max_length)

        # while cond:
        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1 # id = 8
        cond = cond & (torch.max(output) > self.EOS_token) & (id < self.max_length)

        # while cond:
        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1 # id = 9
        cond = cond & (torch.max(output) > self.EOS_token) & (id < self.max_length)
        
        # while cond:
        x = self.embedding(output)
        h = torch.reshape(h, (batch_size, self.hidden_size))
        # lstm start
        ih = torch.matmul(x, self.gru.weight_ih_l0_t)
        hh = torch.matmul(h, self.gru.weight_hh_l0_t)
        ih0 = ih[0] + self.gru.bias_ih_0
        hh0 = hh[0] + self.gru.bias_hh_0
        ih1 = ih[1] + self.gru.bias_ih_1
        hh1 = hh[1] + self.gru.bias_hh_1
        ih2 = ih[2] + self.gru.bias_ih_2
        hh2 = hh[2] + self.gru.bias_hh_2
        ih3 = ih[3] + self.gru.bias_ih_3
        hh3 = hh[3] + self.gru.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        # lstm end
        output = self.out(h) + std[id]
        output = output.argmax(1)
        output_all[id] = output
        id = id + 1 # id = 10
        cond = cond & (torch.max(output) > self.EOS_token) & (id < self.max_length)

        return output_all, h, cond


def load_model():
    if args.overhead_test and args.unroll:
        attn_decoder1 = AttnDecoderRNNUnroll(HIDDEN_SIZE, OUTPUT_SIZE, dropout_p=0.1).to(device).eval()
        attn_decoder1 = attn_decoder1.eval()
    else:
        attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE, OUTPUT_SIZE, dropout_p=0.1).to(device).eval()
        attn_decoder1 = attn_decoder1.eval()
    return attn_decoder1


def gen_mask_from_sequence(std):
    bs = std.shape[0]
    padded_std = torch.zeros((bs, MAX_LENGTH), dtype=std.dtype, device=device)
    padded_std[:, :std.shape[1]] = std
    mask = torch.zeros(bs, MAX_LENGTH, OUTPUT_SIZE, device=device)
    mask[torch.arange(bs).unsqueeze(1), torch.arange(MAX_LENGTH).unsqueeze(0), padded_std] = 1000000.0
    mask = mask.transpose(0, 1).contiguous().clone()
    return mask

def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = torch.from_numpy(np.fromfile(s + ".bin", dtype=dtype)).reshape(shape)
    return tensor

def save_bin(data, path):
    data = data.clone().detach().cpu().numpy()
    with open(path + ".shape", "w") as f: f.write(" ".join(str(x) for x in data.shape))
    data.tofile(path + ".bin")

if __name__ == '__main__':
    with torch.no_grad():
        torch.manual_seed(0)
        torch.set_printoptions(precision=10)
        batch_size = args.bs
        model = load_model()

        std = []
        MAX_LENGTH = 50
        for i in range(batch_size):
            l = max(i, 10)
            l = min(l, MAX_LENGTH)
            lst = list(range(1, l))
            lst.append(0)
            assert(len(lst) <= MAX_LENGTH)
            # pad to MAX_LENGTH
            lst = lst + [0] * (MAX_LENGTH - len(lst))
            std.append(lst)
        std = torch.tensor(std, device=device)
        print("std=", std)
        mask = gen_mask_from_sequence(std)
        encoder_output = torch.randn(MAX_LENGTH, batch_size, HIDDEN_SIZE, device=device)
        h = torch.randn(batch_size, HIDDEN_SIZE, device='cuda')
        c = torch.randn(batch_size, HIDDEN_SIZE, device='cuda')
        if args.run_pytorch:
            test_torch_eval(model, (encoder_output, mask, h, c), args.profile)
        if args.run_sys:
            n_warmup = 100
            n_run = 100
            if not args.overhead_test:
                # loop in cuda (best config)
                workflow_search_flag(model, f'seq2seq_bs{args.bs}', (encoder_output, mask, h, c), args.platform, time_measure=False, enable_control_flow=args.cf)
                # to_torch_func.NNFUSION_CODEGEN_FLAGS = {}
                # if args.bs == 64:
                #     to_torch_func.NNFUSION_CODEGEN_FLAGS['max_grid_dim'] = 80
                #     to_torch_func.NNFUSION_CODEGEN_FLAGS['max_block_dim'] = 256
                # workflow_fix_flag(model, f'seq2seq_bs{args.bs}', (encoder_output, mask, h, c), args.platform, time_measure=False, enable_control_flow=args.cf)
                # loop unroll + loop in cuda
                # to_torch_func.NNFUSION_CODEGEN_FLAGS = {}
                # if args.bs == 64:
                #     to_torch_func.NNFUSION_CODEGEN_FLAGS['max_grid_dim'] = 80
                #     to_torch_func.NNFUSION_CODEGEN_FLAGS['max_block_dim'] = 256
                # workflow_fix_flag(model, f'seq2seq_bs{args.bs}', (encoder_output, mask, h, c), args.platform, time_measure=False, enable_control_flow=args.cf, run_unroll=True)
                # breakdowns
                # loop unroll + loop in c + branch in c
                # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'loop_in_c': True, 'cf_level': 2, 'branch_fine_grained': False}
                # loop unroll + branch in cuda
                # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'loop_in_c': True, 'cf_level': 2, 'branch_fine_grained': False}
                # loop unroll + branch launch then else
                # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'loop_in_c': True, 'cf_level': 2, 'branch_fine_grained': True}
                # if args.bs == 64:
                #     to_torch_func.NNFUSION_CODEGEN_FLAGS['max_grid_dim'] = 80
                #     to_torch_func.NNFUSION_CODEGEN_FLAGS['max_block_dim'] = 256
                # workflow_fix_flag(model, f'seq2seq_bs{args.bs}', (encoder_output, mask, h, c), args.platform, time_measure=False, enable_control_flow=args.cf, run_unroll=True)
                if not args.measure: exit(0)
                len_dataset = 6400
                tokens = read_bin('../data/tatoeba-eng-fra/tokens', dtype=np.int64).cuda()
                masks = gen_mask_from_sequence(tokens)
                for i in range(0, len_dataset, args.bs):
                    if i >= n_warmup * args.bs: break
                    mask = masks[:, i:i+args.bs].contiguous()
                    torch.cuda.synchronize()
                    _ = model.forward(encoder_output, mask, h, c)
                    torch.cuda.synchronize()
                # run
                timer = Timer("ms")
                enable_profile(args.platform)
                profile_start(args.platform)
                for i in range(0, len_dataset, args.bs):
                    if i >= n_run * args.bs: break
                    mask = masks[:, i:i+args.bs].contiguous()
                    torch.cuda.synchronize()
                    timer.start()
                    _ = model.forward(encoder_output, mask, h, c)
                    torch.cuda.synchronize()
                    timer.log()
                timer.report()
                profile_stop(args.platform)
            else:
                std = []
                MAX_LENGTH = 50
                for i in range(batch_size):
                    l = 10
                    lst = list(range(1, l))
                    lst.append(0)
                    assert(len(lst) <= MAX_LENGTH)
                    # pad to MAX_LENGTH
                    lst = lst + [0] * (MAX_LENGTH - len(lst))
                    std.append(lst)
                std = torch.tensor(std, device=device)
                print("std=", std)
                mask = gen_mask_from_sequence(std)
                encoder_output = torch.randn(MAX_LENGTH, batch_size, HIDDEN_SIZE, device=device)
                # fixed_data_prefix = '../data/seq2seq/fix_test'
                # torch.save(model.state_dict(), f"{fixed_data_prefix}/seq2seq.pt")
                # save_bin(mask, f"{fixed_data_prefix}/mask")
                # save_bin(encoder_output, f"{fixed_data_prefix}/encoder_output")
                # save_bin(h, f"{fixed_data_prefix}/h")
                # save_bin(c, f"{fixed_data_prefix}/c")
                # out = model.forward(encoder_output, mask, h, c)
                # save_bin(out[0], f"{fixed_data_prefix}/output_all")
                # save_bin(out[1], f"{fixed_data_prefix}/hidden")
                to_torch_func.NNFUSION_CODEGEN_FLAGS = {
                    "enable_extern_result_inline": False
                }
                workflow_fix_flag(model, f'seq2seq_bs{args.bs}_unroll_{args.unroll}', (encoder_output, mask, h, c), args.platform, time_measure=args.measure, enable_control_flow=args.cf)
