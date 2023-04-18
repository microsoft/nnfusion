# perform unroll inside ast_analyzer
import torch
import torch.nn as nn

from ast_analyzer import workflow_fix_flag, test_torch_eval
from ast_analyzer.utils.argparser import get_parser
from ast_analyzer.to_onnx import to_torch_func
from ast_analyzer.python_std import disable_dynamic_unroll
disable_dynamic_unroll()
parser = get_parser()

parser.add_argument('--bs', type=int, default=1)
args = parser.parse_args()

import sys
import os
sys.setrecursionlimit(2000)
os.system("ulimit -s unlimited")

from ast_analyzer.tensor_opt import buttom_up_feed
buttom_up_feed.SEARCH_ALL_SUBAST = True

from ast_analyzer.python_std.optimizations.loop_full_unrolling import LoopFullUnrolling
LoopFullUnrolling.MAX_NODE_COUNT = 6553600000

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.randn(
            4, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh = nn.Parameter(torch.randn(
            4, hidden_size, hidden_size, dtype=torch.float32))
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
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)


class LSTMBySplit(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_size, hidden_size))
        for i in range(num_layers):
            self.layers.append(LSTMCell(hidden_size, hidden_size))
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, inputs):  # seq_len, batch, input_size
        state_c = (
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
        )
        state_h = (
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
        )
        for i in range(64):
            cur_input = inputs[i]
            for j in range(self.num_layers):
                c = state_c[j]
                h = state_h[j]
                ih = torch.matmul(cur_input, self.layers[j].weight_ih).view(-1, self.hidden_size)
                hh = torch.matmul(h, self.layers[j].weight_hh).view(-1, self.hidden_size)

                ingatei, forgetgatei, cellgatei, outgatei = torch.split(ih, (self.batch_size, self.batch_size, self.batch_size, self.batch_size), dim=0)
                ingateh, forgetgateh, cellgateh, outgateh = torch.split(hh, (self.batch_size, self.batch_size, self.batch_size, self.batch_size), dim=0)

                ingate1 = ingatei + self.layers[j].bias_ih_0 + ingateh + self.layers[j].bias_hh_0
                ingate = torch.sigmoid(ingate1)

                forgetgate1 = forgetgatei + self.layers[j].bias_ih_1 + forgetgateh + self.layers[j].bias_hh_1
                forgetgate = torch.sigmoid(forgetgate1)

                cellgate1 = cellgatei + self.layers[j].bias_ih_2 + cellgateh + self.layers[j].bias_hh_2
                cellgate = torch.tanh(cellgate1)

                outgate1 = outgatei + self.layers[j].bias_ih_3 + outgateh + self.layers[j].bias_hh_3
                outgate = torch.sigmoid(outgate1)

                c = (forgetgate * c) + (ingate * cellgate)
                h = outgate * torch.tanh(c)

                state_c[j].copy_(c)
                state_h[j].copy_(h)
                cur_input = h
        return state_h[self.num_layers - 1]


if __name__ == '__main__':
    num_layers = 10
    input_size = 256
    batch_size = args.bs
    hidden_size = 256
    seq_len = 64
    model = LSTMBySplit(batch_size, input_size, hidden_size, num_layers).cuda().eval()
    inputs = torch.randn(seq_len, batch_size, input_size).cuda()

    with torch.no_grad():
        to_torch_func.NNFUSION_CODEGEN_FLAGS = {}
        workflow_fix_flag(model, f"lstm_bs{args.bs}_unroll", (inputs,), args.platform, args.measure, run_unroll=True, enable_control_flow=args.cf)

