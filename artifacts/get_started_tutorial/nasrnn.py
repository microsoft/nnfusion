from ast_analyzer.shape_inference.types import *
from ast_analyzer import workflow_fix_flag, workflow_search_flag
from ast_analyzer.utils import config
import torch
import torch.nn as nn
from ast_analyzer.utils.argparser import get_parser
from ast_analyzer.to_onnx import to_torch_func
import os
parser = get_parser()

parser.add_argument('--bs', type=int, default=1)
args = parser.parse_args()
import sys
sys.setrecursionlimit(100000)
os.system("ulimit -s unlimited")

class NasRNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size):
        super(NasRNN, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(8, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh = nn.Parameter(torch.randn(8, hidden_size, hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)

    def forward(self, inputs):  # seq_len, batch, input_size
        state_c = torch.zeros(self.batch_size, self.hidden_size, device='cuda')
        state_m = torch.zeros(self.batch_size, self.hidden_size, device='cuda') # TODO: batch_size from shape
        for i in range(inputs.size()[0]): # change to 1000 for fully unrolled exp
            inp = inputs[i]
            state_m = torch.reshape(state_m, (self.batch_size, self.hidden_size))

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

        state = state_c + state_m
        return state


class NasRNNBySplit(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size):
        super(NasRNNBySplit, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(8, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh = nn.Parameter(torch.randn(8, hidden_size, hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)

    def forward(self, inputs):  # seq_len, batch, input_size
        state_c = torch.zeros(self.batch_size, self.hidden_size, device='cuda')
        state_m = torch.zeros(self.batch_size, self.hidden_size, device='cuda') # TODO: batch_size from shape
        for i in range(inputs.size()[0]): # change to 1000 for fully unrolled exp
            inp = inputs[i]
            state_m = torch.reshape(state_m, (self.batch_size, self.hidden_size))

            ih = torch.matmul(inp, self.weight_ih).view(-1, self.hidden_size)
            hh = torch.matmul(state_m, self.weight_hh).view(-1, self.hidden_size)

            i0, i1, i2, i3, i4, i5, i6, i7 = torch.split(ih, (self.batch_size, self.batch_size, self.batch_size, self.batch_size, self.batch_size, self.batch_size, self.batch_size, self.batch_size), dim=0)
            h0, h1, h2, h3, h4, h5, h6, h7 = torch.split(hh, (self.batch_size, self.batch_size, self.batch_size, self.batch_size, self.batch_size, self.batch_size, self.batch_size, self.batch_size), dim=0)

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

        state = state_c + state_m
        return state


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)
    input_size = 256
    batch_size = args.bs
    hidden_size = 256
    seq_len = 1000
    inputs = torch.randn(seq_len, batch_size, input_size).cuda()
    if args.cf:
        model = NasRNN(batch_size, input_size, hidden_size).cuda()
    else:
        model = NasRNNBySplit(batch_size, input_size, hidden_size).cuda()

    
    model = model.eval()
    o = model(inputs)

    with torch.no_grad():
        if args.cf:
            workflow_search_flag(model, f"nasrnn_bs{args.bs}", (inputs,), args.platform, args.measure, enable_control_flow=args.cf)
        else:
            to_torch_func.NNFUSION_CODEGEN_FLAGS = {
                'log_kerneldb_request': config.KERNELDB_REQUEST_FNAME
            }
            workflow_fix_flag(model, f"base_nasrnn_bs{args.bs}", (inputs,), args.platform, args.measure, enable_control_flow=args.cf)