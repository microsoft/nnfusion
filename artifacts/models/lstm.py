from ast_analyzer.shape_inference.types import *
from ast_analyzer.utils.save_tensor import save_tensor_bin
from ast_analyzer.utils import config
from ast_analyzer import workflow_fix_flag, test_torch_eval, test_torch_train, workflow_train_recursion, workflow_search_flag
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ast_analyzer.grad.impl as grad
from ast_analyzer.utils.argparser import get_parser
import os
from ast_analyzer.to_onnx import to_torch_func
parser = get_parser()

parser.add_argument('--bs', type=int, default=1)
args = parser.parse_args()
import sys
sys.setrecursionlimit(100000)
os.system("ulimit -s unlimited")

weight_prefix = "weight/lstm-multi"

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


class LSTM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_size, hidden_size))
        for i in range(num_layers):
            self.layers.append(LSTMCell(hidden_size, hidden_size))
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
    
    def export_weight(self):
        for i in range(self.num_layers):
            save_tensor_bin(os.path.join(weight_prefix, f"weight_ih_l{i}"), self.layers[i].state_dict()['weight_ih'])
            save_tensor_bin(os.path.join(weight_prefix, f"weight_hh_l{i}"), self.layers[i].state_dict()['weight_hh'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_ih_0_l{i}"), self.layers[i].state_dict()['bias_ih_0'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_ih_1_l{i}"), self.layers[i].state_dict()['bias_ih_1'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_ih_2_l{i}"), self.layers[i].state_dict()['bias_ih_2'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_ih_3_l{i}"), self.layers[i].state_dict()['bias_ih_3'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_hh_0_l{i}"), self.layers[i].state_dict()['bias_hh_0'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_hh_1_l{i}"), self.layers[i].state_dict()['bias_hh_1'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_hh_2_l{i}"), self.layers[i].state_dict()['bias_hh_2'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_hh_3_l{i}"), self.layers[i].state_dict()['bias_hh_3'])


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
        for i in range(inputs.size()[0]):
            cur_input = inputs[i]
            for j in range(self.num_layers):
                c = state_c[j]
                h = state_h[j]
                ih = torch.matmul(cur_input, self.layers[j].weight_ih)
                hh = torch.matmul(h, self.layers[j].weight_hh)

                ingatei = ih[0]
                forgetgatei = ih[1]
                cellgatei = ih[2]
                outgatei = ih[3]

                ingateh = hh[0]
                forgetgateh = hh[1]
                cellgateh = hh[2]
                outgateh = hh[3]

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
    
    def export_weight(self):
        for i in range(self.num_layers):
            save_tensor_bin(os.path.join(weight_prefix, f"weight_ih_l{i}"), self.layers[i].state_dict()['weight_ih'])
            save_tensor_bin(os.path.join(weight_prefix, f"weight_hh_l{i}"), self.layers[i].state_dict()['weight_hh'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_ih_0_l{i}"), self.layers[i].state_dict()['bias_ih_0'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_ih_1_l{i}"), self.layers[i].state_dict()['bias_ih_1'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_ih_2_l{i}"), self.layers[i].state_dict()['bias_ih_2'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_ih_3_l{i}"), self.layers[i].state_dict()['bias_ih_3'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_hh_0_l{i}"), self.layers[i].state_dict()['bias_hh_0'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_hh_1_l{i}"), self.layers[i].state_dict()['bias_hh_1'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_hh_2_l{i}"), self.layers[i].state_dict()['bias_hh_2'])
            save_tensor_bin(os.path.join(weight_prefix, f"bias_hh_3_l{i}"), self.layers[i].state_dict()['bias_hh_3'])


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
        for i in range(inputs.size()[0]): # change to 64 for fully unrolled exp
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


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.ih = nn.Parameter(torch.randn(4, 5))
        self.hh = nn.Parameter(torch.randn(5, 5))
        self.bias = nn.Parameter(torch.randn(5))

    def forward(self, x):  # inp: seq_len*bs*hidden
        h = torch.zeros(2, 5, device='cuda')
        for i in range(x.size()[0]):
            h = torch.mm(x[i], self.ih) + torch.mm(h, self.hh) + self.bias
        return torch.sum(h)


def get_loss(model, inputs: TyTorchTensor(np.float32, (3, 2, 4))):
    result = model(inputs)
    return result


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)
    num_layers = 10
    input_size = 256
    batch_size = args.bs
    hidden_size = 256
    seq_len = 64
    # model = LSTMBySplit(batch_size, input_size, hidden_size, num_layers).cuda().eval()
    if args.cf:
        model = LSTM(batch_size, input_size, hidden_size, num_layers).cuda().eval()
    else:
        from ast_analyzer.python_std import disable_dynamic_unroll
        disable_dynamic_unroll()
        model = LSTMBySplit(batch_size, input_size, hidden_size, num_layers).cuda().eval()

    inputs = torch.randn(seq_len, batch_size, input_size).cuda()
    # save_tensor_bin(os.path.join(weight_prefix, f"inputs_b{args.bs}"), inputs)
    # model.export_weight()

    if args.mode == 'eval':
        with torch.no_grad():
            if args.run_pytorch:
                test_torch_eval(model, (inputs,), args.profile)
            if args.run_sys:
                if args.breakdown:
                    from ast_analyzer.python_std import disable_dynamic_unroll
                    disable_dynamic_unroll()
                    workflow_search_flag(model, f"lstm_bs{args.bs}_breakdown", (inputs,), args.platform, args.measure, run_unroll=True, enable_control_flow=args.cf)
                elif args.cf:
                    workflow_search_flag(model, f"lstm_bs{args.bs}", (inputs,), args.platform, args.measure, run_unroll=True, enable_control_flow=args.cf)
                else:
                    to_torch_func.NNFUSION_CODEGEN_FLAGS = {
                        'log_kerneldb_request': config.KERNELDB_REQUEST_FNAME
                    }
                    workflow_fix_flag(model, f"base_lstm_bs{args.bs}", (inputs,), args.platform, args.measure, run_unroll=True, enable_control_flow=args.cf)
                # best (loop unroll + loop in c)
                # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'cf_level': 2}
                # workflow_fix_flag(model, f"lstm_bs{args.bs}", (inputs,), args.platform, args.measure, run_unroll=True, enable_control_flow=args.cf)
                # loop in cuda
                # if args.platform == 'V100':
                #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
                #         "max_grid_dim": 160,
                #     }
                # elif args.platform == 'MI100':
                #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
                #         "max_grid_dim": 240,
                #     }
                # else: raise NotImplementedError
                # from ast_analyzer.python_std import disable_dynamic_unroll
                # disable_dynamic_unroll()
                # to_torch_func.NNFUSION_CODEGEN_FLAGS = {}
                # workflow_fix_flag(model, f"lstm_bs{args.bs}", (inputs,), args.platform, args.measure, run_unroll=True, enable_control_flow=args.cf)

    if args.mode == 'train':
        if args.run_pytorch:
            test_torch_train(model, (inputs,), args.profile)
        if args.run_sys:
            workflow_train_recursion(model, (inputs,), "lstmmulti", "cuda", args.profile=="sys", args.run_sct, use_nnfusion=False)
