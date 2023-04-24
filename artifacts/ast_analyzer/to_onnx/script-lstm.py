import torch
import torch.nn as nn
import torch.jit as jit
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--num_step', default='100', type=int)
parser.add_argument('--num_layer', default='1', type=int)
parser.add_argument('--input_size', default='256', type=int)
parser.add_argument('--hidden_size', default='128', type=int)
parser.add_argument('--batch_size', default='4', type=int)
parser.add_argument('--output_size', default='2', type=int)
parser.add_argument('--warmup', default='5', type=int)
parser.add_argument('--num_iter', default='10', type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--jit', default=False, type=bool)


class LSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.weight_ih_0 = nn.Parameter(torch.randn(input_size, hidden_size, device='cuda'))
        self.weight_hh_0 = nn.Parameter(torch.randn(hidden_size, hidden_size, device='cuda'))
        self.bias_ih_0 = nn.Parameter(torch.randn(hidden_size, device='cuda'))
        self.bias_hh_0 = nn.Parameter(torch.randn(hidden_size, device='cuda'))
        self.weight_ih_1 = nn.Parameter(torch.randn(input_size, hidden_size, device='cuda'))
        self.weight_hh_1 = nn.Parameter(torch.randn(hidden_size, hidden_size, device='cuda'))
        self.bias_ih_1 = nn.Parameter(torch.randn(hidden_size, device='cuda'))
        self.bias_hh_1 = nn.Parameter(torch.randn(hidden_size, device='cuda'))
        self.weight_ih_2 = nn.Parameter(torch.randn(input_size, hidden_size, device='cuda'))
        self.weight_hh_2 = nn.Parameter(torch.randn(hidden_size, hidden_size, device='cuda'))
        self.bias_ih_2 = nn.Parameter(torch.randn(hidden_size, device='cuda'))
        self.bias_hh_2 = nn.Parameter(torch.randn(hidden_size, device='cuda'))
        self.weight_ih_3 = nn.Parameter(torch.randn(input_size, hidden_size, device='cuda'))
        self.weight_hh_3 = nn.Parameter(torch.randn(hidden_size, hidden_size, device='cuda'))
        self.bias_ih_3 = nn.Parameter(torch.randn(hidden_size, device='cuda'))
        self.bias_hh_3 = nn.Parameter(torch.randn(hidden_size, device='cuda'))
        self.hidden_size = hidden_size
        self.out_w = nn.Parameter(torch.randn(hidden_size, output_size))
        self.output_size = output_size

    @jit.script_method
    def forward(self, inputs, state_c, state_h): # seq_len, batch, 
        for i in range(inputs.size()[0]):
            inp = inputs[i]

            ingate1 = torch.matmul(inp, self.weight_ih_0) + self.bias_ih_0 + torch.matmul(state_h, self.weight_hh_0) + self.bias_hh_0
            ingate = torch.sigmoid(ingate1)

            forgetgate1 = torch.matmul(inp, self.weight_ih_1) + self.bias_ih_1 + torch.matmul(state_h, self.weight_hh_1) + self.bias_hh_1
            forgetgate = torch.sigmoid(forgetgate1)

            cellgate1 = torch.matmul(inp, self.weight_ih_2) + self.bias_ih_2 + torch.matmul(state_h, self.weight_hh_2) + self.bias_hh_2
            cellgate = torch.tanh(cellgate1)

            outgate1 = torch.matmul(inp, self.weight_ih_3) + self.bias_ih_3 + torch.matmul(state_h, self.weight_hh_3) + self.bias_hh_3
            outgate = torch.sigmoid(outgate1)

            state_c = (forgetgate * state_c) + (ingate * cellgate)
            state_h = outgate * torch.tanh(state_c)

        result = torch.matmul(state_h, self.out_w)

        return result


def main_pytorch():
    print("GPU support:", torch.cuda.is_available())

    args = parser.parse_args()

    inputs = torch.ones(args.num_step, args.batch_size,
                        args.input_size, device='cuda')
    state_c = torch.zeros(inputs.shape[1], args.hidden_size, device='cuda')
    state_h = torch.zeros(inputs.shape[1], args.hidden_size, device='cuda')
    initial_states = (torch.zeros(args.batch_size, args.hidden_size, device='cuda'),
                      torch.zeros(args.batch_size, args.hidden_size, device='cuda'))
    rnn = LSTM(args.input_size, args.hidden_size, args.output_size).cuda()
    rnn.eval()

    print(rnn.graph_for(inputs, state_c, state_h))

    torch.cuda.synchronize()

    for i in range(args.warmup):
        rnn(inputs, state_c, state_h)
    torch.cuda.synchronize()

    iter_times = []

    for i in range(args.num_iter):
        start_time = time.time()
        result = rnn(inputs, state_c, state_h)
        torch.cuda.synchronize()
        iter_time = (time.time() - start_time) * 1000
        iter_times.append(iter_time)
        print("Iteration time %f ms" % (iter_time))

    print("Summary: [min, max, mean] = [%f, %f, %f] ms" % (
        min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))

    torch.onnx.export(rnn, (inputs, state_c, state_h), 'lstm-simple.onnx', verbose = True, opset_version = 11, example_outputs = (result,))


if __name__ == "__main__":
    main_pytorch()
