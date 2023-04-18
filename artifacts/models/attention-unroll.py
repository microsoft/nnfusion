import torch
import torch.nn as nn

from ast_analyzer import workflow_fix_flag, test_torch_eval
from ast_analyzer.utils.argparser import get_parser
from ast_analyzer.to_onnx import to_torch_func
parser = get_parser()

parser.add_argument('--bs', type=int, default=1)
args = parser.parse_args()

START_LEN = 32
SEQ_LEN = 64
NUM_HEAD = 12
SIZE_PER_HEAD = 64

class AttentionUnroll(nn.Module):
    def __init__(self, num_head, size_per_head, start_len, seq_len):
        super().__init__()
        self.weight_q = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_k = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_v = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_o = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight_q)
        nn.init.xavier_uniform_(self.weight_k)
        nn.init.xavier_uniform_(self.weight_v)
        nn.init.xavier_uniform_(self.weight_o)
        self.num_head = num_head
        self.size_per_head = size_per_head
        self.start_len = start_len
        self.seq_len = seq_len

    def forward(self, x, k, v): # (batch_size, num_head, 1, size_per_head)
        k = k + 0.0
        v = v + 0.0
        batch_size = x.size()[0]
        gen_id = self.start_len
        attn = torch.zeros(batch_size, self.num_head, 1, self.seq_len, device='cuda')
        
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1
        q = torch.matmul(x, self.weight_q)
        k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
        v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
        attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
        attn = attn * 0.125
        attn = torch.softmax(attn, dim=3)
        x = torch.matmul(attn, v)
        x = torch.matmul(x, self.weight_o)
        gen_id = gen_id + 1

        k = k + 0.0
        v = v + 0.0

        return k, v, x


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = args.bs
    model = AttentionUnroll(NUM_HEAD, SIZE_PER_HEAD, START_LEN, SEQ_LEN).cuda().eval()
    x = torch.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD).cuda()
    k = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    k[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    to_torch_func.NNFUSION_CODEGEN_FLAGS = {}

    with torch.no_grad():
        workflow_fix_flag(model, f"attention_bs{args.bs}_unroll", (x, k, v), args.platform, args.measure, enable_control_flow=args.cf)
