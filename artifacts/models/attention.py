import torch
import torch.nn as nn

from ast_analyzer import workflow_fix_flag, test_torch_eval, workflow_search_flag
from ast_analyzer.utils.argparser import get_parser
from ast_analyzer.to_onnx import to_torch_func
parser = get_parser()

parser.add_argument('--bs', type=int, default=1)
args = parser.parse_args()

START_LEN = 32
SEQ_LEN = 64
NUM_HEAD = 12
SIZE_PER_HEAD = 64

class Attention(nn.Module):
    def __init__(self, num_head, size_per_head):
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
        self.start_len = START_LEN
        self.seq_len = SEQ_LEN

    def forward(self, x, k, v): # (batch_size, num_head, 1, size_per_head)
        k = k + 0.0
        v = v + 0.0
        batch_size = x.size()[0]
        gen_id = self.start_len
        attn = torch.zeros(batch_size, self.num_head, 1, self.seq_len, device='cuda')
        for i in range(k.size()[2] - self.start_len):
            q = torch.matmul(x, self.weight_q)
            k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
            v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
            attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
            attn = attn * 0.125
            attn = torch.softmax(attn, dim=3)
            x = torch.matmul(attn, v)
            x = torch.matmul(x, self.weight_o)
            gen_id = gen_id + 1
        return k, v, x


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = args.bs
    model = Attention(NUM_HEAD, SIZE_PER_HEAD).cuda().eval()
    x = torch.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD).cuda()
    k = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    k[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    with torch.no_grad():
        if args.run_pytorch:
            test_torch_eval(model, (x, k, v), args.profile)
        if args.run_sys:
            # best: bs=1 loop in cuda + dynamic unroll bs=64 loop in cuda
            # tofix: split_For.split_body is not working, change can_merge_body to True to run this config

            # if batch_size == 1:
            #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            #         'max_block_dim': 64
            #     }
            # elif batch_size == 64:
            #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            #         'max_block_dim': 128,
            #         'max_grid_dim': 320,
            #     }
            # else:
            #     raise NotImplementedError
            if args.breakdown:
                workflow_search_flag(model, f"attention_bs{args.bs}_breakdown", (x, k, v), args.platform, args.measure, run_unroll=False, enable_control_flow=args.cf)
            elif args.cf:
                if args.platform == 'V100':
                    run_unroll = (batch_size == 1)
                elif args.platform == 'MI100':
                    run_unroll = False
                else:
                    raise NotImplementedError
                workflow_search_flag(model, f"attention_bs{args.bs}", (x, k, v), args.platform, args.measure, run_unroll=run_unroll, enable_control_flow=args.cf)
            else:
                raise NotImplementedError("please run the manual version")
            # workflow_fix_flag(model, f"attention_bs{args.bs}", (x, k, v), args.platform, args.measure, run_unroll=run_unroll, enable_control_flow=args.cf)

            # loop in cuda
            # if batch_size == 1:
            #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            #         'max_block_dim': 64
            #     }
            # elif batch_size == 64:
            #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            #         'max_block_dim': 128,
            #         'max_grid_dim': 320
            #     }
            # else:
            #     raise NotImplementedError
            # workflow_fix_flag(model, f"attention_bs{args.bs}", (x, k, v), args.platform, args.measure, enable_control_flow=args.cf)

            # dynamic unroll
            # to_torch_func.NNFUSION_CODEGEN_FLAGS = {'cf_level': 2}
            # workflow_fix_flag(model, f"attention_bs{args.bs}", (x, k, v), args.platform, args.measure, run_unroll=True, enable_control_flow=args.cf)

            # dynamic unroll + loop in cuda
            # if batch_size == 1:
            #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            #         'max_block_dim': 64
            #     }
            # elif batch_size == 64:
            #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            #         'max_block_dim': 128,
            #         'max_grid_dim': 160
            #     }
            # workflow_fix_flag(model, f"attention_bs{args.bs}", (x, k, v), args.platform, args.measure, run_unroll=True, enable_control_flow=args.cf)


            # pytorch + our kernel
            # if batch_size == 1:
            #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            #         'max_block_dim': 64,
            #     }
            # elif batch_size == 64:
            #     to_torch_func.NNFUSION_CODEGEN_FLAGS = {
            #         'max_block_dim': 128,
            #     }
            # else:
            #     raise NotImplementedError
            
            # workflow_fix_flag(model, f"attention_bs{args.bs}", (x, k, v), args.platform, args.measure, enable_control_flow=False)


    # y = model(x)
    # print(y.size())


            

