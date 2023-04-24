import torch
import torch.nn as nn

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

        return k, v, x