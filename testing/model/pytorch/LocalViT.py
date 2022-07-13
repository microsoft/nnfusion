import torch
from torch import nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class ConvFF(nn.Module):

    def __init__(self, dim = 192, scale = 4, depth_kernel = 3, patch_height = 14, patch_width = 14, dropout=0.):
        super().__init__()

        scale_dim = dim*scale
        self.up_proj = nn.Sequential(
                                    Rearrange('b (h w) c -> b c h w', h=patch_height, w=patch_width),
                                    nn.Conv2d(dim, scale_dim, kernel_size=1),
                                    nn.Hardswish()
                                    )

        self.depth_conv = nn.Sequential(
                        nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=True),
                        nn.Conv2d(scale_dim, scale_dim, kernel_size=1, bias=True),
                        nn.Hardswish()
                        )

        self.down_proj = nn.Sequential(
                                    nn.Conv2d(scale_dim, dim, kernel_size=1),
                                    nn.Dropout(dropout),
                                    Rearrange('b c h w ->b (h w) c')
                                    )

    def forward(self, x):

        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, patch_height, patch_width, scale = 4, depth_kernel = 3, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, ConvFF(dim, scale, depth_kernel, patch_height, patch_width)))
            ]))
    def forward(self, x):

        for attn, convff in self.layers:
            x = attn(x)
            cls_tokens = x[:, 0]
            x = convff(x[:, 1:])
            x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        return x



class LocalViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim = 192, depth = 12, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., depth_kernel = 3, scale_dim = 4):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'



        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        patch_height = int(image_size//patch_size)
        patch_width = int(image_size//patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, patch_height, patch_width, scale_dim, depth_kernel, dropout)



        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
