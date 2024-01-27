import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
from functools import partial
import math


def default(val, d):
    if (val is not None):
        return val
    else:
        return d


class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)


class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x
    

class FullAttention(nn.Module):
    def __init__(self,
                 dim,
                 heads = 8,
                 dim_head = 64,
                 dropout = 0.,
                 qkv_bias = False
                ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x) # [B, N, H*D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) # [B, H, N, D]

        scaled_attention_logits = torch.einsum("bhxd,bhyd -> bhxy", q, k) / torch.sqrt(torch.tensor(self.dim_head, dtype=torch.float32))
        attention_weights = F.softmax(scaled_attention_logits, dim=-1) # [B, H, N, N]

        attn_output = torch.einsum("bhnx,bhxd -> bhnd", attention_weights, v) # [B, H, N, D]
        out = rearrange(attn_output, 'b h n d -> b n (h d)', h = h) # [B, N, H*D]
        out = self.to_out(out)

        return self.dropout(out)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, batch, max_len=120):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.batch = batch
        self.max_len = max_len

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print('pe', pe.shape)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.batch = batch

    def forward(self, x):
        # print(x.shape)
        # print(x.size(1))
        # print(Variable(self.pe[:, :x.size(1)], requires_grad=False).shape)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 batch_size,
                 dim_head = 64,
                 ff_chunks = 1,
                 ff_mult = 4,
                 ff_glu = False,
                 ff_dropout = 0.,
                 attn_dropout = 0.,
                 qkv_bias = True,
                ):
        super().__init__()
        self.depth = depth
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.pos_emb = PositionalEncoding(dim, 0.2, batch_size)

        wrapper_fn = partial(PreLayerNorm, dim)
        for _ in range(depth):
            self.attns.append(wrapper_fn(FullAttention(dim, heads, dim_head, attn_dropout, qkv_bias)))
            self.ffns.append(wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1)))

    def forward(self, x):
        # print(x.shape)
        # print(self.pos_emb(x).shape)
        x = x + self.pos_emb(x)
        for i in range(self.depth):
            x = x + self.attns[i](x) # residual link
            x = x + self.ffns[i](x) # residual link
        return x