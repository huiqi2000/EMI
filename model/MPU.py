# *_*coding:utf-8 *_*
"""
adapted from: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""
import copy

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_seq_len=1024, learnable=False):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        else:
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            #pe[:, 1::2] = torch.cos(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
            pe = pe.unsqueeze(0) # Note: pe with size (1, seq_len, feature_dim)
            self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: with size (batch_size, seq_len, feature_dim)
        :return:
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


# Note: GEGLU() is different from that (i.e., GELU()) in mbt.py
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h) # (B*h, 1, T2)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, ff_expansion=4, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim, mult=ff_expansion, dropout=ff_dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


# half MPU (mutual promotion unit)
class CrossSelfTransformer(nn.Module):
    def __init__(self, latent_dim, input_dim, depth, heads, dim_head, ff_expansion=4, attn_dropout=0., ff_dropout=0.):
        """
        :param latent_dim: dim of target (query)
        :param input_dim:  dim of source/context (key/value)
        :param depth: number of layers
        :param heads: number of attention heads
        :param dim_head: dim of each head
        :param ff_expansion: expansion factor of feed-forward layer
        :param attn_dropout:
        :param ff_dropout:
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, context_dim=input_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                        context_dim=input_dim),
                PreNorm(latent_dim, Attention(latent_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(latent_dim, FeedForward(latent_dim, mult=ff_expansion, dropout=ff_dropout))
            ]))

    def forward(self, x, context, mask=None, context_mask=None):
        """
        :param x: latent array, (B, T1, D1)
        :param context: input array, (B, T2, D2)
        :param mask: padding mask, (B, T1)
        :param context_mask: padding mask for context, (B, T2)
        :return: (B, T1, D1)
        """
        for cross_attn, self_attn, ff in self.layers:
            x = cross_attn(x, context=context, mask=context_mask) + x
            x = self_attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class DualCrossSelfTransformer(nn.Module):
    def __init__(self, a_dim, b_dim, len_feature, numclasses):
        super().__init__()
        self.input_a = nn.Linear(a_dim, 128, bias=False)
        self.input_b = nn.Linear(b_dim, 128, bias=False)

        self.b_a = CrossSelfTransformer(latent_dim=128, input_dim=128, depth=1, heads=4, dim_head=32,
                                    ff_expansion=4, attn_dropout=0., ff_dropout=0.)
        
        self.a_b = CrossSelfTransformer(latent_dim=128, input_dim=128, depth=1, heads=4, dim_head=32,
                                    ff_expansion=4, attn_dropout=0., ff_dropout=0.)

        self.regress = nn.Sequential(
            nn.Linear(128*2*len_feature, 128),
            nn.ReLU(),
            nn.Linear(128, numclasses)
        )
    
    def forward(self, x_a, x_b):

        x_a = self.input_a(x_a)
        x_b = self.input_b(x_b)

        b_a = self.b_a(x_a, x_b)
        a_b = self.a_b(x_b, x_a)

        x = torch.concat((b_a, a_b), dim=-1)
        bathsize = x.shape[0]
        x = x.reshape(bathsize, -1)

        return self.regress(x)

if __name__ == "__main__":
    import torch

    # A = torch.randn(4, 32, 128)
    # B = torch.randn(4, 32, 128)

    # modelB_A = CrossSelfTransformer(latent_dim=128, input_dim=128, depth=1, heads=4, dim_head=32,
    #                                 ff_expansion=4, attn_dropout=0., ff_dropout=0.)
    
    # modelA_B = CrossSelfTransformer(latent_dim=128, input_dim=128, depth=1, heads=4, dim_head=32,
    #                                 ff_expansion=4, attn_dropout=0., ff_dropout=0.)
    
    # B_A = modelB_A(A, B)

    # A_B = modelA_B(B, A)

    x_v = torch.rand(4, 32, 384)
    x_a = torch.rand(4, 32, 768)
    model = DualCrossSelfTransformer(a_dim=384, b_dim=768, len_feature=32, numclasses=6)

    out = model(x_v, x_a)

     
    print(out.shape)