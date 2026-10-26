"""Transformer encoder with learnable summary tokens (modality encoders, as in LNLN / ALMT)."""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm_q(q), self.norm_k(k), self.norm_v(v), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, q, k, v, attn_mask=None):
        """attn_mask: optional bool tensor (B, 1, 1, N_keys), True = the key takes part in attention."""
        h = self.heads
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, x, x, attn_mask=attn_mask) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    """Sequence encoder with `token_len` learnable summary tokens prepended."""

    def __init__(self, *, num_frames, token_len, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.token_len = token_len
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x, key_padding_mask=None):
        """x: (B, N, dim). key_padding_mask: optional bool (B, N), True = missing / padded frame."""
        b, n, _ = x.shape
        attn_mask = None
        if key_padding_mask is not None:
            keep = torch.cat((torch.ones(b, self.token_len, dtype=torch.bool, device=x.device), ~key_padding_mask), dim=1)
            attn_mask = keep[:, None, None, :]
        extra_token = repeat(self.extra_token, '1 n d -> b n d', b=b)
        x = torch.cat((extra_token, x), dim=1)
        x = x + self.pos_embedding[:, :n + self.token_len]
        x = self.dropout(x)
        return self.encoder(x, attn_mask=attn_mask)
