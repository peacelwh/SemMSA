"""Visual / acoustic prompt adapters of CSR.

M learnable prompts attend to the positionally-encoded frame (segment) features through cross-attention, interact
through self-attention and a feed-forward layer; after B blocks they are projected into the LLM embedding space.
"""
import torch
import torch.nn as nn


class AdapterBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, prompts, feats, key_padding_mask=None):
        kv = self.norm_kv(feats)
        q = self.norm_q1(prompts)
        prompts = prompts + self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)[0]
        q = self.norm_q2(prompts)
        prompts = prompts + self.self_attn(q, q, q, need_weights=False)[0]
        return prompts + self.ffn(self.norm_ffn(prompts))


class PromptAdapter(nn.Module):
    def __init__(self, in_dim, dim, out_dim, num_prompts=8, depth=2, heads=4, mlp_ratio=4, dropout=0., max_len=512):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim)
        self.in_norm = nn.LayerNorm(dim)
        self.pos_embed = nn.Parameter(nn.init.trunc_normal_(torch.zeros(1, max_len, dim), std=0.02))
        self.prompts = nn.Parameter(nn.init.trunc_normal_(torch.zeros(1, num_prompts, dim), std=0.02))
        self.blocks = nn.ModuleList([AdapterBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_dim)

    def forward(self, x, key_padding_mask=None):
        """x: (B, T, in_dim), zeros at missing positions -> (B, M, out_dim)."""
        b, t, _ = x.shape
        feats = self.in_norm(self.in_proj(x)) + self.pos_embed[:, :t]
        prompts = self.prompts.expand(b, -1, -1)
        for blk in self.blocks:
            prompts = blk(prompts, feats, key_padding_mask)
        return self.out_proj(self.out_norm(prompts))

