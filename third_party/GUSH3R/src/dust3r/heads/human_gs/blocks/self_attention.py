import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv=None):
        q_norm = self.norm1(q)
        attn_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        q = q + attn_out

        q = q + self.mlp(self.norm3(q))
        return q


class HumanSelfTransformer(nn.Module):
    def __init__(self, depth, hidden_dim, num_heads, mlp_dim, use_checkpoint=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, q, kv=None):
        for layer in self.layers:
            if self.training and self.use_checkpoint and torch.is_grad_enabled():
                q = checkpoint(layer, q, use_reentrant=False)
            else:
                q = layer(q)
        return self.norm_out(q)
