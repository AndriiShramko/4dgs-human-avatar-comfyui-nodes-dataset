import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, kv):
        # 1. Self-Attention: クエリ点同士と human_token で全体像を共有
        q_norm = self.norm1(q)
        attn_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        q = q + attn_out

        # 2. Cross-Attention: 画像特徴量 (kv) から色やテクスチャを抽出
        q_norm2 = self.norm2(q)
        kv_norm = self.norm3(kv)
        attn_out, _ = self.cross_attn(query=q_norm2, key=kv_norm, value=kv_norm, need_weights=False)
        q = q + attn_out

        # 3. FFN: 抽出した特徴を各点で非線形変換して定着
        q = q + self.mlp(self.norm3(q))
        return q


class HumanTransformer(nn.Module):
    def __init__(self, depth, hidden_dim, num_heads, mlp_dim, use_checkpoint=True):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, q, kv):
        for layer in self.layers:
            if self.training and self.use_checkpoint and torch.is_grad_enabled():
                q = checkpoint(layer, q, kv, use_reentrant=False)
            else:
                q = layer(q, kv)
        return self.norm_out(q)
