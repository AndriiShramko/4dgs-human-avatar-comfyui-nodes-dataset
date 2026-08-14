# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Features                #
#################################################################################


class FeatureEmbedder(nn.Module):
    """
    連続値の特徴量ベクトルを受け取り、DiTの隠れ層次元（hidden_size）に射影するクラス。
    """
    def __init__(self, in_features, hidden_size):
        super().__init__()
        # 特徴量をDiTの次元に合わせるためのMLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, features):
        """
        features: [B, in_features] の画像特徴量テンソル
        """
        # [B, in_features] -> [B, hidden_size]
        embeddings = self.mlp(features)
        return embeddings

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=128,
        out_channels=100,
        hidden_size=1152,
        human_embed_dim=1792,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.point_embedder = nn.Linear(in_channels, hidden_size)
        self.h_embedder = FeatureEmbedder(in_features=human_embed_dim, hidden_size=hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize human feature embedding MLP:
        nn.init.normal_(self.h_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.h_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.xavier_uniform_(self.final_layer.linear.weight, gain=0.01)
        nn.init.constant_(self.final_layer.linear.bias, 0.0)


    def forward(self, x, h):
        """
        Forward pass of DiT.
        x: (B, N, in_channels) tensor of 3D query points (after embedding)
        h: (B, human_embed_dim) tensor of human features (after embedding)
        """
        x = self.point_embedder(x)                # (B, N, D)
        h = self.h_embedder(h)                   # (B, D)
        for block in self.blocks:
            x = block(x, h)                      # (B, N, D)
        x = self.final_layer(x, h)                # (B, N, out_channels)
        return x

    def forward_with_cfg(self, x, h, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined_x = torch.cat([half, half], dim=0)
        
        model_out = self.forward(combined_x, h)
        
        cond_out, uncond_out = torch.split(model_out, len(model_out) // 2, dim=0)
        half_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        return torch.cat([half_out, half_out], dim=0)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=2, hidden_size=384, num_heads=6, **kwargs)

DiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L':  DiT_L,
    'DiT-B':  DiT_B,
    'DiT-S':  DiT_S,
}