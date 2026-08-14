from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------
# Utils
# -----------------------
def inverse_sigmoid(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = torch.clamp(y, eps, 1 - eps)
    return torch.log(y / (1 - y))

def trunc_exp(x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    return torch.exp(torch.clamp(x, max=clip))

# -----------------------
# Output container
# -----------------------
@dataclass
class GaussianOutput:
    xyz: torch.Tensor        # [B, N, 3]
    scaling: torch.Tensor    # [B, N, 3]  (positive)
    rotation: torch.Tensor   # [B, N, 4]  (normalized quaternion)
    opacity: torch.Tensor    # [B, N, 1]  (0..1)
    shs: torch.Tensor        # [B, N, K, 3] (K=1 if use_rgb else (sh_degree+1)^2
    use_rgb: bool


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @property
    def first(self):
        return self.net[0]

    @property
    def last(self):
        return self.net[4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------
# Minimal GS Head
# -----------------------
class GSLayer(nn.Module):
    """
    Input:
      - feat: [B, N, D]
      - pts : [B, N, 3]
    Output:
      - GaussianOutput with per-point attributes [B, N, ...]
    """
    def __init__(
        self,
        D: int,
        num_per_group: int = 4,
        use_rgb: bool = False,
        sh_degree: int | None = 2,
        clip_scaling: float =0.02, # 元々0.02でした。
        init_scaling: float = -4.5, # 元々-5.0でした
        init_density: float = 0.7, # 元々0.5でした
        xyz_offset_max_step: float = 1.2 / 32,
        restrict_offset: bool = True,
        fix_opacity: bool = False,
        fix_rotation: bool = False,
        s_min: float = 0.0001, # 元々0.001でした
        s_max: float = 0.03,
    ):
        super().__init__()
        self.use_rgb = use_rgb
        self.clip_scaling = clip_scaling
        self.xyz_offset_max_step = xyz_offset_max_step
        self.restrict_offset = restrict_offset
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation

        # スケール設定例
        self.s_min = s_min  # 1mm
        self.s_max = s_max  # 5cm

        # SH channels
        if use_rgb:
            self.K = 1
            sh_out = 3
        else:
            self.K = (sh_degree + 1) ** 2
            sh_out = self.K * 3

        hidden_dim = D * 2
        self.fc_sh = MLPHead(D, hidden_dim, sh_out)
        self.fc_scaling = MLPHead(D, hidden_dim, 3)
        self.fc_offset = MLPHead(D, hidden_dim, 3)
        self.fc_opacity = None if fix_opacity else MLPHead(D, hidden_dim, 1)
        self.fc_rot = None if fix_rotation else MLPHead(D, hidden_dim, 4)

        # ---- init（元コードの「安定スタート」思想だけ踏襲）----
        nn.init.normal_(self.fc_sh.first.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc_sh.first.bias, 0.0)
        nn.init.normal_(self.fc_sh.last.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc_sh.last.bias, 0.0)


        nn.init.normal_(self.fc_offset.first.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc_offset.first.bias, 0.0)
        nn.init.constant_(self.fc_offset.last.weight, 0.0)
        nn.init.constant_(self.fc_offset.last.bias, 0.0)

        nn.init.normal_(self.fc_scaling.first.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc_scaling.first.bias, 0.0)
        nn.init.constant_(self.fc_scaling.last.weight, 0.0)
        nn.init.constant_(self.fc_scaling.last.bias, init_scaling)

        if not fix_opacity:
            nn.init.normal_(self.fc_opacity.first.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.fc_opacity.first.bias, 0.0)
            nn.init.constant_(self.fc_opacity.last.weight, 0.0)
            b = inverse_sigmoid(torch.tensor(init_density)).item()
            nn.init.constant_(self.fc_opacity.last.bias, b)

        if not fix_rotation:
            nn.init.normal_(self.fc_rot.first.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.fc_rot.first.bias, 0.0)
            nn.init.constant_(self.fc_rot.last.weight, 0.0)
            nn.init.constant_(self.fc_rot.last.bias, 0.0)
            with torch.no_grad():
                self.fc_rot.last.bias.view(4)[0] = 1.0

    def forward(
        self,
        feat: torch.Tensor,
        pts: torch.Tensor,
        scaling_max: torch.Tensor = None,
        offset_max: torch.Tensor = None,
    ) -> GaussianOutput:
        """
        feat: [B,G,D] (クラスタの特徴量)
        pts : [B,N,3] (並び替え済みの全点座標)
        scaling_max: [B,N,1] or [B,N,3] (スケールの最大値)
        offset_max: [B,N,1] or [B,N,3] (offset の絶対値上限)
        """
        assert feat.dim() == 3, f"feat must be [B,N,D], got {feat.shape}"
        assert pts.dim() == 3 and pts.size(-1) == 3, f"pts must be [B,N,3], got {pts.shape}"
        B, N, D = feat.shape
        assert pts.shape[1] == N, f"feat/pts point count mismatch: {N} vs {pts.shape[1]}"

        # 1. 各点の特徴量から属性を独立に出力
    
        # rotation
        if self.fix_rotation:
            rot = feat.new_zeros((B, N, 4))
            rot[..., 0] = 1.0
        else:
            rot = self.fc_rot(feat).reshape(B, N, 4)  # [B, N, 4]
            rot = F.normalize(rot, dim=-1)

        # scaling
        # # 1. softplusという案
        # raw_scaling = self.fc_scaling(feat).reshape(B, N, 3)
        # scaling = F.softplus(raw_scaling) + 1e-6  # positive

        # 2. 上限と下限を決めて、logとsigmoidでマッピングする案
        raw_scaling = self.fc_scaling(feat).reshape(B, N, 3)
        scaling = self.get_scaling(raw_scaling, s_max=scaling_max)

        # 3. 単純にexpして上限をクリップする案 (元コードはこれでした)
        # raw_scaling = self.fc_scaling(feat).reshape(B, N, 3)
        # scaling = trunc_exp(raw_scaling) 
        # if self.clip_scaling is not None:
        #     scaling = torch.clamp(scaling, min=0.0, max=self.clip_scaling)

        # opacity
        if self.fix_opacity:
            opacity = feat.new_ones((B, N, 1))
        else:
            opacity = torch.sigmoid(self.fc_opacity(feat).reshape(B, N, 1))  # [0,1]

        # shs / rgb -> [B, N, K, 3]
        sh = self.fc_sh(feat).reshape(B, N, self.K, 3)  # [B, N, K*3] -> [B, N, K, 3]
        if self.use_rgb:
            sh = torch.sigmoid(sh)  # [0,1]

        # offset -> xyz
        off = self.fc_offset(feat).reshape(B, N, 3)  # [B*N,3]
        if self.restrict_offset:
            if offset_max is not None:
                if offset_max.dim() == 2:
                    offset_max = offset_max.unsqueeze(-1)
                offset_max = torch.clamp(offset_max, min=0.0)
                off = (torch.sigmoid(off) - 0.5) * (2.0 * offset_max)
            else:
                off = (torch.sigmoid(off) - 0.5) * self.xyz_offset_max_step

        # 2. 座標の合成
        xyz = pts + off

        return GaussianOutput(
            xyz=xyz,
            scaling=scaling,
            rotation=rot,
            opacity=opacity,
            shs=sh,
            use_rgb=self.use_rgb,
        )


    def get_scaling(self, raw_scaling, s_max=None):
        log_s_min = math.log(self.s_min)
        log_s_max = math.log(self.s_max)

        if s_max is not None:
            if s_max.dim() == 2:
                s_max = s_max.unsqueeze(-1)
            s_max = torch.clamp(s_max, min=self.s_min)
            log_s_max = torch.log(s_max)

        # 0~1に正規化して、log空間の範囲にマッピング
        t = torch.sigmoid(raw_scaling)
        log_s = log_s_min + (log_s_max - log_s_min) * t
        
        return torch.exp(log_s)
