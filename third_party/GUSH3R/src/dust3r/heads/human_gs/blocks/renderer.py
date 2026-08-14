import math
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

import math
import torch

def getProjectionMatrix(znear, zfar, fovX, fovY, device):
    """
    あなたが貼ってくれた getProjectionMatrix と同一の中身。
    注意: この関数が返す P は「転置前」の行列。
    Cameraクラスはこれを .transpose(0,1) して projection_matrix として持つ。
    """
    tanHalfFovY = math.tan(float(fovY) / 2.0)
    tanHalfFovX = math.tan(float(fovX) / 2.0)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device, dtype=torch.float32)

    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def intrinsic_to_fov(K, H, W):
    """
    K から FoV を復元する。
    FoVx = 2 atan(W/(2fx)), FoVy = 2 atan(H/(2fy))
    """
    fx = K[0, 0]
    fy = K[1, 1]
    fovx = 2.0 * torch.atan(torch.tensor(float(W), device=K.device) / (2.0 * fx))
    fovy = 2.0 * torch.atan(torch.tensor(float(H), device=K.device) / (2.0 * fy))
    return fovx, fovy

class GaussianRenderer:
    """
    Cameraクラス互換のレンダラ。

    入力:
      - gs: GaussianOutput（xyz/scaling/rotation/opacity/shs すべて [B,N,...]）
      - K : [B,3,3] intrinsic
      - w2c: [B,4,4] world->camera（Camera.__init__ の w2c と同じ想定）
      - H, W: 画像サイズ

    出力:
      - comp_rgb   : [B,H,W,3]
      - comp_depth : [B,H,W,1]
      - comp_mask  : [B,H,W,1]
    """

    def __init__(self, sh_degree, scaling_modifier=1.0, force_float32=True,
                 znear=0.01, zfar=100.0):
        self.sh_degree = sh_degree
        self.scaling_modifier = scaling_modifier
        self.force_float32 = force_float32
        # Cameraクラスの default と合わせる（貼ってくれた値）
        self.znear = znear
        self.zfar = zfar

    def _make_bg(self, background_color, device):
        if background_color is None:
            return torch.zeros(3, device=device, dtype=torch.float32)
        return background_color.to(device=device, dtype=torch.float32)

    def _to_raster_dtype(self, x):
        """
        3DGS rasterizer は float32 かつ 'contiguous'（メモリ連続）前提です。
        """
        if x is None:
            return None
        # .contiguous() を必ず付与する
        if self.force_float32:
            return x.float().contiguous()
        return x.contiguous()

    def _build_camera_mats(self, Kb, w2c_b, H, W, device):
        """
        Camera クラスが内部で作る以下を完全に再現する:

          world_view_transform = w2c.transpose(0,1)
          projection_matrix = getProjectionMatrix(...).transpose(0,1)
          full_proj_transform = world_view_transform @ projection_matrix
          camera_center = world_view_transform.inverse()[3,:3]

        注意:
          Cameraは transpose した行列を使うので、
          Renderer側も transpose を踏襲しないと結果が一致しない。
        """
        # 1) FoV を K から作る（Camera.from_c2w と同じ役割）
        fovx, fovy = intrinsic_to_fov(Kb, H, W)

        # 2) world_view_transform（Cameraは w2c を transpose して使う）
        world_view_transform = w2c_b.transpose(0, 1).contiguous()

        # 3) projection_matrix（getProjectionMatrixを作って transpose）
        P = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar,
            fovX=fovx, fovY=fovy,
            device=device
        )
        projection_matrix = P.transpose(0, 1).contiguous()

        # 4) full_proj_transform（Cameraは view @ proj）
        full_proj_transform = (world_view_transform @ projection_matrix).contiguous()

        # 5) camera_center（Cameraの式をそのまま踏襲）
        #    注意: ここは一般的な inv(view)[:3,3] ではない。
        camera_center = torch.inverse(world_view_transform)[3, :3].contiguous()

        return fovx, fovy, world_view_transform, full_proj_transform, camera_center

    def render_batch(self, gs, K, c2w, H, W, background_color=None, ret_mask=True):
        device = gs.xyz.device
        bg = self._make_bg(background_color, device)

        B = gs.xyz.shape[0]
        rgb_list = []
        depth_list = []
        mask_list = []

        for b in range(B):
            # ---- b番目のカメラ行列を Camera互換で構築 ----
            Kb = K[b].to(device)
            c2w_b = c2w[b].to(device)
            w2c_b = torch.inverse(c2w_b)  # [4,4] world->camera
            fovx, fovy, view, full_proj, cam_center = self._build_camera_mats(
                Kb, w2c_b, H, W, device
            )

            # rasterizer settings は Camera.forward_single_view と同じ構成にする
            tanfovx = math.tan(float(fovx) * 0.5)
            tanfovy = math.tan(float(fovy) * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(H),
                image_width=int(W),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg,
                scale_modifier=self.scaling_modifier,
                viewmatrix=view,
                projmatrix=full_proj.float(),   # Camera.full_proj_transform に相当
                sh_degree=self.sh_degree,
                campos=cam_center,
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            # ---- ガウシアン属性を取り出す（[N,*]）----
            means3D = gs.xyz[b]       # [N,3]
            opacity = gs.opacity[b]   # [N,1]
            scales  = gs.scaling[b]   # [N,3]
            rots    = gs.rotation[b]  # [N,4]

            # ---- means2D（互換維持のためのダミー）----
            # 元コードと同様、スクリーンスペース点のgradを取れるようにしておく
            screenspace_points = (
                torch.zeros_like(means3D, dtype=means3D.dtype, device=device, requires_grad=True) + 0
            )
            try:
                screenspace_points.retain_grad()
            except Exception:
                pass

            # ---- 色入力の分岐（RGB直 or SH）----
            if gs.use_rgb:
                colors_precomp = gs.shs[b].squeeze(1).float()  # [N,3]
                shs = None
            else:
                colors_precomp = None
                shs = gs.shs[b].float()                        # [N,K,3]

            # ---- rasterize ----
            # print(f"Rendering batch {b+1}/{B} with {means3D.shape[0]} Gaussians...")
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=self._to_raster_dtype(means3D),
                means2D=self._to_raster_dtype(screenspace_points),
                shs=self._to_raster_dtype(shs),
                colors_precomp=self._to_raster_dtype(colors_precomp),
                opacities=self._to_raster_dtype(opacity),
                scales=self._to_raster_dtype(scales),
                rotations=self._to_raster_dtype(rots),
                cov3D_precomp=None,
            )

            # ---- 出力を [H,W,C] に整形（元コードと同じ）----
            comp_rgb = rendered_image.permute(1, 2, 0)  # [H,W,3]

            # depth/alpha の shape は実装差があるので安全に揃える
            if rendered_depth.dim() == 2:
                comp_depth = rendered_depth.unsqueeze(-1)  # [H,W,1]
            else:
                comp_depth = rendered_depth.permute(1, 2, 0)

            if rendered_alpha.dim() == 2:
                comp_mask = rendered_alpha.unsqueeze(-1)   # [H,W,1]
            else:
                comp_mask = rendered_alpha.permute(1, 2, 0)

            rgb_list.append(comp_rgb)
            depth_list.append(comp_depth)
            mask_list.append(comp_mask)

        # ---- バッチ化 ----
        comp_rgb = torch.stack(rgb_list, dim=0)      # [B,H,W,3]
        comp_depth = torch.stack(depth_list, dim=0)  # [B,H,W,1]
        comp_mask = torch.stack(mask_list, dim=0)    # [B,H,W,1]

        out = {
            "comp_rgb": comp_rgb,
            "comp_depth": comp_depth,
        }
        if ret_mask:
            out["comp_mask"] = comp_mask
        return out
