import torch
import torch.nn as nn
import torch.nn.functional as F


from accelerate.logging import get_logger
from torch.utils.checkpoint import checkpoint

from dust3r.heads.human_gs.smplx_mesh.smplx_mesh import SMPLX_Mesh
from dust3r.heads.human_gs.smplx_mesh.point_embed import PointEmbed
from dust3r.heads.human_gs.blocks.dit import DiT_S
from dust3r.heads.human_gs.blocks.cross_attention import HumanTransformer
from dust3r.heads.human_gs.blocks.self_attention import HumanSelfTransformer
from dust3r.heads.human_gs.blocks.gs_layer import GSLayer
from dust3r.heads.human_gs.blocks.renderer import GaussianRenderer

logger = get_logger(__name__)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()
        # 3層（Linear 3回）の構成
        # [in_dim] -> [hidden] -> [hidden] -> [out_dim]
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # 1層目
        x = self.layer1(x)
        # 2層目（1層目の出力とhidden_dimが同じなので、残差接続が可能）
        x = x + self.layer2(x) 
        # 3層目（最終出力）
        x = self.layer3(x)
        return x
    
class HumanGSHead(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg

        self.hidden_dim = 512 # 1トークンの次元
        self.image_token_dim = 768
        self.raw_feature_dim = 128
        self.smpl_query_dim = 1792
        self.num_per_group = 10  # SMPLXの各頂点を何個のガウシアンで表現するか
        self.num_groups = 1000 # SMPLXメッシュのグループの数（=1000点を選ぶ）
        self.num_points = self.num_per_group * self.num_groups  # ガウシアンの総数


        # 1. SMPlXメッシュの定義
        smplx_model_path = getattr(cfg, "human_model_path", "src/models") if cfg is not None else "src/models"
        if cfg is not None:
            sampling_mode = getattr(cfg, "human_gs_sampling_mode", "fps")
            sampling_seed = getattr(cfg, "human_gs_sampling_seed", 0)
            sampling_dir = getattr(
                cfg,
                "human_gs_sampling_dir",
                "dust3r/heads/human_gs/smplx_mesh/sampling_cache",
            )
        else:
            sampling_mode = "fps"
            sampling_seed = 0
            sampling_dir = "dust3r/heads/human_gs/smplx_mesh/sampling_cache"
        self.smplx_mesh = SMPLX_Mesh(smplx_model_path, cano_pose_type=1, num_points=self.num_points, num_per_group=self.num_per_group,
                                     sampling_mode=sampling_mode, sampling_seed=sampling_seed, sampling_dir=sampling_dir)

        # 2. Point Embedの定義
        self.point_embed = PointEmbed(hidden_dim=48, dim=128)
        self.point_embed_dim = self.point_embed.mlp.out_features

        # 3. Transformerの実装
        self.proj_human = nn.Linear(self.smpl_query_dim, self.hidden_dim) # SMPLクエリをそのまま通すためのプロジェクション
        self.proj_image = nn.Linear(self.image_token_dim, self.hidden_dim) # DINO特徴をGS Layerに渡す前に圧縮するためのプロジェクション
        self.proj_query = nn.Linear(128, self.hidden_dim) # クエリ特徴をGS Layerに渡す前にプロジェクション
        self.raw_image_merger = nn.Sequential(
            nn.Conv2d(3, self.raw_feature_dim, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
        )
        self.raw_image_proj = nn.Linear(self.raw_feature_dim, self.image_token_dim)
        self.human_gs_transformer_type = getattr(
            cfg, "human_gs_transformer_type", "cross_attention"
        ) if cfg is not None else "cross_attention"
        transformer_cls_by_type = {
            "cross": HumanTransformer,
            "cross_attention": HumanTransformer,
            "cross_no_human": HumanTransformer,
            "cross_no_human_token": HumanTransformer,
            "cross_attention_no_human": HumanTransformer,
            "cross_attention_no_human_token": HumanTransformer,
            "self": HumanSelfTransformer,
            "self_attention": HumanSelfTransformer,
        }
        if self.human_gs_transformer_type not in transformer_cls_by_type:
            raise ValueError(
                "human_gs_transformer_type must be one of "
                f"{sorted(transformer_cls_by_type)}, got {self.human_gs_transformer_type!r}."
            )
        transformer_cls = transformer_cls_by_type[self.human_gs_transformer_type]
        self.human_transformer = transformer_cls(
            depth=4,
            hidden_dim=self.hidden_dim,
            num_heads=8,
            mlp_dim=self.hidden_dim * 2,
        )
        self.use_human_token = self.human_gs_transformer_type not in {
            "cross_no_human",
            "cross_no_human_token",
            "cross_attention_no_human",
            "cross_attention_no_human_token",
        }

        # 5. GS Layerの定義
        self.gs_layer = GSLayer(
            D=self.hidden_dim + self.point_embed_dim,
            num_per_group=self.num_per_group,
            use_rgb=True,
            sh_degree=0,
        )
        # 6. GS Layerの出力をレンダリングするためのRenderer
        self.gs_renderer = GaussianRenderer(
            sh_degree=0,
        )

        # 7. 人間メモリートークンを追加
        self.num_human_memory_tokens = int(
            getattr(cfg, "human_gs_num_memory_tokens", 0) if cfg is not None else 0
        )
        if self.num_human_memory_tokens < 0:
            raise ValueError("human_gs_num_memory_tokens must be non-negative.")
        if self.num_human_memory_tokens > 0:
            self.init_human_memory = nn.Parameter(
                torch.empty(self.num_human_memory_tokens, self.hidden_dim)
            )
            nn.init.trunc_normal_(self.init_human_memory, std=0.02)
            # https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_
        else:
            self.register_parameter("init_human_memory", None)

        # LHMに入れる特徴量を作る
        # 3層MLPの定義
        # self.human_feat_extractor = MLP(in_dim=self.smpl_query_dim + 64, out_dim=1536, hidden_dim=2048)
        # self.motion_feat_extractor = MLP(in_dim=self.smpl_query_dim, out_dim=2048, hidden_dim=2048)

    def get_initial_human_memory(self, batch_size, max_humans, device, dtype):
        if self.num_human_memory_tokens == 0:
            return None
        memory = self.init_human_memory.to(device=device, dtype=dtype)
        return memory.view(1, 1, self.num_human_memory_tokens, self.hidden_dim).expand(
            batch_size,
            max_humans,
            -1,
            -1).clone()  # [B, H, num_human_memory_tokens, hidden_dim]

    def _get_patch_grid_shape(self, raw_image, patch_size):
        if patch_size is None:
            raise ValueError("patch_size is required for raw-image/token alignment.")

        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size

        raw_h, raw_w = raw_image.shape[-2:]
        if raw_h % patch_h != 0 or raw_w % patch_w != 0:
            raise ValueError(
                f"raw_image spatial size {(raw_h, raw_w)} must be divisible by patch_size {(patch_h, patch_w)}."
            )
        return raw_h // patch_h, raw_w // patch_w

    def _fuse_raw_image_tokens(self, image_tokens, raw_image, patch_size):
        if raw_image is None:
            return image_tokens

        grid_h, grid_w = self._get_patch_grid_shape(raw_image, patch_size)
        _, num_tokens, token_dim = image_tokens.shape
        expected_tokens = grid_h * grid_w

        if num_tokens != expected_tokens:
            raise ValueError(
                f"image_tokens has {num_tokens} tokens, expected {expected_tokens} from raw_image and patch_size."
            )
        if token_dim != self.image_token_dim:
            raise ValueError(f"Expected image token dim {self.image_token_dim}, got {token_dim}")

        raw_feat = self.raw_image_merger(raw_image)
        raw_feat = F.adaptive_avg_pool2d(raw_feat, (grid_h, grid_w))
        raw_feat = raw_feat.flatten(2).transpose(1, 2)
        raw_feat = self.raw_image_proj(raw_feat)
        return image_tokens + raw_feat

    def forward(
        self,
        smpl_query,
        image_tokens,
        smpl_params,
        extrinsics,
        intrinsics,
        height=512,
        width=512,
        background_color=None,
        debug_image=None,
        debug_path="debug_joint.png",
        raw_image=None,
        patch_size=None,
        human_memory=None,
        render=True,
    ):
        """
        SMPLクエリとSMPLパラメータからそのポーズでのガウシアンを生成する。
        一人分のデータを処理する想定で、バッチ次元はないものとする。
        Args:
            smpl_query: [B*H, 1792] SMPLクエリ特徴量。Bはバッチサイズ、Hはシーン内の人間の数。1792はSMPLクエリの特徴量の次元数。
            image_tokens: [B*H, N, 768] イメージトークン。DINO/CUT3R特徴マップ。
            smpl_params: dictで、SMPLの各種パラメータを含む。例えば、'betas': [B*H, 10] など。必要なパラメータはLHMのrendererが決める。
            extrinsics: [B*H, 4, 4] c2w のカメラ外部パラメータ
            intrinsics: [B*H, 3, 3] カメラの内部パラメータ
            debug_image: 可視化用の元画像。与えた場合だけ debug_path に関節射影を保存する。
            debug_path: 関節射影の可視化画像の保存先。
        Returns:
        """
        if smpl_query.shape[-1] != self.smpl_query_dim:
            raise ValueError(
                f"Expected smpl_query dim {self.smpl_query_dim}, got {smpl_query.shape[-1]}"
            )

        # 0. shapeの確認
        # print(f"smpl_query: {smpl_query.shape}, dino_query: {dino_query.shape}, smpl_params['betas']: {smpl_params['betas'].shape}, extrinsics: {extrinsics.shape}, intrinsics: {intrinsics.shape}")
        
        # 1. クエリ点を取得
        betas = smpl_params['betas']  # [B*H, shape_param_dim]
        out = self.smplx_mesh.get_query_points(betas=betas)
        point_embed_weight = self.point_embed.mlp.weight
        query_points = out.points_apose.to(
            device=point_embed_weight.device,
            dtype=point_embed_weight.dtype,
        )  # [B*H, num_points(=1000*4), 3]
        query_points_zero = out.points_zero.to(
            device=point_embed_weight.device,
            dtype=point_embed_weight.dtype,
        )  # [B*H, num_points(=1000*4), 3]
        point_s_max = out.scaling_max.to(
            device=point_embed_weight.device,
            dtype=point_embed_weight.dtype,
        )  # [B*H, num_points(=1000*4), 1]
        point_offset_max = out.offset_max.to(
            device=point_embed_weight.device,
            dtype=point_embed_weight.dtype,
        )  # [B*H, num_points(=1000*4), 1]

        # 2. クエリ点の頂点埋め込みを実施
        query_means = query_points.reshape(-1, self.num_groups, self.num_per_group, 3).mean(dim=2)  # [B*H, num_groups(=1000), 3]
        query_points_embed = self.point_embed(query_means)  # [B*H, num_groups(=1000), 128]
        point_features = self.point_embed(query_points_zero)  # [B*H, num_points, 128]


        # 3. LBSでガウシアンを変形してガウシアンの中心を求める
        T_P, head_posed, posed_joints = self.smplx_mesh.get_target_transform(smplx_data=smpl_params)  
        # [B*H, num_points, 4, 4], [B*H, 3], [B*H, num_points, 3]
        smpl_params["head_posed"] = head_posed  # [B*H, 3] 頭部の位置を追加で保存
        posed_query_points = self.smplx_mesh.pose_points_from_zero(
            points_zero=query_points_zero,  # [B*H, num_points, 3]
            T_P=T_P,  # [B*H, num_points, 4, 4]
            smplx_data=smpl_params
        ) # [B*H, num_points, 3]

        # 4. Human transformer
        # クエリ: [B*H, 1+num_groups, 128]
        # - human_query: [B*H, 1792]
        # - query_feats: [B*H, num_groups, 128]
        # cross-attention mode only uses image_tokens as key/value: [B*H, H'*W', 768]
        # print("shape before projection", posed_query_points.shape, query_points_embed.shape, smpl_query.shape, image_tokens.shape)
        q_query = self.proj_query(query_points_embed) # [B*H, num_groups, hidden_dim]
        q_parts = []
        if self.use_human_token:
            q_human = self.proj_human(smpl_query) # [B*H, hidden_dim]
            q_parts.append(q_human.unsqueeze(1))
        # メモリーもqとして追加。0 tokenなら旧来どおり human token + query token のみ。
        if self.num_human_memory_tokens > 0:
            if human_memory is None:
                human_memory = self.init_human_memory.to(
                    device=q_query.device, dtype=q_query.dtype
                ).view(1, self.num_human_memory_tokens, self.hidden_dim).expand(
                    q_query.shape[0], -1, -1
                )
            expected_memory_shape = (
                q_query.shape[0],
                self.num_human_memory_tokens,
                self.hidden_dim,
            )
            if human_memory.shape != expected_memory_shape:
                raise ValueError(
                    f"Expected human_memory shape {expected_memory_shape}, "
                    f"got {tuple(human_memory.shape)}."
                )
            q_parts.append(human_memory)
        q_parts.append(q_query)
        q = torch.cat(q_parts, dim=1) # [B*H, human?+M?+num_groups, hidden_dim]
        if self.human_gs_transformer_type in ("self", "self_attention"):
            kv = None
        else:
            image_tokens = self._fuse_raw_image_tokens(
                image_tokens=image_tokens,
                raw_image=raw_image,
                patch_size=patch_size,
            )
            kv = self.proj_image(image_tokens) # [B*H, H'*W', hidden_dim]
        # print("shape after projection", q_human.shape, q_query.shape, kv.shape)
        transformer_out = self.human_transformer(q, kv) # [B*H, 1+num_groups, hidden_dim]

        # 取り出しも変わる
        query_start = 1 if self.use_human_token else 0
        if self.num_human_memory_tokens > 0:
            memory_start = query_start
            memory_end = memory_start + self.num_human_memory_tokens
            updated_human_memory = transformer_out[:, memory_start:memory_end, :]  # [B*H, num_human_memory_tokens, hidden_dim]
            query_start = memory_end
        else:
            updated_human_memory = None
        query_feats = transformer_out[:, query_start:, :]  # [B*H, num_groups, hidden_dim]
        # print("shape after transformer", query_feats.shape)

        # 5. GS Layerでガウシアンを生成
        # feat: [B*H, num_points, hidden_dim + point_embed_dim]
        # pts: [B*H, num_points, 3]
        expanded_query_feats = query_feats.unsqueeze(2).expand(
            -1, -1, self.num_per_group, -1
        ).reshape(query_feats.shape[0], self.num_points, self.hidden_dim)
        gs_features = torch.cat([expanded_query_feats, point_features], dim=-1)
        gaussians = self.gs_layer(
            feat=gs_features,
            pts=posed_query_points,
            # scaling_max=point_s_max,
            # offset_max=point_offset_max,
        )  # GaussianOutput (offset_xyz, opacity, rotation, scaling, shs)

        if (not render) or extrinsics is None or intrinsics is None:
            return {
                "gaussians": gaussians,
                "updated_human_memory": updated_human_memory,
            }
        
        # 6. GS Layerの出力をレンダリングして結果を返す
        render_results = self.gs_renderer.render_batch(
            gs=gaussians,
            K=intrinsics,
            c2w=extrinsics,
            H=height,
            W=width,
            background_color=background_color,
        )
        render_results["gaussians"] = gaussians  # ガウシアンの出力も一緒に返す
        render_results["updated_human_memory"] = updated_human_memory  # 更新された人間メモリーも返す
        return render_results

from plyfile import PlyData, PlyElement
import numpy as np
def save_xyz_ply(path, xyz):
    xyz_np = xyz.detach().float().reshape(-1, 3).cpu().numpy()
    if xyz_np.size == 0:
        return
    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    vertices = np.empty(xyz_np.shape[0], dtype=vertex_dtype)
    vertices["x"] = xyz_np[:, 0]
    vertices["y"] = xyz_np[:, 1]
    vertices["z"] = xyz_np[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)

if __name__ == "__main__":
    # 1. デバイスの定義
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. モデルを初期化して GPU へ転送
    human_gs_head = HumanGSHead(cfg=None).to(device)

    # 3. ダミーデータを GPU 上で生成
    dummy_smpl_query = torch.randn((2, 1792), device=device)
    dummy_dino_query = torch.randn((2, 1024), device=device)
    body_pose = torch.zeros((2, 21, 3), device=device)
    body_pose[:, 1, 0] = 0.5
    body_pose[:, 2, 0] = -0.5
    shape = torch.randn((2, 11), device=device)
    shape[:, 0] = 0.0
    shape[:, 1] = -0.0
    
    dummy_smpl_params = {
        'betas': shape,
        'body_pose': body_pose,
        'trans': torch.zeros((2, 3), device=device)
    }

    # 【修正箇所】現実的なカメラの内部パラメータ（fx=500, fy=500, cx=256, cy=256）
    K_real = torch.tensor([
        [500.0,   0.0, 256.0],
        [  0.0, 500.0, 256.0],
        [  0.0,   0.0,   1.0]
    ], device=device)
    dummy_intrinsics = K_real.unsqueeze(0).repeat(2, 1, 1)

    c2w_real = torch.eye(4, device=device)
    c2w_real[0, 0] = -1.0  # Xを反転
    c2w_real[1, 1] = -1.0  # Yを反転（これで正立する）
    c2w_real[2, 3] = -2.5  # Z=-2.5 の位置から前を見る
    dummy_extrinsics = c2w_real.unsqueeze(0).repeat(2, 1, 1)
    
    # 4. 推論実行
    render_results = human_gs_head(
        smpl_query=dummy_smpl_query,
        dino_query=dummy_dino_query,
        smpl_params=dummy_smpl_params,
        extrinsics=dummy_extrinsics,
        intrinsics=dummy_intrinsics
    )
    
    import numpy as np
    from PIL import Image

    # RGBの保存 (色はランダムノイズやグレー/真っ黒になる可能性大)
    img_rgb = (render_results['comp_rgb'][0].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_rgb).save("sample_rgb.png")

    # マスクの保存 (形さえ合っていれば、ここに確実に人間のシルエットが白く浮き出る！)
    img_mask = (render_results['comp_mask'][0].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_mask.squeeze(-1), mode='L').save("sample_mask.png")

# CUDA_VISIBLE_DEVICES=7 python dust3r/heads/human_gs/human_gs_head.py
