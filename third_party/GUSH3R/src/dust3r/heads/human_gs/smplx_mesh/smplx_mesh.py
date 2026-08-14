# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from smplx import create as smplx_create
from smplx.lbs import batch_rigid_transform
from plyfile import PlyData, PlyElement

from k_means_constrained import KMeansConstrained
from pytorch3d.ops import knn_points


@dataclass
class QueryPointsOut:
    """Returned by SMPLX_Mesh.get_query_points()."""
    points_apose: torch.Tensor          # [B, N, 3]
    points_zero: torch.Tensor           # [B, N, 3] (Zero poseでのポイント位置)
    transform_mat_apose: torch.Tensor   # [B, J, 4, 4] (Zero -> A-pose への変換行列)
    scaling_max: torch.Tensor           # [B, N, 1]
    offset_max: torch.Tensor            # [B, N, 1]


class SMPLX_Mesh:
    """
    Minimal SMPL-X wrapper for HumanGS.
    """

    def __init__(
        self,
        human_model_path: str,
        shape_param_dim: int = 10,
        expr_param_dim: int = 10,
        cano_pose_type: int = 0,
        sampling_mode: str = "fps",
        sampling_seed: int = 0,
        sampling_dir: str = "dust3r/heads/human_gs/smplx_mesh/sampling_cache",
        device: Optional[torch.device | str] = None,
        num_points: Optional[int] = None,
        num_per_group: Optional[int] = 4
    ):
        self.human_model_path = human_model_path
        self.shape_param_dim = int(shape_param_dim)
        self.expr_param_dim = int(expr_param_dim)
        self.cano_pose_type = int(cano_pose_type)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sampling_mode = sampling_mode
        self.sampling_seed = int(sampling_seed)
        self.sampling_dir = sampling_dir

        # SMPL-X レイヤーの構築
        self.layer_arg = {
            "create_global_orient": False,
            "create_body_pose": False,
            "create_left_hand_pose": False,
            "create_right_hand_pose": False,
            "create_jaw_pose": False,
            "create_leye_pose": False,
            "create_reye_pose": False,
            "create_betas": False,
            "create_expression": False,
            "create_transl": False,
            "use_pca": False,
            "use_face_contour": True,
            "flat_hand_mean": True,
        }

        # 0. ジェンダーごとにレイヤーを作成し、内部で保持する
        self.layer: Dict[str, torch.nn.Module] = {}
        for gender in ["neutral", "male", "female"]:
            self.layer[gender] = smplx_create(
                self.human_model_path,
                model_type="smplx",
                gender=gender,
                num_betas=self.shape_param_dim,
                num_expression_coeffs=self.expr_param_dim,
                **self.layer_arg,
            ).to(self.device)

        # 1. サンプリングインデックスを初期化時に決定して固定
        self.vertex_num = 10475
        self.joint_num = 55
        self.faces = torch.as_tensor(self.layer["neutral"].faces.astype("int64"), device=self.device)
        template_vertices = self.layer["neutral"].v_template.detach()

        if num_points is not None:
            self.sample_indices = self._load_or_create_sample_indices(
                vertices=template_vertices,
                num_points=num_points,
                num_per_group=num_per_group,
            )
        else:
            self.sample_indices = torch.arange(self.vertex_num, device=self.device, dtype=torch.long)
        
        self.num_points = len(self.sample_indices)

        # 1.5 knnを使って、サンプリングポイントに対する最近傍頂点の距離を計算しておく（スケーリングのクリッピングに利用）
        sampled_template = template_vertices[self.sample_indices].unsqueeze(0)  # [1, N, 3]
        knn = knn_points(sampled_template, sampled_template, K=4) # [1, N, 4]
        nn_dist = torch.sqrt(knn.dists[0, :, 1:].clamp(min=1e-12)).mean(dim=-1)  # [N]
        self.s_max = 1.5 * nn_dist # [N]
        self.offset_max = 0.25 * nn_dist # [N]

        # 2. LBSウェイトを出力せず、内部で保持する
        self.lbs_weights: Dict[str, torch.Tensor] = {}
        for gender in ["neutral", "male", "female"]:
            w_full = self.layer[gender].lbs_weights.to(self.device, dtype=torch.float32)
            self.lbs_weights[gender] = w_full[self.sample_indices]  # [N, J]

        # 3. Neutral Body Pose (A-pose用)
        self.neutral_body_pose = torch.zeros((21, 3), dtype=torch.float32)
        if self.cano_pose_type == 0:
            self.neutral_body_pose[0] = torch.tensor([0.0, 0.0, 1.0])
            self.neutral_body_pose[1] = torch.tensor([0.0, 0.0, -1.0])
        else:
            self.neutral_body_pose[0] = torch.tensor([0.0, 0.0, math.pi / 9.0])
            self.neutral_body_pose[1] = torch.tensor([0.0, 0.0, -math.pi / 9.0])

        self.neutral_jaw_pose = torch.tensor([1.0 / 3.0, 0.0, 0.0], dtype=torch.float32)

    def _get_sampling_cache_path(self, num_points: int, num_per_group: int) -> str:
        filename = (
            f"smplx_sampling_"
            f"{self.sampling_mode}_"
            f"seed{self.sampling_seed}_"
            f"verts{self.vertex_num}_"
            f"points{num_points}_"
            f"group{num_per_group}.pt"
        )
        return os.path.join(self.sampling_dir, filename)

    def _load_or_create_sample_indices(
        self,
        vertices: torch.Tensor,
        num_points: int,
        num_per_group: int,
    ) -> torch.Tensor:
        cache_path = self._get_sampling_cache_path(num_points, num_per_group)
        os.makedirs(self.sampling_dir, exist_ok=True)

        if os.path.isfile(cache_path):
            payload = torch.load(cache_path, map_location="cpu")
            sample_indices = payload["sample_indices"].to(device=self.device, dtype=torch.long)
            if sample_indices.numel() != num_points:
                raise ValueError(
                    f"Cached sample_indices has {sample_indices.numel()} points, expected {num_points}: {cache_path}"
                )
            print(f"Loaded cached sample indices from {cache_path}")
            return sample_indices

        if self.sampling_mode == "fps":
            sample_indices = self._get_constrained_indices(vertices, num_points, num_per_group)
        elif self.sampling_mode == "uniform":
            sample_indices = self._get_uniform_constrained_indices(vertices, num_points, num_per_group)
        else:
            raise ValueError(f"Unsupported sampling_mode: {self.sampling_mode}")

        torch.save(
            {
                "sample_indices": sample_indices.detach().cpu(),
                "sampling_mode": self.sampling_mode,
                "sampling_seed": self.sampling_seed,
                "vertex_num": self.vertex_num,
                "num_points": num_points,
                "num_per_group": num_per_group,
            },
            cache_path,
        )
        print(f"Saved sample indices to cache: {cache_path}")
        return sample_indices

    def _get_uniform_constrained_indices(
        self,
        vertices: torch.Tensor,
        num_points: int,
        num_per_group: int = 4,
    ) -> torch.Tensor:
        """
        1. Uniform sampling で点を選ぶ
        2. k-means-constrained で num_per_group ごとにグループ化
        3. ラベル順に並べて返す
        """
        uniform_indices = self._uniform_point_sampling(vertices, num_points)
        sampled_coords = vertices[uniform_indices].cpu().numpy()

        num_clusters = num_points // num_per_group
        clf = KMeansConstrained(
            n_clusters=num_clusters,
            size_min=num_per_group,
            size_max=num_per_group,
            random_state=0,
            n_jobs=-1,
        )
        labels = clf.fit_predict(sampled_coords)

        sort_idx = np.argsort(labels)
        sorted_uniform_indices = uniform_indices[sort_idx]
        return sorted_uniform_indices


    def _uniform_point_sampling(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Uniform sampling over vertex indices.
        vertices: [V, 3]
        returns: sampled vertex indices [num_points]
        """
        V = vertices.shape[0]
        device = vertices.device

        if num_points > V:
            raise ValueError(f"num_points={num_points} exceeds number of vertices V={V}")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.sampling_seed)
        perm = torch.randperm(V, generator=generator)
        return perm[:num_points].to(device=device, dtype=torch.long)

    def _get_constrained_indices(self, vertices: torch.Tensor, num_points: int, num_per_group: int = 4) -> torch.Tensor:
            """
            1. FPSで広範囲にサンプリング
            2. k-means-constrainedで4点ずつのグループに分ける
            3. ラベル順にインデックスをソートして返す
            """
            # 1. FPSを実行してベースとなる4000点を選出
            fps_indices = self._farthest_point_sampling(vertices, num_points)
            sampled_coords = vertices[fps_indices].cpu().numpy() # [4000, 3]

            # 2. クラスタリングの実行 (サイズを厳密に num_per_group に固定)
            num_clusters = num_points // num_per_group
            clf = KMeansConstrained(
                n_clusters=num_clusters,
                size_min=num_per_group,
                size_max=num_per_group,
                random_state=0,
                n_jobs=-1 # 並列処理
            )
            labels = clf.fit_predict(sampled_coords)

            # 3. ラベルに基づいて fps_indices を並び替える
            # labelsが [2, 0, 1, 0, 2, 1, ...] のようになっているので、
            # これを 0,0,0,0, 1,1,1,1, ... になるようにソート
            sort_idx = np.argsort(labels)
            sorted_fps_indices = fps_indices[sort_idx]

            return sorted_fps_indices
    
    def _farthest_point_sampling(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Farthest Point Sampling (最遠点サンプリング)
        vertices: [V, 3] のテンソル (V=10475)
        num_points: サンプリングする点数
        """
        V = vertices.shape[0]
        device = vertices.device

        fps_indices = torch.zeros(num_points, dtype=torch.long, device=device)
        # 距離を無限大で初期化
        distances = torch.full((V,), float('inf'), device=device)

        # 最初の1点はランダムに選択
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.sampling_seed)
        current_idx = torch.randint(0, V, (1,), generator=generator)[0].to(device=device)

        for i in range(num_points):
            fps_indices[i] = current_idx
            
            # 選ばれた点と全頂点との距離の2乗を計算
            current_point = vertices[current_idx]
            dist_to_current = torch.sum((vertices - current_point) ** 2, dim=-1)
            
            # 各頂点について、「これまで選ばれた点群との最短距離」を更新
            distances = torch.min(distances, dist_to_current)
            
            # 次の点は、既存の点群からの最短距離が「最も遠い」点を選ぶ
            current_idx = torch.argmax(distances)

        return fps_indices

    def extract_smplx_params(self, B: int, smplx_data: dict) -> torch.Tensor:
        """Helper: SMPLパラメータを連結して [B, 55, 3] の pos を作成"""
        def get_p(keys, dim1):
            if isinstance(keys, str):
                keys = (keys,)
            val = None
            for key in keys:
                val = smplx_data.get(key, None)
                if val is not None:
                    break
            if val is None:
                return torch.zeros((B, dim1, 3), device=self.device)
            return val.to(self.device, dtype=torch.float32).view(B, dim1, 3)

        global_orient = get_p(("root_pose", "global_orient"), 1)
        body_pose = get_p("body_pose", 21)
        jaw_pose = get_p("jaw_pose", 1)
        leye_pose = get_p("leye_pose", 1)
        reye_pose = get_p("reye_pose", 1)
        left_hand_pose = get_p(("lhand_pose", "left_hand_pose"), 15)
        right_hand_pose = get_p(("rhand_pose", "right_hand_pose"), 15)

        return torch.cat([global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose], dim=1)

    def get_joints_and_transform(self, layer, betas, expression, pose) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zero poseでの関節位置を取得し、そこから入力ポーズへの4x4変換行列を順運動学で計算する。
        """
        B = betas.shape[0]
        # Zero Pose で forward して、T-pose(レスト状態) のジョイント位置を取得
        zero_pose = torch.zeros((B, 55, 3), device=self.device)
        out_zero = layer(
            betas=betas,
            expression=expression,
            global_orient=zero_pose[:, 0],
            body_pose=zero_pose[:, 1:22],
            jaw_pose=zero_pose[:, 22],
            leye_pose=zero_pose[:, 23],
            reye_pose=zero_pose[:, 24],
            left_hand_pose=zero_pose[:, 25:40],
            right_hand_pose=zero_pose[:, 40:55],
        )
        J_rest = out_zero.joints[:, :55, :].contiguous()

        # 正規の順運動学による変換行列の計算
        pose_mat = axis_angle_to_matrix(pose)
        parents = layer.parents[:55].to(self.device)
        posed_joints, T_forward = batch_rigid_transform(pose_mat, J_rest, parents)
        return posed_joints, T_forward

    @torch.no_grad()
    def get_query_points(
        self,
        betas: torch.Tensor,
        gender: str = "neutral",
        expression: Optional[torch.Tensor] = None,
        jaw_zero_pose: bool = True,
    ) -> QueryPointsOut:
        """固定インデックスでA-poseのサンプリングポイントと変換行列を返す"""
        layer = self.layer[gender]
        betas = betas.to(self.device, dtype=torch.float32)
        B = betas.shape[0]

        if expression is None:
            expression = torch.zeros((B, self.expr_param_dim), device=self.device, dtype=torch.float32)
        else:
            expression = expression.to(self.device, dtype=torch.float32)

        # A-pose用データのセットアップ
        smplx_data_A = {
            "body_pose": self.neutral_body_pose.to(self.device).unsqueeze(0).expand(B, -1, -1),
        }
        pose_A = self.extract_smplx_params(B, smplx_data_A)

        # A-pose頂点の計算
        out = layer(
            betas=betas,
            expression=expression,
            global_orient=pose_A[:, 0],
            body_pose=pose_A[:, 1:22],
            jaw_pose=pose_A[:, 22],
            leye_pose=pose_A[:, 23],
            reye_pose=pose_A[:, 24],
            left_hand_pose=pose_A[:, 25:40],
            right_hand_pose=pose_A[:, 40:55],
        )
        verts_apose_full = out.vertices # [B, 10475, 3]
        points_apose = verts_apose_full[:, self.sample_indices, :]  # [B, N, 3]

        # Zero poseでのポイント位置を計算
        smplx_data_zero = {
            "body_pose": torch.zeros((B, 21, 3), device=self.device),
        }
        pose_zero = self.extract_smplx_params(B, smplx_data_zero)
        out_zero = layer(
            betas=betas,
            expression=expression,
            global_orient=pose_zero[:, 0],
            body_pose=pose_zero[:, 1:22],
            jaw_pose=pose_zero[:, 22],
            leye_pose=pose_zero[:, 23],
            reye_pose=pose_zero[:, 24],
            left_hand_pose=pose_zero[:, 25:40],
            right_hand_pose=pose_zero[:, 40:55],
        )
        points_zero = out_zero.vertices[:, self.sample_indices, :] # [B, N, 3]

        # Zero -> A-pose への変換行列を取得
        _, T_A = self.get_joints_and_transform(layer, betas, expression, pose_A)

        return QueryPointsOut(
            points_apose=points_apose,
            points_zero=points_zero,
            transform_mat_apose=T_A,
            scaling_max=self.s_max.to(device=self.device, dtype=torch.float32).view(1, -1, 1).expand(B, -1, 1),
            offset_max=self.offset_max.to(device=self.device, dtype=torch.float32).view(1, -1, 1).expand(B, -1, 1),
        )

    @torch.no_grad()
    def get_target_transform(self, smplx_data: dict, gender: str = "neutral", return_head: bool = True) -> torch.Tensor:
        """
        Target Pose のパラメータを受け取り、Zero -> Target への変換行列(T_P)を返す。
        Returns: [B, 55, 4, 4]
        """
        layer = self.layer[gender]
        betas = smplx_data.get("betas").to(self.device, dtype=torch.float32)
        expression = smplx_data.get("expr", smplx_data.get("expression"))
        if expression is None:
            expression = torch.zeros((betas.shape[0], self.expr_param_dim), device=self.device, dtype=torch.float32)
        if expression is not None:
            expression = expression.to(self.device, dtype=torch.float32)
        
        B = betas.shape[0]
        pose = self.extract_smplx_params(B, smplx_data)
        
        # Zero -> Target への変換行列を取得
        posed_points, T_P = self.get_joints_and_transform(layer, betas, expression, pose)
        if return_head:
            head_idx = 15
            head_posed = posed_points[:, head_idx, :]
            return T_P, head_posed, posed_points
        else:
            _, T_P = self.get_joints_and_transform(layer, betas, expression, pose)
            return T_P, posed_points

    def pose_points_from_zero(self, points_zero, T_P, gender: str = "neutral", smplx_data=None):
        """
        Zero-pose (T-pose) のポイントを Target Pose へ LBS を使って変形する。
        """
        device = points_zero.device
        dtype = points_zero.dtype
        W = self.lbs_weights[gender].to(device=device, dtype=dtype) # [N, J]
        T_P = T_P.to(device=device, dtype=dtype)

        # 頂点ごとの変換行列をブレンド: Σ w_ij T_P_j
        T_vertex = torch.einsum("bjac,nj->bnac", T_P, W)  # [B, N, 4, 4]

        # ポイントに変換行列を適用
        xyz1 = torch.cat((points_zero, torch.ones_like(points_zero[..., :1])), dim=-1)  # [B, N, 4]
        xyz_posed = torch.einsum("bnac, bnc -> bna", T_vertex, xyz1)[..., :3]     # [B, N, 3]

        if smplx_data is not None:
            trans = smplx_data.get("trans", smplx_data.get("transl"))
            head_posed = smplx_data.get("head_posed")
            if trans is not None:
                add_t = trans if head_posed is None else (trans - head_posed)
                xyz_posed = xyz_posed + add_t.unsqueeze(1)

        return xyz_posed


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
    # Quick test
    human_model_path = "/home/abe/project/experiment/human-scene/Human3R/src/models/"
    
    # Init時に点数を固定 (ここで内部的にインデックスとウェイトが固定されます)
    smplx_mesh = SMPLX_Mesh(human_model_path, cano_pose_type=1, num_points=1000)
    
    # A-poseの取得
    betas_fat = torch.randn((2, 11)) * 4.0
    out = smplx_mesh.get_query_points(betas=betas_fat)
    
    print("points_apose shape:", out.points_apose.shape)        # [2, 1000, 3]
    print("transform_mat_apose:", out.transform_mat_apose.shape)  # [2, 55, 4, 4]

    save_xyz_ply("test_points_apose.ply", out.points_apose[0])

    # Target pose の適用テスト
    smpl_data_target = {
        "betas": betas_fat,
        "global_orient": torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]), # ちょっとお辞儀させるなど
        "body_pose": torch.zeros((2, 21, 3))
    }
    
    # ターゲットの4x4変換行列を取得
    T_P = smplx_mesh.get_target_transform(smpl_data_target)
    
    # A-pose頂点をTarget poseに動かす
    posed_points = smplx_mesh.apose_points_to_pose(out.points_apose, out.transform_mat_apose, T_P)
    print("posed_points shape:", posed_points.shape)            # [2, 1000, 3]
    save_xyz_ply("test_points_posed.ply", posed_points[0])
