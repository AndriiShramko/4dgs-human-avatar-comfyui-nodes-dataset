# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from dust3r.heads.postprocess import (
    postprocess,
    postprocess_desc,
    postprocess_rgb,
    postprocess_pose_conf,
    postprocess_pose,
    reg_dense_conf,
    postprocess_smpl,
    postprocess_score,
)
import dust3r.utils.path_to_croco  # noqa: F401
from models.dpt_block import DPTOutputAdapter, Interpolate  # noqa
from dust3r.utils.camera import pose_encoding_to_camera, PoseDecoder
from dust3r.blocks import ConditionModulationBlock
from torch.utils.checkpoint import checkpoint
from dust3r.smpl_model import SMPLDecoder, MEAN_PARAMS, regression_mlp
import numpy as np
import torch.nn.functional as F

from dust3r.heads.gaussian_utils.gaussian_adapter import UnifiedGaussianAdapter

class GS_DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    Gaussian Parameterを出せるように変更します。
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)

        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

        # ここは今一旦適当な実装にしている。
        self.path1_merger = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 7, 1, 3),
            nn.ReLU(),
        )
        self.raw_image_merger = nn.Sequential(
            nn.Conv2d(3, self.feature_dim, 7, 1, 3),
            nn.ReLU(),
        )

    def forward(self, encoder_tokens: List[torch.Tensor], raw_image: torch.Tensor = None, image_size=None, ret_feat=False):
        assert (
            self.dim_tokens_enc is not None
        ), "Need to call init(dim_tokens_enc) function first"

        image_size = self.image_size if image_size is None else image_size
        H, W = image_size

        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        layers = [encoder_tokens[hook] for hook in self.hooks] # hooksというのは、[2, 5, 8, 11]らしい。
        # shape: B x N x C のテンソルの配列。 (CUT3Rのイメージトークン)
        layers = [self.adapt_tokens(l) for l in layers]
        # 何もしていないらしい?

        layers = [
            rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers
        ]
        # shape: B x N x C -> B x C x N_H x N_W

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # 各層に対して、畳み込み or 逆畳み込みを実行している。
        # 例: 1: [8, 96, 64, 64], 2: [8, 192, 32, 32], 3: [8, 384, 16, 16], 4: [8, 768, 8, 8]

        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
        # チャンネル数の調整
        # 例: 1: [8, 256, 64, 64], 2: [8, 256, 32, 32], 3: [8, 256, 16, 16], 4: [8, 256, 8, 8]

        path_4 = self.scratch.refinenet4(layers[3])[
            :, :, : layers[2].shape[2], : layers[2].shape[3]
        ] # [8, 256, 18, 32]
        path_3 = self.scratch.refinenet3(path_4, layers[2]) # [8, 256, 36, 64]
        path_2 = self.scratch.refinenet2(path_3, layers[1]) # [8, 256, 72, 128]
        path_1 = self.scratch.refinenet1(path_2, layers[0]) # [8, 256, 144, 256]

        # ここまでで通常のDPTの出力を得る。
        # print(f"paths shapes: {[path_1.shape, path_2.shape, path_3.shape, path_4.shape]}")
        path1 =  self.path1_merger(path_1)  # [8, self.feature_dim, H, W]
        path1 = F.interpolate(path1, size=(H, W), mode='bilinear', align_corners=True)
        direct_img_feat = self.raw_image_merger(raw_image)  # [8, self.feature_dim, H, W]
        out =  path1 + direct_img_feat
        out = self.head(out) # [8, output_dim, H, W]
        
        if ret_feat:
            return out, path_1

        return out

class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)

        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None, ret_feat=False):
        assert (
            self.dim_tokens_enc is not None
        ), "Need to call init(dim_tokens_enc) function first"

        image_size = self.image_size if image_size is None else image_size
        H, W = image_size

        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        layers = [encoder_tokens[hook] for hook in self.hooks]

        layers = [self.adapt_tokens(l) for l in layers]

        layers = [
            rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers
        ]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]

        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        path_4 = self.scratch.refinenet4(layers[3])[
            :, :, : layers[2].shape[2], : layers[2].shape[3]
        ]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        out = self.head(path_1)
        
        if ret_feat:
            return out, path_1

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(
        self,
        *,
        n_cls_token=0,
        hooks_idx=None,
        dim_tokens=None,
        output_width_ratio=1,
        num_channels=1,
        postprocess=None,
        depth_mode=None,
        conf_mode=None,
        **kwargs
    ):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(
            output_width_ratio=output_width_ratio, num_channels=num_channels, **kwargs
        )
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_dpt_head(net, has_conf=False):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim // 2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(
        num_channels=out_nchan + has_conf,
        feature_dim=feature_dim,
        last_dim=last_dim,
        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
        dim_tokens=[ed, dd, dd, dd],
        postprocess=postprocess,
        depth_mode=net.depth_mode,
        conf_mode=net.conf_mode,
        head_type="regression",
    )


class DPTPts3dPose(nn.Module):
    def __init__(self, net, has_conf=False, has_rgb=False, has_pose=False):
        super(DPTPts3dPose, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode

        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose = has_pose

        pts_channels = 3 + has_conf
        rgb_channels = has_rgb * 3
        feature_dim = 256
        last_dim = feature_dim // 2
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        hooks_idx = [0, 1, 2, 3]
        dim_tokens = [ed, dd, dd, dd]
        head_type = "regression"
        output_width_ratio = 1

        pts_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=pts_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        rgb_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=rgb_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        if hooks_idx is not None:
            pts_dpt_args.update(hooks=hooks_idx)
            rgb_dpt_args.update(hooks=hooks_idx)

        self.dpt_self = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_self.init(**dpt_init_args)

        self.final_transform = nn.ModuleList(
            [
                ConditionModulationBlock(
                    net.dec_embed_dim,
                    net.dec_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    rope=net.rope,
                )
                for _ in range(2)
            ]
        )

        self.dpt_cross = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_cross.init(**dpt_init_args)

        if has_rgb:
            self.dpt_rgb = DPTOutputAdapter_fix(**rgb_dpt_args)
            dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
            self.dpt_rgb.init(**dpt_init_args)

        if has_pose:
            in_dim = net.dec_embed_dim
            self.pose_head = PoseDecoder(hidden_size=in_dim)

    def forward(self, x, img_info, **kwargs):
        if self.has_pose:
            pose_token = x[-1][:, 0].clone()
            token = x[-1][:, 1:]
            with torch.cuda.amp.autocast(enabled=False):
                pose = self.pose_head(pose_token)

            token_cross = token.clone()
            for blk in self.final_transform:
                token_cross = blk(token_cross, pose_token, kwargs.get("pos"))
            x = x[:-1] + [token]
            x_cross = x[:-1] + [token_cross]

        with torch.cuda.amp.autocast(enabled=False):
            self_out = checkpoint(
                self.dpt_self,
                x,
                image_size=(img_info[0], img_info[1]),
                use_reentrant=False,
            )

            final_output = postprocess(self_out, self.depth_mode, self.conf_mode)
            final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
            final_output["conf_self"] = final_output.pop("conf")

            if self.has_rgb:
                rgb_out = checkpoint(
                    self.dpt_rgb,
                    x,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                rgb_output = postprocess_rgb(rgb_out)
                final_output.update(rgb_output)

            if self.has_pose:
                pose = postprocess_pose(pose, self.pose_mode)
                final_output["camera_pose"] = pose  # B,7
                cross_out = checkpoint(
                    self.dpt_cross,
                    x_cross,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                tmp = postprocess(cross_out, self.depth_mode, self.conf_mode)
                final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
                final_output["conf"] = tmp.pop("conf")
        return final_output


class DPTPts3dPoseSMPL(nn.Module):
    def __init__(self, net, has_conf=False, has_rgb=False, has_pose=False, has_msk=False):
        super(DPTPts3dPoseSMPL, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode

        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose = has_pose
        self.has_msk = has_msk

        pts_channels = 3 + has_conf
        rgb_channels = has_rgb * 3
        feature_dim = 256
        last_dim = feature_dim // 2
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        hooks_idx = [0, 1, 2, 3]
        dim_tokens = [ed, dd, dd, dd]
        head_type = "regression"
        output_width_ratio = 1

        pts_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=pts_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        rgb_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=rgb_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        if hooks_idx is not None:
            pts_dpt_args.update(hooks=hooks_idx)
            rgb_dpt_args.update(hooks=hooks_idx)

        self.dpt_self = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_self.init(**dpt_init_args)

        self.final_transform = nn.ModuleList(
            [
                ConditionModulationBlock(
                    net.dec_embed_dim,
                    net.dec_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    rope=net.rope,
                )
                for _ in range(2)
            ]
        )

        self.dpt_cross = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_cross.init(**dpt_init_args)

        if has_rgb:
            self.dpt_rgb = DPTOutputAdapter_fix(**rgb_dpt_args)
            dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
            self.dpt_rgb.init(**dpt_init_args)

        if has_pose:
            in_dim = net.dec_embed_dim
            self.pose_head = PoseDecoder(hidden_size=in_dim)
            self.in_dim = in_dim

        ########### ここにGS推定を追加する #########
        self.gaussian_adapter = UnifiedGaussianAdapter()
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in 
        self.gs_channels = self.raw_gs_dim
        gs_head_type = "gs"
        gs_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=self.gs_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=gs_head_type,
        )
        if hooks_idx is not None:
            gs_dpt_args.update(hooks=hooks_idx)
        self.dpt_gs = GS_DPTOutputAdapter_fix(**gs_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_gs.init(**dpt_init_args)


        ##########################################

        # MHMR Heads - Detection
        backbone_dim = net.backbone_dim
        self.bb_patch_size = net.bb_patch_size
        self.mlp_classif = regression_mlp([backbone_dim, backbone_dim, 1]) # bg or human
        self.mlp_offset = regression_mlp([backbone_dim, backbone_dim, 2]) # offset
        if has_msk:
            self.mlp_msk = SMPLDecoder(
                hidden_size=backbone_dim, target_dim=self.bb_patch_size**2, num_layers=2, mlp_ratio=1)

        # feature fuse
        self.mlp_fuse = SMPLDecoder(hidden_size=ed+backbone_dim, target_dim=dd, num_layers=2, mlp_ratio=4)

        # SMPL
        self.joint_rep_type, self.joint_rep_dim = '6d', 6
        self.nrot = 53
        self.num_body_joints = self.nrot - 1

        npose = self.joint_rep_dim * (self.num_body_joints + 1)
        self.npose = npose

        self.input_is_mean_shape = True
        self.num_betas = 10
        assert self.num_betas in [10, 11]

        # SMPL param heads
        self.deccam = SMPLDecoder(
                        hidden_size=in_dim, 
                        target_dim=3, 
                        num_layers=2, 
                        mlp_ratio=4)
        self.decpose, self.decshape, self.decexpression = [
            SMPLDecoder(
                hidden_size=in_dim+backbone_dim, 
                target_dim=od, 
                num_layers=2, 
                mlp_ratio=4) for od in [self.npose, self.num_betas, 10]]

        self.set_smpl_init()


    def set_smpl_init(self):
        """ Fetch saved SMPL parameters and register buffers."""
        mean_params = np.load(MEAN_PARAMS)
        if self.nrot == 53:
            init_body_pose = torch.eye(3).reshape(1,3,3).repeat(self.nrot,1,1)[:,:,:2].flatten(1).reshape(1, -1)
            init_body_pose[:,:24*6] = torch.from_numpy(mean_params['pose'][:]).float() # global_orient+body_pose from SMPL
        else:
            init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)

        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        init_betas_kid = torch.cat([init_betas, torch.zeros_like(init_betas[:,[0]])],1)
        init_expression = 0. * torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)

        if self.num_betas == 11:
            init_betas = torch.cat([init_betas, torch.zeros_like(init_betas[:,:1])], 1)

        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_betas_kid', init_betas_kid)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_expression', init_expression)
        
    def detect_mhmr(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            scores = postprocess_score(self.mlp_classif(x)) # per token detection score.
        return scores

    def segment(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            msks = postprocess_score(self.mlp_msk(x))
        return msks
    
    def forward(self, x, img_info, **kwargs):
        # ここでSMPLの推定をしている
        if self.has_pose:
            pose_token = x[-1][:, 0].clone()
            n_humans_i = kwargs.get("n_humans")
            token = x[-1][:, 1:]

            with torch.cuda.amp.autocast(enabled=False):
                pose = self.pose_head(pose_token)
                if n_humans_i > 0:
                    smpl_token = kwargs.get("smpl_token")   # CUT3R smpl token (bs, 10, 768)
                    pred_body_pose = self.decpose(smpl_token) + self.init_body_pose
                    pred_betas = self.decshape(smpl_token) + self.init_betas
                    pred_cam = self.deccam(smpl_token[..., :self.in_dim])
                    pred_expression = self.decexpression(smpl_token) + self.init_expression
                    pred_smpl = [pred_body_pose, pred_betas, pred_cam, pred_expression]
                    
            token_cross = token.clone()
            for blk in self.final_transform:
                token_cross = blk(token_cross, pose_token, kwargs.get("pos"))
            x = x[:-1] + [token]
            x_cross = x[:-1] + [token_cross]

        # ここで点群の推定をしている
        with torch.cuda.amp.autocast(enabled=False):
            self_out = checkpoint(
                self.dpt_self,
                x,
                image_size=(img_info[0], img_info[1]),
                # ret_feat=True,
                use_reentrant=False,
            )

            final_output = postprocess(self_out, self.depth_mode, self.conf_mode)
            final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
            final_output["conf_self"] = final_output.pop("conf")

            if self.has_rgb:
                rgb_out = checkpoint(
                    self.dpt_rgb,
                    x,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                rgb_output = postprocess_rgb(rgb_out)
                final_output.update(rgb_output)

            if self.has_pose:
                pose = postprocess_pose(pose, self.pose_mode)
                final_output["camera_pose"] = pose  # B,7
                cross_out = checkpoint(
                    self.dpt_cross,
                    x_cross,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                tmp = postprocess(cross_out, self.depth_mode, self.conf_mode)
                final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
                final_output["conf"] = tmp.pop("conf")
            
            if n_humans_i > 0:
                smpl_out = postprocess_smpl(pred_smpl, self.depth_mode)
                final_output.update(smpl_out)

        # ここで背景の3DGSの推定を入れる。
        # 入力: x (画像トークン)、元画像、画像サイズ
        # 出力: ガウシアンパラメータ
        # print(img_info) # (288, 512)
        # print(kwargs.get("image").shape) # (8, 3, 288, 512)
        gs_params = self.dpt_gs(encoder_tokens=x, raw_image=kwargs.get("image"), image_size=(img_info[0], img_info[1]))
        # print(f"gs_params shape: {gs_params.shape}") # (8, 83, 288, 512)

        anchor_feats = gs_params[:, : self.raw_gs_dim] # (8, 83, 288, 512)
        pts_3d = final_output['pts3d_in_other_view'] # (8, 288, 512, 3)

        neural_feats = anchor_feats.permute(0,2,3,1).flatten(1,2) # (8, 288*512, 83)
        neural_pts = pts_3d.flatten(1,2) # (8, 288*512, 3)

        depths = neural_pts[..., -1] # (8, N)
        densities = neural_feats[..., 0].sigmoid() # (8, N)

        opacity = densities # (8, N)

        # gaussians = self.gaussian_adapter.forward(
        #     neural_pts, # (8, N, 3)
        #     depths, # (8, N)
        #     opacity, # (8, N)
        #     neural_feats[..., 1:], # (8, N, 82)
        # )
        # final_output['gaussians'] = gaussians
        final_output['gs_feats'] = neural_feats # (8, 288*512, 83)
        final_output['gs_pts'] = neural_pts # (8, 288*512, 3)

        return final_output


class NaiveDPTPts3dPoseSMPL(nn.Module):
    def __init__(self, net, has_conf=False, has_rgb=False, has_pose=False):
        super(NaiveDPTPts3dPoseSMPL, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode

        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose = has_pose

        pts_channels = 3 + has_conf
        rgb_channels = has_rgb * 3
        feature_dim = 256
        last_dim = feature_dim // 2
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        hooks_idx = [0, 1, 2, 3]
        dim_tokens = [ed, dd, dd, dd]
        head_type = "regression"
        output_width_ratio = 1

        pts_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=pts_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        rgb_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=rgb_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        if hooks_idx is not None:
            pts_dpt_args.update(hooks=hooks_idx)
            rgb_dpt_args.update(hooks=hooks_idx)

        self.dpt_self = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_self.init(**dpt_init_args)

        self.final_transform = nn.ModuleList(
            [
                ConditionModulationBlock(
                    net.dec_embed_dim,
                    net.dec_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    rope=net.rope,
                )
                for _ in range(2)
            ]
        )

        self.dpt_cross = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_cross.init(**dpt_init_args)

        if has_rgb:
            self.dpt_rgb = DPTOutputAdapter_fix(**rgb_dpt_args)
            dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
            self.dpt_rgb.init(**dpt_init_args)

        if has_pose:
            in_dim = net.dec_embed_dim
            self.pose_head = PoseDecoder(hidden_size=in_dim)

        # MHMR Heads - Detection
        backbone_dim = net.backbone_dim
        self.bb_patch_size = net.bb_patch_size
        self.mlp_classif = regression_mlp([backbone_dim, backbone_dim, 1]) # bg or human
        self.mlp_offset = regression_mlp([backbone_dim, backbone_dim, 2]) # offset

        # SMPL
        self.joint_rep_type, self.joint_rep_dim = '6d', 6
        self.nrot = 53
        self.num_body_joints = self.nrot - 1

        npose = self.joint_rep_dim * (self.num_body_joints + 1)
        self.npose = npose

        self.input_is_mean_shape = True
        self.num_betas = 10
        assert self.num_betas in [10, 11]

        # MHMR Heads - SMPL
        self.decpose, self.decshape, self.deccam, self.decexpression = [
            nn.Linear(1024, od) for od in [self.npose, self.num_betas, 3, 10]] # MLP(1024, x)

        self.set_smpl_init()

    def set_smpl_init(self):
        """ Fetch saved SMPL parameters and register buffers."""
        mean_params = np.load(MEAN_PARAMS)
        if self.nrot == 53:
            init_body_pose = torch.eye(3).reshape(1,3,3).repeat(self.nrot,1,1)[:,:,:2].flatten(1).reshape(1, -1)
            init_body_pose[:,:24*6] = torch.from_numpy(mean_params['pose'][:]).float() # global_orient+body_pose from SMPL
        else:
            init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)

        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        init_betas_kid = torch.cat([init_betas, torch.zeros_like(init_betas[:,[0]])],1)
        init_expression = 0. * torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)

        if self.num_betas == 11:
            init_betas = torch.cat([init_betas, torch.zeros_like(init_betas[:,:1])], 1)

        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_betas_kid', init_betas_kid)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_expression', init_expression)
        
    def detect_mhmr(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            scores = postprocess_score(self.mlp_classif(x)) # per token detection score.
        return scores
      
    def forward(self, x, img_info, **kwargs):
        if self.has_pose:
            pose_token = x[-1][:, 0].clone()
            n_humans_i = kwargs.get("n_humans")
            token = x[-1][:, 1:]

            with torch.cuda.amp.autocast(enabled=False):
                pose = self.pose_head(pose_token)
                if n_humans_i > 0:
                    smpl_token = kwargs.get("smpl_token") 
                    decoders = [self.decpose, self.decshape, self.deccam, self.decexpression]
                    inits = [self.init_body_pose, self.init_betas, self.init_cam, self.init_expression]
                    pred_smpl = [d(smpl_token) + i for d, i in zip(decoders, inits)]

            token_cross = token.clone()
            for blk in self.final_transform:
                token_cross = blk(token_cross, pose_token, kwargs.get("pos"))
            x = x[:-1] + [token]
            x_cross = x[:-1] + [token_cross]

        with torch.cuda.amp.autocast(enabled=False):
            self_out = checkpoint(
                self.dpt_self,
                x,
                image_size=(img_info[0], img_info[1]),
                use_reentrant=False,
            )

            final_output = postprocess(self_out, self.depth_mode, self.conf_mode)
            final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
            final_output["conf_self"] = final_output.pop("conf")

            if self.has_rgb:
                rgb_out = checkpoint(
                    self.dpt_rgb,
                    x,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                rgb_output = postprocess_rgb(rgb_out)
                final_output.update(rgb_output)

            if self.has_pose:
                pose = postprocess_pose(pose, self.pose_mode)
                final_output["camera_pose"] = pose  # B,7
                cross_out = checkpoint(
                    self.dpt_cross,
                    x_cross,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                tmp = postprocess(cross_out, self.depth_mode, self.conf_mode)
                final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
                final_output["conf"] = tmp.pop("conf")
            
            if n_humans_i > 0:
                smpl_out = postprocess_smpl(pred_smpl, self.depth_mode, naive_mode=True)
                final_output.update(smpl_out)

        return final_output
