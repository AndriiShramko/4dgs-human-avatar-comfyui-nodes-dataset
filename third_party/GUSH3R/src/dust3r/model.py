import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
import math
import inspect
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    fix_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder, pose_encoding_to_camera
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

from dust3r.smpl_model import nms, apply_threshold
from einops import rearrange

from dust3r.utils.geometry import inverse_perspective_projection, get_camera_parameters, geotrf
from dust3r.utils.image import unpad_uv, unpad_image, log_optimal_transport
from mhmr.blocks import Dinov2Backbone, FourierPositionEncoding, TransformerDecoder
printer = get_logger(__name__, log_level="DEBUG")

from dust3r.utils.device import to_cpu, to_gpu

from dust3r.heads.gaussian_splatter.decoder_splatting_cuda import DecoderSplattingCUDA
from dust3r.heads.human_gs.human_gs_head import HumanGSHead

from torch_scatter import scatter_add, scatter_max
import roma
from torchvision.utils import save_image

@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def strip_module_mhmr(state_dict):
    """
    Load Multi-HMR pretrained model
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(("mlp_classif.", "mlp_offset.")):
            name = f"downstream_head.{k}"
        elif k.startswith(("x_attention_head.dec")):
            name = f"downstream_head.{k[17:]}"
        elif k.startswith(("x_attention_head.transformer.", "x_attention_head.cross_")):
            name = k[17:]
        elif k.startswith(("backbone")):
            name = k
        else:
            continue
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if "human_gs_num_memory_tokens=" not in args:
        ckpt_memory = ckpt["model"].get("human_gs_head.init_human_memory")
        human_gs_num_memory_tokens = int(ckpt_memory.shape[0]) if ckpt_memory is not None else 0
        args = args[:-2] + f", human_gs_num_memory_tokens={human_gs_num_memory_tokens}))"
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        msk_head=False,
        use_prompt=False,
        is_shallow=False,
        prompt_size=None,
        backbone='dinov2_vitl14',
        mhmr_img_res=None,
        gs_mode='bg',
        gs_point_conf_threshold=1.5,
        bg_mask_threshold=0.05,
        bg_mask_dilation=3,
        bg_voxel_size=0.01,
        bg_gaussian_max=200000,
        bg_merge_point_mode="max_conf",
        bg_point_offset_enabled=False,
        bg_point_offset_max_ratio=0.5,
        human_gs_sampling_mode="fps",
        human_gs_sampling_seed=0,
        human_gs_sampling_dir="dust3r/heads/human_gs/smplx_mesh/sampling_cache",
        human_gs_num_memory_tokens=0,
        human_gs_transformer_type="cross_attention",
        human_gs_teacher_force_gt_smpl=False,
        arcface_ckpt=None,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.msk_head = msk_head
        self.backbone = backbone
        self.mhmr_img_res = mhmr_img_res
        self.croco_kwargs = croco_kwargs

        self.gs_mode = gs_mode
        self.gs_point_conf_threshold = gs_point_conf_threshold
        self.bg_mask_threshold = bg_mask_threshold
        self.bg_mask_dilation = bg_mask_dilation
        self.bg_voxel_size = bg_voxel_size
        self.bg_gaussian_max = bg_gaussian_max
        self.bg_merge_point_mode = bg_merge_point_mode
        self.bg_point_offset_enabled = bg_point_offset_enabled
        self.bg_point_offset_max_ratio = bg_point_offset_max_ratio
        self.human_gs_sampling_mode = human_gs_sampling_mode
        self.human_gs_sampling_seed = human_gs_sampling_seed
        self.human_gs_sampling_dir = human_gs_sampling_dir
        self.human_gs_num_memory_tokens = human_gs_num_memory_tokens
        self.human_gs_transformer_type = human_gs_transformer_type
        self.human_gs_teacher_force_gt_smpl = human_gs_teacher_force_gt_smpl
        self.arcface_ckpt = arcface_ckpt


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        croco_param_names = set(inspect.signature(CrocoConfig.__init__).parameters)
        # Older checkpoints may contain experiment-only config keys.
        config.croco_kwargs = {
            k: v for k, v in croco_kwargs.items() if k in croco_param_names
        }
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        self.msk_head_flag = config.msk_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim)
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_smpl_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )

        # MHMR
        # 'dinov2_vits14': 384, 'dinov2_vitb14': 768, 'dinov2_vitl14': 1024
        self.backbone = Dinov2Backbone(config.backbone, pretrained=False)
        self.bb_patch_size = self.backbone.patch_size
        self.backbone_dim = self.backbone.embed_dim
        self.mhmr_img_res = config.mhmr_img_res
        self.bb_token_res = self.mhmr_img_res // self.bb_patch_size

        if config.output_mode == 'naive':
            self.fourier_camera = FourierPositionEncoding(n=3, num_bands=16, max_resolution=64)
            self.camera_embed_dim = self.fourier_camera.channels
            context_dim = self.backbone_dim + self.camera_embed_dim

            transformer_args = dict(
                num_tokens=1,
                token_dim=(318+10+3+context_dim),
                dim=1024,
                depth=2,
                heads=8,
                mlp_dim=1024,
                dim_head=32,
                dropout=0.0,
                emb_dropout=0.0,
                context_dim=context_dim,
            )
            self.transformer = TransformerDecoder(**transformer_args)
            # Init learned embeddings for queries
            self.cross_queries_x = nn.Parameter(torch.zeros(self.bb_token_res, context_dim))
            torch.nn.init.normal_(self.cross_queries_x, std=0.2)
            self.cross_queries_y = nn.Parameter(torch.zeros(self.bb_token_res, context_dim))
            torch.nn.init.normal_(self.cross_queries_y, std=0.2)
            self.cross_values_x = nn.Parameter(torch.zeros(self.bb_token_res, context_dim))
            torch.nn.init.normal_(self.cross_values_x, std=0.2)
            self.cross_values_y = nn.Parameter(torch.zeros(self.bb_token_res, context_dim))
            torch.nn.init.normal_(self.cross_values_y, std=0.2)


        self.mhmr_masked_smpl_token = nn.Parameter(
            torch.randn(
                1, context_dim if config.output_mode == "naive" else self.backbone_dim
                ) * 0.02, requires_grad=True
        )
        self.mhmr_masked_img_token = nn.Parameter(
            torch.randn(1, self.backbone_dim) * 0.02, requires_grad=True
        )

        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            config.msk_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)

        ### 以下追加実装している機能
        self.gs_mode = config.gs_mode
        self.gs_point_conf_threshold = config.gs_point_conf_threshold
        self.bg_mask_threshold = config.bg_mask_threshold
        self.bg_mask_dilation = config.bg_mask_dilation
        self.bg_voxel_size = config.bg_voxel_size
        self.bg_gaussian_max = config.bg_gaussian_max
        self.bg_merge_point_mode = getattr(config, "bg_merge_point_mode", "max_conf")
        self.bg_point_offset_enabled = getattr(config, "bg_point_offset_enabled", False)
        self.bg_point_offset_max_ratio = getattr(config, "bg_point_offset_max_ratio", 0.5)
        self.human_gs_teacher_force_gt_smpl = config.human_gs_teacher_force_gt_smpl
        # Gaussian Splatting Decoder
        if config.gs_mode in ["bg", "both"]:
            self.decoder_splatting = DecoderSplattingCUDA()
            if self.bg_point_offset_enabled:
                offset_hidden_dim = self.downstream_head.raw_gs_dim * 2
                self.bg_point_offset_head = nn.Sequential(
                    nn.LayerNorm(self.downstream_head.raw_gs_dim),
                    nn.Linear(self.downstream_head.raw_gs_dim, offset_hidden_dim),
                    nn.GELU(),
                    nn.Linear(offset_hidden_dim, offset_hidden_dim),
                    nn.GELU(),
                    nn.Linear(offset_hidden_dim, 3),
                )
                nn.init.zeros_(self.bg_point_offset_head[-1].weight)
                nn.init.zeros_(self.bg_point_offset_head[-1].bias)

        # # 人間用のガウシアンヘッド
        if config.gs_mode in ["human", "both"]:
            self.human_gs_head = HumanGSHead(cfg=config)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
            "encoder_and_decoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
                self.downstream_head.dpt_self,
                self.downstream_head.final_transform,
                self.downstream_head.dpt_cross,
                self.downstream_head.dpt_rgb,
                self.downstream_head.pose_head,
            ],
            "mhmr": [
                self.backbone,
                self.mhmr_masked_img_token,
                self.downstream_head.mlp_classif,
                self.downstream_head.mlp_offset,
            ],
        }
        if self.output_mode == "naive":
            to_be_frozen["all"] = [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
                self.downstream_head.dpt_self,
                self.downstream_head.final_transform,
                self.downstream_head.dpt_cross,
                self.downstream_head.dpt_rgb,
                self.downstream_head.pose_head,
                self.backbone,
                self.mhmr_masked_smpl_token,
                self.mhmr_masked_img_token,
                self.transformer,
                self.cross_queries_x,
                self.cross_queries_y,
                self.cross_values_x,
                self.cross_values_y,
                self.downstream_head.mlp_classif,
                self.downstream_head.mlp_offset,
                self.downstream_head.decpose, 
                self.downstream_head.decshape, 
                self.downstream_head.deccam, 
                self.downstream_head.decexpression,
            ]

        if freeze == "encoder_and_decoder_and_head":
            fix_all_params(to_be_frozen["encoder_and_decoder_and_head"]) # will not be updated
            freeze_all_params(to_be_frozen["encoder"]) # requires_grad = False
            freeze_all_params(to_be_frozen["mhmr"])
        else:
            freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        msk_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
            has_msk=bool(msk_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        )
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views_mhmr(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos

        # MHMR
        imgs_mhmr = torch.stack(
            [view["img_mhmr"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        imgs_mhmr = imgs_mhmr.view(
            -1, *imgs_mhmr.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        selected_imgs_mhmr = imgs_mhmr[img_masks_flat]
        if selected_imgs_mhmr.size(0) > 0:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
            selected_imgs_mhmr = (selected_imgs_mhmr * 0.5 + 0.5 - mean) / std
            mhmr_img_out = [self.backbone(selected_imgs_mhmr)] # image[bs, 3, h, w] -> image feature [bs, h_nb_patches * w_nb_patches, D]
        else:
            raise NotImplementedError
        
        mhmr_full_out = [
            torch.zeros(
                len(views) * batch_size, *mhmr_img_out[0].shape[1:], device=mhmr_img_out[0].device
            )
            for _ in range(len(mhmr_img_out))
        ]
        for i in range(len(mhmr_img_out)):
            mhmr_full_out[i][img_masks_flat] += mhmr_img_out[i]
            mhmr_full_out[i][~img_masks_flat] += self.mhmr_masked_img_token

        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
            [mhmr_out.chunk(len(views), dim=0) for mhmr_out in mhmr_full_out],
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, f_smpl, pos_smpl, use_ttt3r=False):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img)
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            if f_smpl is not None:
                f_img = torch.cat([f_pose, f_img, f_smpl], dim=1)
                pos_img = torch.cat([pos_pose, pos_img, pos_smpl], dim=1)
            else:
                f_img = torch.cat([f_pose, f_img], dim=1) # used for naive CUT3R+MHMR
                pos_img = torch.cat([pos_pose, pos_img], dim=1) # used for naive CUT3R+MHMR
        final_output.append((f_state, f_img))
        cross_attn_states = []
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _, cross_attn_state = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    use_ttt3r=use_ttt3r,
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _, _ = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    use_ttt3r=False,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _, cross_attn_state = blk_state(*final_output[-1][::+1], pos_state, pos_img, use_ttt3r=use_ttt3r)
                f_img, _, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state, use_ttt3r=False)

            final_output.append((f_state, f_img))
            cross_attn_states.append(cross_attn_state)

        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )

        return zip(*final_output), cross_attn_states

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        smpl_feat,
        smpl_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        use_ttt3r=False,
    ):
        (new_state_feat, dec), cross_attn_states = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos, smpl_feat, smpl_pos, use_ttt3r
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec, cross_attn_states

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def embedd_camera(self, K, n_patch):
        """ Embed viewing directions using fourrier encoding."""
        bs = K.shape[0]
        _h, _w = n_patch
        points = torch.stack([
            torch.arange(0,_h,1).reshape(-1,1).repeat(1,_w), 
            torch.arange(0,_w,1).reshape(1,-1).repeat(_h,1)],
            -1).to(K.device).float() # [h,w,2]
        points = points * self.bb_patch_size + self.bb_patch_size // 2 # move to pixel space - we give the pixel center of each token
        points = points.reshape(1,-1,2).repeat(bs,1,1) # (bs, hw, 2): 2D points
        distance = torch.ones(bs,points.shape[1],1).to(K.device) # (bs, N, 1): distance in the 3D world
        rays = inverse_perspective_projection(points, K, distance) # (bs, N, 3)
        rays_embeddings = self.fourier_camera(pos=rays)

        # Repeat for each element of the batch
        z_K = rays_embeddings.reshape(bs,_h,_w,self.camera_embed_dim) # [bs,h,w,99]
        return z_K 

    def smpl_tokenizer_mhmr(self, feat, pos, views, inference=False):
        feat = torch.stack([f.detach() for f in feat], dim=0) #(num_view, bs, 576, 1024)
        num_view, batch_size = feat.shape[:2]

        feat = feat.view(-1, *feat.shape[2:]) #(num_view * bs, 576, 1024)
        scores = self.downstream_head.detect_mhmr(feat) #(num_view * bs, 576, 1)
        if self.msk_head_flag:
            msks = self.downstream_head.segment(feat.detach()) #(num_view * bs, 576, 14*14)

        # Restore Height and Width dimensions.
        n_patch = self.bb_token_res # H,W
        scores = rearrange(scores, "b (nh nw) c -> b c nh nw", nh=n_patch, nw=n_patch)
        feat = rearrange(feat, "b (nh nw) c -> b nh nw c", nh=n_patch, nw=n_patch) # head token extraction: (num_view * bs, h, w, 1024)
        if self.msk_head_flag:
            msks = rearrange(msks, "b (nh nw) c -> b c nh nw", nh=n_patch, nw=n_patch)
            msks = F.pixel_shuffle(msks, self.bb_patch_size)  # (num_view * bs, 1, h, w)

        if self.output_mode == "naive":
            # use GT K
            K = torch.stack([v['K_mhmr'] for v in views], dim=0)
            K = K.view(-1, *K.shape[2:])
            # # use pseudo K
            # K = get_camera_parameters(self.mhmr_img_res, device=feat.device)
            # K = K.expand(feat.shape[0], -1, -1)
            feat_K = self.embedd_camera(K, [n_patch, n_patch]) # Embed viewing directions. [num_view * bs,h,w,99]
        
        if inference:
            scores = nms(scores, kernel=3) # (num_view * bs, 1, h, w)
            _scores = scores.permute((0, 2, 3, 1)) # (num_view * bs, h, w, 1)
            # Binary decision (keep confident detections)
            idx = apply_threshold(0.3, _scores)
            img_id, h_id, w_id = idx[0], idx[1], idx[2]
        else:
            smpl_mask = torch.stack([view["smpl_mask"] for view in views], dim=0)
            smpl_mask = smpl_mask.view(-1, *smpl_mask.shape[2:])
            max_humans = smpl_mask.shape[1]
            smpl_uv = torch.stack([view["smpl_uv"] for view in views], dim=0)
            smpl_uv = smpl_uv.view(-1, *smpl_uv.shape[2:])[smpl_mask]
            img_id = torch.where(smpl_mask)[0]
            h_id, w_id = smpl_uv.T

        # Scores  
        scores = scores.permute((0, 2, 3, 1)) # (num_view * bs, h, w, 1)
        if self.msk_head_flag:
            msks = msks.permute((0, 2, 3, 1)) # (num_view * bs, h, w, 1)
        
        # Head token and offset
        feat_central = feat[img_id, h_id, w_id] # (nvh, 1024)
        offset = self.downstream_head.mlp_offset(feat_central)# [nhv,2]
        # Distance for estimating the 3D location in 3D space
        loc = torch.stack([w_id, h_id]).permute(1,0) # x,y
        loc = (loc + 0.5 + offset) * self.bb_patch_size # Moving to higher res the location of the pelvis

        if self.output_mode == "naive":
            # Concat with camera embedding
            feat_K_central = feat_K[img_id, h_id, w_id] # (nvh, 99)
            feat_central = torch.cat([feat_central, feat_K_central], 1) # feature + camera embedding for heads only to query tokens [nhv, 1123]
            feat_all = torch.cat([feat, feat_K], -1).permute(0,3,1,2) # feature + camera embedding for full image for the cross-attention only. [bs,1123,nh,nw]

            # Get learned embeddings for queries, at positions with detected people.
            queries_xy = self.cross_queries_x[h_id] + self.cross_queries_y[w_id]
            # Add the embedding to the central features.
            feat_central = feat_central + queries_xy # [nhv, 1123]
            # Inject leared embeddings for key/values at detected locations. 
            values_xy = self.cross_values_x[h_id] + self.cross_values_y[w_id]
            feat_all[img_id, :, h_id, w_id] += values_xy  # [bs, 1123, nh, nw]
            feat_all = rearrange(feat_all, "b c h w -> b (h w) c") # (num_view * bs, nh*nw, 1024)

        if inference:
            head_token = feat_central
            head_loc = loc
            expand = lambda x: x.expand(*feat_central.shape[:-1] , -1)
        else:
            # concat with mask token
            full_out = torch.zeros(
                num_view * batch_size, max_humans, feat_central.shape[1], 
                device=feat_central.device
            )
            full_out[smpl_mask] += feat_central
            full_out[~smpl_mask] += self.mhmr_masked_smpl_token

            loc_full_out = torch.zeros(
                num_view * batch_size, max_humans, loc.shape[1], 
                device=loc.device
            )
            loc_full_out[smpl_mask] += loc
        
            head_token = full_out
            head_loc = loc_full_out
            expand = lambda x: x.expand(num_view * batch_size, max_humans , -1)

        if self.output_mode == "naive":
            # Get initial smpl token from MHMR
            pred_body_pose, pred_betas, pred_cam, pred_expression = [expand(x) for x in
                    [self.downstream_head.init_body_pose, 
                    self.downstream_head.init_betas, 
                    self.downstream_head.init_cam, 
                    self.downstream_head.init_expression,
                    ]]
            head_token = torch.cat([
                head_token, pred_body_pose, pred_betas, pred_cam, 
                ], dim=-1)  # training: [bs, 10, 1454]; inference: [nhv, 1454]

        if inference:
            smpl_query_list, smpl_pos_list = [], []
            for i in range(num_view * batch_size):
                if self.output_mode == "naive":
                    smpl_query = self.transformer(
                        head_token[img_id == i].unsqueeze(0), 
                        context=feat_all[i].unsqueeze(0), 
                        mask=None) # train:[bs, 10, 1024]), inference:[1, nhv, 1024]
                else:
                    smpl_query = head_token[img_id == i].unsqueeze(0)   # use mhmr vit token
                smpl_query_list.append(smpl_query)
                smpl_pos = torch.zeros(
                    *smpl_query.shape[:2], 2).to(smpl_query.device).to(pos[0].dtype)
                smpl_pos_list.append(smpl_pos)
            loc_list = [
                head_loc[img_id == i].unsqueeze(0) for i in range(num_view * batch_size)]
        else:
            if self.output_mode == "naive":
                smpl_query = self.transformer(
                    head_token, 
                    context=feat_all, 
                    mask=smpl_mask.type(torch.float32)) # [bs, 10, 1024])
            else:
                smpl_query = head_token   # use mhmr vit token
            smpl_query_list = smpl_query.chunk(num_view, dim=0)
            loc_list = head_loc.chunk(num_view, dim=0)
            full_pos = torch.zeros(
                *smpl_query.shape[:2], 2).to(smpl_query.device).to(pos[0].dtype)
            smpl_pos_list = full_pos.chunk(num_view, dim=0)

        return (
            scores.chunk(num_view, dim=0), 
            smpl_query_list,
            smpl_pos_list,
            loc_list,
            msks.chunk(num_view, dim=0) if self.msk_head_flag else None,
        )

    def smpl_tokenizer_cut3r(self, feat, pos, views, loc, inference=False):
        feat = torch.stack([f.detach() for f in feat], dim=0) #(num_view, bs, 576, 1024)
        num_view, batch_size = feat.shape[:2]

        feat = feat.view(-1, *feat.shape[2:]) #(num_view * bs, 576, 1024)
        pos = torch.stack([p.detach() for p in pos], dim=0) #(num_view, bs, 576, 2)
        pos = pos.view(-1, *pos.shape[2:]) #(num_view * bs, 576, 2)

        # Restore Height and Width dimensions.
        n_patch = views[0]["true_shape"][0] // self.croco_args['patch_size'] # H,W
        feat = rearrange(feat, "b (nh nw) c -> b nh nw c", nh=n_patch[0], nw=n_patch[1]) # (num_view * bs, h, w, 1024)
        pos = rearrange(pos, "b (nh nw) c -> b nh nw c", nh=n_patch[0], nw=n_patch[1]) # (num_view * bs, h, w, 2)

        if inference:
            num_humans = [l.shape[1] for l in loc]
            img_id = torch.repeat_interleave(
                torch.arange(len(loc), device=loc[0].device), 
                torch.tensor(num_humans, device=loc[0].device)
            )
            loc = torch.cat([l.squeeze(0) for l in loc], dim=0) # (nvh, 2)
            loc_cut3r = unpad_uv(loc, self.mhmr_img_res, *views[0]["true_shape"][0])
            smpl_uv = (loc_cut3r // self.croco_args['patch_size']).int()
            w_id, h_id = smpl_uv.T
        else:
            smpl_mask = torch.stack([view["smpl_mask"] for view in views], dim=0)
            smpl_mask = smpl_mask.view(-1, *smpl_mask.shape[2:])
            max_humans = smpl_mask.shape[1]
            loc = torch.stack([l.detach() for l in loc], dim=0) # high-res head uv in mhmr: (num_view, bs, 10, 2)
            loc = loc.view(-1, *loc.shape[2:]) #(num_view * bs, 10, 2)
            loc_cut3r = unpad_uv(loc[smpl_mask], self.mhmr_img_res, *views[0]["true_shape"][0]) # high-res head uv in cut3r
            smpl_uv = (loc_cut3r // self.croco_args['patch_size']).int() # low-res head uv in cut3r
            img_id = torch.where(smpl_mask)[0]
            w_id, h_id = smpl_uv.T

        # Head token
        feat_central = feat[img_id, h_id, w_id] # (nvh, 1024)
        pos_central = pos[img_id, h_id, w_id] # (nvh, 2)

        if inference:
            smpl_query = feat_central
            head_uv = smpl_uv
        else:
            # concat with mask token and mean SMPL params
            full_out = torch.zeros(
                num_view * batch_size, max_humans, feat_central.shape[1], 
                device=feat_central.device
            )
            full_pos = torch.zeros(
                num_view * batch_size, max_humans, pos_central.shape[1], 
                device=pos_central.device, dtype=pos_central.dtype,
            )
            full_out[smpl_mask] += feat_central
            full_out[~smpl_mask] += self.masked_smpl_token
            full_pos[smpl_mask] += pos_central
            smpl_query = full_out

            uv_full_out = torch.zeros(
                num_view * batch_size, max_humans, smpl_uv.shape[1], 
                device=loc.device,
                dtype=smpl_uv.dtype
            )
            uv_full_out[smpl_mask] += smpl_uv
            head_uv = uv_full_out

        if inference:
            smpl_query_list = [
                smpl_query[img_id == i].unsqueeze(0) for i in range(num_view * batch_size)]
            smpl_pos_list = [
                pos_central[img_id == i].unsqueeze(0) for i in range(num_view * batch_size)]
            smpl_uv_list = [
                head_uv[img_id == i].unsqueeze(0) for i in range(num_view * batch_size)]
        else:
            smpl_query_list = smpl_query.chunk(num_view, dim=0)
            smpl_pos_list = full_pos.chunk(num_view, dim=0)
            smpl_uv_list = head_uv.chunk(num_view, dim=0)

        return (
            smpl_query_list,
            smpl_pos_list,
            smpl_uv_list,
        )

    def token_fuse(self, tk_mhmr, tk_cut3r, inference):
        if inference:
            num_humans = [t.shape[1] for t in tk_mhmr]
            num_view = len(tk_mhmr)
            img_id = torch.repeat_interleave(
                torch.arange(num_view, device=tk_mhmr[0].device), 
                torch.tensor(num_humans, device=tk_mhmr[0].device)
            )
            tk_mhmr = torch.cat([t.squeeze(0) for t in tk_mhmr], dim=0) # (nvh, 1024)
            tk_cut3r = torch.cat([t.squeeze(0) for t in tk_cut3r], dim=0) # (nvh, 1024)
            tk = torch.cat([tk_mhmr, tk_cut3r], dim=-1) #(nvh, 2048)
            fused_tk = self.downstream_head.mlp_fuse(tk)
            fused_tk_list = [
                fused_tk[img_id == i].unsqueeze(0) for i in range(num_view)]
        else:
            tk_mhmr = torch.stack([t.detach() for t in tk_mhmr], dim=0) #(num_view, bs, 10, 1024)
            tk_cut3r = torch.stack([t.detach() for t in tk_cut3r], dim=0) #(num_view, bs, 10, 1024)
            num_view, batch_size = tk_mhmr.shape[:2]
        
            tk_mhmr = tk_mhmr.view(-1, *tk_mhmr.shape[2:]) #(num_view * bs, 10, 1024)
            tk_cut3r = tk_cut3r.view(-1, *tk_cut3r.shape[2:]) #(num_view * bs, 10, 1024)
            tk = torch.cat([tk_mhmr, tk_cut3r], dim=-1) #(num_view * bs, 10, 2048)

            fused_tk = self.downstream_head.mlp_fuse(tk) #(num_view * bs, 10, 768)
            fused_tk_list = fused_tk.chunk(num_view, dim=0)
        return fused_tk_list

    def _forward_impl(self, views, ret_state=False, inference=False):
        with torch.no_grad():
            # ここは単純に画像をエンコードしているだけ。
            shape, feat_ls, pos, mhmr_feat_ls = self._encode_views_mhmr(views)
            feat = feat_ls[-1]
            mhmr_feat = mhmr_feat_ls[-1]

            scores, smpl_tk_mhmr, pos_mhmr, smpl_loc, msks = self.smpl_tokenizer_mhmr(
                mhmr_feat, pos, views, inference)
            smpl_tk_cut3r, pos_cut3r, smpl_uv_cut3r = self.smpl_tokenizer_cut3r(
                feat, pos, views, smpl_loc, inference)
            
            # fuse CUT3R and MHMR smpl tokens
            smpl_query = self.token_fuse(smpl_tk_mhmr, smpl_tk_cut3r, inference)
            pos_central = pos_cut3r

            state_feat, state_pos = self._init_state(feat[0], pos[0])
            mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)  # [b, 256, 1536]
            init_state_feat = state_feat.clone()
            init_mem = mem.clone()
            all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]

        ####### フレームループが始まる前の処理
        # ガウシアン関係
        from dust3r.heads.gaussian_utils.types import Gaussians
        B = feat[0].shape[0]
        N_max = getattr(self, "bg_gaussian_max", 200000) # 最大ガウシアン数
        init_cap = N_max
        # 初期値を極小値で埋める
        global_conf = torch.full((B, init_cap), -1e10, device=feat[0].device, dtype=feat[0].dtype)
        global_pts =  torch.full((B, init_cap, 3), -1e4, device=feat[0].device, dtype=feat[0].dtype)
        global_feats = torch.full((B, init_cap, self.downstream_head.raw_gs_dim), -1e10, device=feat[0].device, dtype=feat[0].dtype)
        gaussian_counts = torch.zeros(B, dtype=torch.long) # 現在各バッチで何個埋まっているか

        # 人間メモリー関係
        human_memory_bank = None
        if (
            not inference
            and self.gs_mode in ["human", "both"]
            and self.human_gs_head.num_human_memory_tokens > 0
        ):
            max_humans = views[0]["smpl_mask"].shape[1]
            human_memory_bank = self.human_gs_head.get_initial_human_memory(
                batch_size=B,
                max_humans=max_humans,
                device=feat[0].device,
                dtype=feat[0].dtype,
            )
        ##################################################  
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            smpl_feat_i = smpl_query[i]
            smpl_pos_i = pos_central[i]
            n_humans_i = smpl_feat_i.shape[1]

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)    # [b, 1, 1024]
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)   # coarse pose feat: [b, 1, 768]
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)   # coarse pose feat: [b, 1, 768]
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None

            new_state_feat, dec, _ = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                smpl_feat_i,
                smpl_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]   # After Cross-Attention, refined pose feat: [b, 1, 768]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )   # [b, 256, 1536]

            assert len(dec) == self.dec_depth + 1
            if n_humans_i > 0:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:-n_humans_i].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:-n_humans_i].float(),
                    dec[self.dec_depth][:, :-n_humans_i].float(),
                ]
                smpl_token = dec[self.dec_depth][:, -n_humans_i:].float()
                smpl_token = torch.cat([smpl_token, smpl_tk_mhmr[i]], dim=-1)
            else:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:].float(),
                    dec[self.dec_depth].float(),
                ]
                smpl_token = None
            res = self._downstream_head(
                head_input, shape[i], pos=pos_i, n_humans=n_humans_i, smpl_token=smpl_token, image=views[i]["img"])

            if self.msk_head_flag:
                res['msk'] = msks[i]

            ###  3DGSのSplattingをここで実施する。
            camera_pose_i = res["camera_pose"] # [B, 7]
            extrinsics_i = pose_encoding_to_camera(camera_pose_i).unsqueeze(1)  # [B, 1, 4, 4]
            intrinsics_gt_i = views[i]['camera_intrinsics'].unsqueeze(1)  # [B, 1, 3, 3]
            # intrinsicsの正規化
            # In the multi-view path, `shape[i]` keeps the batch dimension as [B, 2].
            H = shape[i][0][0].item()
            W = shape[i][0][1].item()
            intrinsics_scale = torch.tensor([1.0/W, 1.0/H, 1.0], device=intrinsics_gt_i.device, dtype=intrinsics_gt_i.dtype)
            intrinsics_scale = intrinsics_scale.view(1, 1, 3, 1) # [B, 1, 3, 3] に合わせるためのreshape
            intrinsics_norm_i = intrinsics_gt_i * intrinsics_scale
            if self.gs_mode in ["bg", "both"]:
                # print(msks[i].shape, shape[i].shape)
                # 人間部分を取り除く
                gs_feats_i = res['gs_feats']  # [B, N, D]
                gs_pts_i, pts_conf_i = self._get_viewer_aligned_bg_gs_points_and_conf(
                    res,
                    extrinsics_i.squeeze(1),
                )
                gs_pts_i = self._apply_bg_point_offsets(gs_pts_i, gs_feats_i)
                msks_gs = unpad_image(msks[i].permute(0, 3, 1, 2), (H, W)) # (B, 1, H, W)
                msks_gs = self._dilate_human_mask_for_bg(msks_gs)
                msks_gs = msks_gs.permute(0, 2, 3, 1).reshape(B, -1)  # [B, N]
                bg_valid_mask = msks_gs < getattr(self, "bg_mask_threshold", 0.1) # [B, N]
                # self._print_bg_filter_stats(i, msks_gs, pts_conf_i, bg_valid_mask)
                # ここで人間フィルタと信頼度フィルタをかける
                gs_feats_i, gs_pts_i, gs_conf_i = self._filter_background_gaussian_candidates(
                    gs_feats_i,
                    gs_pts_i,
                    pts_conf_i,
                    bg_valid_mask,
                )

                # Gaussiansの統合
                global_feats, global_pts, global_conf = self._merge_gaussians(global_feats=global_feats,
                                                                        global_pts=global_pts,
                                                                        global_conf=global_conf,
                                                                        new_feats=gs_feats_i,
                                                                        new_pts=gs_pts_i,
                                                                        new_conf=gs_conf_i,
                                                                        voxel_size=getattr(self, "bg_voxel_size", 0.05),
                                                                        N_max=N_max,
                                                                        frame_idx=i)
                depths = global_pts[..., -1] # (8, N)
                densities = global_feats[..., 0].sigmoid() # (8, N)
                opacity = densities
                
                gaussians_i = self.downstream_head.gaussian_adapter.forward(
                    global_pts, # (8, N, 3)
                    depths, # (8, N)
                    opacity, # (8, N)
                    global_feats[..., 1:], # (8, N, 82)
                )

                b, v = extrinsics_i.shape[:2]
                splatted_2d_gaussians_i = self.decoder_splatting(
                    gaussians =  gaussians_i,
                    extrinsics = extrinsics_i,
                    intrinsics = intrinsics_norm_i,
                    near = torch.ones(1, v, device=extrinsics_i.device) * 0.01,
                    far =  torch.ones(1, v, device=extrinsics_i.device) * 100.0,
                    image_shape = (H, W),
                )
                splatted_image = splatted_2d_gaussians_i.color.squeeze(1) # (B, 3, H, W)
                splatted_alpha = splatted_2d_gaussians_i.alpha
                splatted_depth = splatted_2d_gaussians_i.depth
                if splatted_alpha is not None and splatted_alpha.dim() == 4 and splatted_alpha.shape[1] == 1:
                    splatted_alpha = splatted_alpha.squeeze(1) # (B, H, W)
                if splatted_depth is not None and splatted_depth.dim() == 4 and splatted_depth.shape[1] == 1:
                    splatted_depth = splatted_depth.squeeze(1) # (B, H, W)
                ress.append({
                    **res, 'smpl_scores': scores[i], 'smpl_loc': smpl_loc[i], 'splatted_image': splatted_image, 'splatted_alpha': splatted_alpha, 'splatted_depth': splatted_depth, 'gaussians': gaussians_i,
                })

            # ### 人間のガウシアンの推定をする
            if self.gs_mode in ["human"]:
                del res["gs_feats"], res["gs_pts"]
            if self.gs_mode in ["human", "both"] and n_humans_i > 0:
                # Keep the HumanGS query at decoder token + MHMR token = 1792 dims.
                human_smpl_query = smpl_token.flatten(0, 1) # (B*n_humans_i, 1792)

                # HumanGSHead receives flattened human tokens, so camera params must match [B*H, V, ...].
                human_extrinsics_i = extrinsics_i.repeat_interleave(n_humans_i, dim=0).squeeze(1) # [B*H, 4, 4]
                human_intrinsics_i = intrinsics_gt_i.repeat_interleave(n_humans_i, dim=0).squeeze(1) # [B*H, 3, 3]
                patch_size = self.croco_args["patch_size"]

                image_token_i = head_input[-1][:,1:] # [B, H*W, D]
                image_token_i = image_token_i.repeat_interleave(n_humans_i, dim=0) # [B*H, H*W, D]

                # SMPLパラメータをカメラ座標系からワールド座標系に変換する
                smpl_rotmat = res['smpl_rotmat']  # [B, H, 53, 3, 3]

                smpl_rotvec = roma.rotmat_to_rotvec(smpl_rotmat)
                cam_c2w_R = human_extrinsics_i[:, :3, :3]  # [B*H, 3, 3]
                cam_c2w_t = human_extrinsics_i[:, :3, 3]   # [B*H, 3]

                root_rotmat_cam = smpl_rotmat[:, :, 0, :, :].reshape(-1, 3, 3)  # [B*H, 3, 3]
                root_rotmat_world = torch.matmul(cam_c2w_R, root_rotmat_cam)
                root_pose_world = roma.rotmat_to_rotvec(root_rotmat_world) # [B*H, 3]

                trans_cam = res['smpl_transl'].reshape(-1, 3)  # [B*H, 3]
                trans_world = torch.matmul(cam_c2w_R, trans_cam.unsqueeze(-1)).squeeze(-1) + cam_c2w_t # [B*H, 3]

                smpl_params = {}
                smpl_params['betas'] = res['smpl_shape'].reshape(-1, res['smpl_shape'].shape[-1])  # [B*H, 10]
                smpl_params['root_pose'] = root_pose_world # [B*H, 3]
                smpl_params['body_pose'] = smpl_rotvec[:, :, 1:22, :].reshape(-1, 21, 3)
                smpl_params['lhand_pose'] = smpl_rotvec[:, :, 22:37, :].reshape(-1, 15, 3) # [B*H, 15, 3]
                smpl_params['rhand_pose'] = smpl_rotvec[:, :, 37:52, :].reshape(-1, 15, 3) # [B*H, 15, 3]
                smpl_params['jaw_pose'] = smpl_rotvec[:, :, 52, :].reshape(-1, 3)  # [B*H, 3]
                smpl_params['expr'] = res['smpl_expression'].reshape(-1, res['smpl_expression'].shape[-1])  # [B*H, 10]
                smpl_params['leye_pose'] = torch.zeros_like(smpl_params['jaw_pose'])  # [B*H, 3]
                smpl_params['reye_pose'] = torch.zeros_like(smpl_params['jaw_pose'])  # [B*H, 3]
                smpl_params['trans'] = trans_world # [B*H, 3]

                if self.human_gs_teacher_force_gt_smpl and not inference:
                    total_humans = human_smpl_query.shape[0]
                    root_pose_cam = views[i]["smplx_root_pose"].reshape(total_humans, 3).to(device=cam_c2w_R.device, dtype=cam_c2w_R.dtype)
                    root_rotmat_world = torch.matmul(cam_c2w_R, roma.rotvec_to_rotmat(root_pose_cam))
                    head_cam = views[i]["smpl_transl"].reshape(total_humans, 3).to(device=cam_c2w_R.device, dtype=cam_c2w_R.dtype)
                    head_world = torch.matmul(cam_c2w_R, head_cam.unsqueeze(-1)).squeeze(-1) + cam_c2w_t
                    smpl_params = {
                        "betas": views[i]["smplx_shape"].reshape(total_humans, -1)[:, :res["smpl_shape"].shape[-1]].to(device=cam_c2w_R.device, dtype=cam_c2w_R.dtype),
                        "root_pose": roma.rotmat_to_rotvec(root_rotmat_world),
                        "body_pose": views[i]["smplx_body_pose"].reshape(total_humans, 21, 3).to(device=cam_c2w_R.device, dtype=cam_c2w_R.dtype),
                        "lhand_pose": views[i]["smplx_left_hand_pose"].reshape(total_humans, 15, 3).to(device=cam_c2w_R.device, dtype=cam_c2w_R.dtype),
                        "rhand_pose": views[i]["smplx_right_hand_pose"].reshape(total_humans, 15, 3).to(device=cam_c2w_R.device, dtype=cam_c2w_R.dtype),
                        "jaw_pose": views[i]["smplx_jaw_pose"].reshape(total_humans, 3).to(device=cam_c2w_R.device, dtype=cam_c2w_R.dtype),
                        "leye_pose": views[i]["smplx_leye_pose"].reshape(total_humans, 3).to(device=cam_c2w_R.device, dtype=cam_c2w_R.dtype),
                        "reye_pose": views[i]["smplx_reye_pose"].reshape(total_humans, 3).to(device=cam_c2w_R.device, dtype=cam_c2w_R.dtype),
                        "expr": torch.zeros(total_humans, res['smpl_expression'].shape[-1], device=cam_c2w_R.device, dtype=cam_c2w_R.dtype),
                        "trans": head_world,
                    }

                # Train時はパディング人間が混ざるため、validな人だけHumanGSに通す。
                total_humans = human_smpl_query.shape[0]
                valid_human_mask = torch.ones(total_humans, dtype=torch.bool, device=human_smpl_query.device)
                if not inference and "smpl_mask" in views[i]:
                    valid_human_mask = views[i]["smpl_mask"].reshape(-1).to(device=human_smpl_query.device).bool()

                human_rgb = torch.zeros(
                    (total_humans, H, W, 3), device=human_smpl_query.device, dtype=human_smpl_query.dtype
                )
                human_mask = torch.zeros(
                    (total_humans, H, W, 1), device=human_smpl_query.device, dtype=human_smpl_query.dtype
                )

                if torch.any(valid_human_mask):
                    smpl_params_valid = {k: v[valid_human_mask] for k, v in smpl_params.items()}
                    # Use the current GPU view here so boolean masking stays on the same device.
                    raw_images_i = views[i]["img"].repeat_interleave(n_humans_i, dim=0)
                    human_memory_i = None
                    if human_memory_bank is not None:
                        human_memory_i = human_memory_bank.reshape(
                            total_humans,
                            self.human_gs_head.num_human_memory_tokens,
                            self.human_gs_head.hidden_dim,
                        )
                    human_gs_result = self.human_gs_head(
                        smpl_query=human_smpl_query[valid_human_mask],
                        image_tokens=image_token_i[valid_human_mask],
                        smpl_params=smpl_params_valid,
                        extrinsics=human_extrinsics_i[valid_human_mask],
                        intrinsics=human_intrinsics_i[valid_human_mask],
                        height=H,
                        width=W,
                        debug_image=raw_images_i[valid_human_mask],
                        raw_image=raw_images_i[valid_human_mask],
                        patch_size=patch_size,
                        human_memory=human_memory_i[valid_human_mask] if human_memory_i is not None else None,
                    )
                    if human_memory_i is not None:
                        memory_update_rate = 0.5
                        next_memory = human_memory_i.clone()
                        old_valid_memory = human_memory_i[valid_human_mask]
                        updated_valid_memory = human_gs_result["updated_human_memory"]
                        next_memory[valid_human_mask] = (
                            old_valid_memory
                            + memory_update_rate * (updated_valid_memory - old_valid_memory)
                        )
                        human_memory_bank = next_memory.view_as(human_memory_bank)
                    human_rgb[valid_human_mask] = human_gs_result['comp_rgb']
                    human_mask[valid_human_mask] = human_gs_result['comp_mask']
                    human_gaussians = human_gs_result['gaussians']

                human_rgb = human_rgb.permute(0, 3, 1, 2) # [BH, 3, 288, 512]
                human_mask = human_mask.permute(0, 3, 1, 2) # [BH, 1, 288, 512]
                # print("human_rgb shape:", human_rgb.shape)
                # print("human_mask shape:", human_mask.shape)

                ## デバッグ用の画像保存
                # debug_dir =  "debug/human_gs/"
                # os.makedirs(debug_dir, exist_ok=True)

                # valid_human_mask_2d = valid_human_mask.view(views[i]["img"].shape[0], n_humans_i)
                # for b in range(views[i]["img"].shape[0]):
                #     for h in range(n_humans_i):
                #         if not valid_human_mask_2d[b, h].item():
                #             continue
                #         save_image(human_rgb[b*n_humans_i + h], f"{debug_dir}/{b}_{i}_rgb_{h}.png")
                #         save_image(human_mask[b*n_humans_i + h], f"{debug_dir}/{b}_{i}_mask_{h}.png")
                # # print("original image shape:", views[i]["img"].shape)
                # for b in range(views[i]["img"].shape[0]):
                #     save_image(views[i]["img"][b], f"{debug_dir}/{b}_{i}_original.png")

                ress.append({
                    **res, 'smpl_scores': scores[i], 'smpl_loc': smpl_loc[i], 'human_rgb': human_rgb, 'human_mask': human_mask,
                    "human_gaussians": human_gaussians if torch.any(valid_human_mask) else None,
                })
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def _dilate_human_mask_for_bg(self, mask):
            dilation = int(getattr(self, "bg_mask_dilation", 0) or 0)
            if dilation <= 0:
                return mask
            return F.max_pool2d(mask, kernel_size=2 * dilation + 1, stride=1, padding=dilation)

    def _get_viewer_aligned_bg_gs_points_and_conf(self, res, camera_c2w):
            if "pts3d_in_self_view" in res and "conf_self" in res:
                pts = geotrf(camera_c2w, res["pts3d_in_self_view"]).flatten(1, 2)
                return pts, res["conf_self"]
            return res["gs_pts"], res.get("conf", None)

    def _apply_bg_point_offsets(self, gs_pts, gs_feats):
            if not getattr(self, "bg_point_offset_enabled", False):
                return gs_pts
            if not hasattr(self, "bg_point_offset_head"):
                return gs_pts
            max_offset = float(getattr(self, "bg_point_offset_max_ratio", 0.5)) * float(
                getattr(self, "bg_voxel_size", 0.01)
            )
            if max_offset <= 0:
                return gs_pts
            offsets = torch.tanh(self.bg_point_offset_head(gs_feats.float())).to(dtype=gs_pts.dtype)
            return gs_pts + offsets * max_offset

    def _print_bg_filter_stats(self, frame_idx, msks_gs, pts_conf, bg_valid_mask):
            if pts_conf is None:
                return
            conf_flat = pts_conf.reshape(bg_valid_mask.shape[0], -1)
            if self.gs_point_conf_threshold is None:
                conf_valid_mask = torch.ones_like(bg_valid_mask, dtype=torch.bool)
            else:
                conf_valid_mask = conf_flat > self.gs_point_conf_threshold
            q = torch.tensor([0.5, 0.9, 0.95, 0.99], device=conf_flat.device)
            for b in range(bg_valid_mask.shape[0]):
                conf_q50, conf_q90, conf_q95, conf_q99 = torch.quantile(conf_flat[b].float(), q)
                mask_b = msks_gs[b].float()
                print(
                    "[BG filter] "
                    f"frame={frame_idx} batch={b} "
                    f"mask_valid={bg_valid_mask[b].sum().item()} "
                    f"conf_valid={conf_valid_mask[b].sum().item()} "
                    f"both_valid={(bg_valid_mask[b] & conf_valid_mask[b]).sum().item()} "
                    f"conf_q50={conf_q50.item():.4f} "
                    f"conf_q90={conf_q90.item():.4f} "
                    f"conf_q95={conf_q95.item():.4f} "
                    f"conf_q99={conf_q99.item():.4f} "
                    f"mask_min={mask_b.min().item():.4f} "
                    f"mask_mean={mask_b.mean().item():.4f} "
                    f"mask_max={mask_b.max().item():.4f}",
                    flush=True,
                )

    def _filter_background_gaussian_candidates(self, gs_feats, gs_pts, pts_conf, bg_valid_mask):
            if pts_conf is None:
                raise ValueError("Background GS merge now requires point-cloud confidence `conf`.")

            B = gs_feats.shape[0]
            gs_conf = pts_conf.reshape(B, -1)
            threshold = self.gs_point_conf_threshold
            if threshold is not None:
                bg_valid_mask = bg_valid_mask & (gs_conf > threshold)

            gs_feats = gs_feats.masked_fill(~bg_valid_mask[..., None], -1e10)
            gs_pts = gs_pts.masked_fill(~bg_valid_mask[..., None], -1e4)
            gs_conf = gs_conf.masked_fill(~bg_valid_mask, -1e10)

            return gs_feats, gs_pts, gs_conf

    def _merge_gaussians(self, global_feats, global_pts, global_conf, new_feats, new_pts, new_conf, voxel_size=0.05, N_max=None, frame_idx=None):
            B, current_N_max = global_feats.shape[:2]
            N_max = N_max or current_N_max
            
            # 1. 連結
            combined_conf = torch.cat([global_conf, new_conf], dim=1)
            combined_means = torch.cat([global_pts, new_pts], dim=1)
            combined_feats = torch.cat([global_feats, new_feats], dim=1)

            out_feats, out_pts, out_conf = [], [], []

            for b in range(B):
                new_valid = new_conf[b] > -1e9
                if torch.any(new_valid):
                    new_voxels = torch.unique(
                        torch.round(new_pts[b][new_valid] / voxel_size).int(),
                        dim=0,
                    ).shape[0]
                else:
                    new_voxels = 0

                combined_valid = combined_conf[b] > -1e9
                if torch.any(combined_valid):
                    merged_valid_voxels = torch.unique(
                        torch.round(combined_means[b][combined_valid] / voxel_size).int(),
                        dim=0,
                    ).shape[0]
                else:
                    merged_valid_voxels = 0

                v_indices = torch.round(combined_means[b] / voxel_size).int()
                # uniqueを使ってvoxel内の重複を確認
                unique_voxels, inverse_indices = torch.unique(v_indices, return_inverse=True, dim=0)
                num_voxels = unique_voxels.shape[0]

                # --- ここが「置き換え」の核心 ---
                # 各voxel内で、combined_confが最大値をとる「元のインデックス」を取得する
                conf_voxel_max, voxel_argmax = scatter_max(combined_conf[b], inverse_indices, dim=0, dim_size=num_voxels) # [N_unique]
                conf_exp = torch.exp(combined_conf[b] - conf_voxel_max[inverse_indices]) # [N_total]
                voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0, dim_size=num_voxels) # [N_unique]
                weights = conf_exp / (voxel_weights[inverse_indices] + 1e-8) # [N_total]

                weighted_voxel_means = scatter_add(combined_means[b] * weights.unsqueeze(-1), inverse_indices, dim=0, dim_size=num_voxels) # [N_unique, 3]
                merge_point_mode = getattr(self, "bg_merge_point_mode", "max_conf")
                if merge_point_mode == "max_conf":
                    voxel_means = combined_means[b][voxel_argmax] # [N_unique, 3]
                elif merge_point_mode == "weighted_mean":
                    voxel_means = weighted_voxel_means
                else:
                    raise ValueError(f"Unknown bg_merge_point_mode: {merge_point_mode}")
                voxel_feats = scatter_add(combined_feats[b] * weights.unsqueeze(-1), inverse_indices, dim=0, dim_size=num_voxels) # [N_unique, d_gs]
                voxel_conf = conf_voxel_max # [N_unique]

                # 2. N_maxへのフィッティング (Top-K)
                if num_voxels > N_max:
                    # 自信度を確率分布に変換 (Softmaxなどで正規化)
                    # Tを大きくするとより一様(背景も拾う)になり、小さくするとTop-Kに近づく
                    keep_idx = torch.randperm(num_voxels, device=combined_conf.device)[:N_max]
                    
                    v_pts = voxel_means[keep_idx]
                    v_feats = voxel_feats[keep_idx]
                    v_conf = voxel_conf[keep_idx]
                else:
                    pad_n = N_max - num_voxels
                    v_pts = torch.cat([voxel_means, torch.full((pad_n, 3), -1e4, device=global_pts.device, dtype=global_pts.dtype)], dim=0)
                    v_feats = torch.cat([voxel_feats, torch.full((pad_n, global_feats.shape[2]), -1e10, device=global_feats.device, dtype=global_feats.dtype)], dim=0)
                    v_conf = torch.cat([voxel_conf, torch.full((pad_n,), -1e10, device=global_conf.device, dtype=global_conf.dtype)], dim=0)

                out_pts.append(v_pts)
                out_feats.append(v_feats)
                out_conf.append(v_conf)

                frame_msg = f"frame={frame_idx}" if frame_idx is not None else "frame=?"
                kept_valid = (v_conf > -1e9).sum().item()
                print(
                    "[BG voxel] "
                    f"{frame_msg} batch={b} "
                    f"new_pixels={new_conf.shape[1]} "
                    f"new_valid={new_valid.sum().item()} "
                    f"new_voxels={new_voxels} "
                    f"merged_valid_before={combined_valid.sum().item()} "
                    f"merged_voxels={merged_valid_voxels} "
                    f"kept_valid={kept_valid}/{N_max}",
                    flush=True,
                )

            return torch.stack(out_feats), torch.stack(out_pts), torch.stack(out_conf)


    def _forward_impl_naive(self, views, ret_state=False, inference=False):
        shape, feat_ls, pos, mhmr_feat_ls = self._encode_views_mhmr(views)
        feat = feat_ls[-1]
        mhmr_feat = mhmr_feat_ls[-1]

        scores, smpl_tk_mhmr, pos_mhmr, smpl_loc, msks = self.smpl_tokenizer_mhmr(
            mhmr_feat, pos, views, inference)

        # naive CUT3R+MHMR
        smpl_query = smpl_tk_mhmr
        pos_central = pos_mhmr

        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)  # [b, 256, 1536]
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            smpl_feat_i = smpl_query[i]
            smpl_pos_i = pos_central[i]
            n_humans_i = smpl_feat_i.shape[1]

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)    # [b, 1, 1024]
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)   # coarse pose feat: [b, 1, 768]
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)   # coarse pose feat: [b, 1, 768]
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None

            new_state_feat, dec, _ = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                None,
                None,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )   # [b, 256, 1536]

            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            if n_humans_i > 0:
                smpl_token = smpl_feat_i # used for naive CUT3R+MHMR
            else:
                smpl_token = None
            res = self._downstream_head(
                head_input, shape[i], pos=pos_i, n_humans=n_humans_i, smpl_token=smpl_token)

            ress.append({
                **res, 'smpl_scores': scores[i], 'smpl_loc': smpl_loc[i]})
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False, inference=False):
        if self.output_mode == "naive":
            if ret_state:
                ress, views, state_args = self._forward_impl_naive(views, ret_state=ret_state, inference=inference)
                return ARCroco3DStereoOutput(ress=ress, views=views), state_args
            else:
                ress, views = self._forward_impl_naive(views, ret_state=ret_state)
                return ARCroco3DStereoOutput(ress=ress, views=views)
        else:
            if ret_state:
                ress, views, state_args = self._forward_impl(views, ret_state=ret_state, inference=inference)
                return ARCroco3DStereoOutput(ress=ress, views=views), state_args
            else:
                ress, views = self._forward_impl(views, ret_state=ret_state)
                return ARCroco3DStereoOutput(ress=ress, views=views)


    def forward_recurrent_lighter(
        self,
        views,
        device,
        ret_state=False,
        use_ttt3r=False,
        save_all_bg_gaussians=False,
        output_callback=None,
        keep_outputs=True,
    ):
        ress = []
        all_state_args = []
        # last_smpl_tk = None
        # last_smpl_id = None
        # max_smpl_id = -1
        # reset_mask = False

        max_humans = 10
        recurrent_human_slots = None
        human_memory_bank = None
        reset_mask = False

        ####### 全体のガウシアンを保持する。
        from dust3r.heads.gaussian_utils.types import Gaussians
        B = views[0]["img"].shape[0]
        N_max = getattr(self, "bg_gaussian_max", 1000000) # 最大ガウシアン数
        init_cap = N_max
        # 初期値を極小値で埋める
        global_conf = torch.full((B, init_cap), -1e10, device=device, dtype=views[0]["img"].dtype)
        global_pts =  torch.full((B, init_cap, 3), -1e4, device=device, dtype=views[0]["img"].dtype)
        global_feats = torch.full((B, init_cap, self.downstream_head.raw_gs_dim), -1e10, device=device, dtype=views[0]["img"].dtype)
        for i, _view in enumerate(views):
            view = to_gpu(_view, device)
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None

            shape = shapes
            feat_i = img_out[-1]
            pos_i = img_pos
            
            # MHMR vit
            imgs_mhmr = view["img_mhmr"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            imgs_mhmr = imgs_mhmr.view(
                -1, *imgs_mhmr.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            selected_imgs_mhmr = imgs_mhmr[img_masks_flat]
            if selected_imgs_mhmr.size(0) > 0:
                mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
                selected_imgs_mhmr = (selected_imgs_mhmr * 0.5 + 0.5 - mean) / std
                mhmr_img_out = [self.backbone(selected_imgs_mhmr)] # image[bs, 3, h, w] -> image feature [bs, h_patches*w_patches, D]
            feat_mhmr_i = mhmr_img_out[-1]

            # MHMR smpl tokenizer
            n_patch_mhmr = self.bb_token_res
            scores = self.downstream_head.detect_mhmr(feat_mhmr_i) #(num_view * bs, h_patches*w_patches, 1)
            scores = rearrange(scores, "b (nh nw) c -> b c nh nw", nh=n_patch_mhmr, nw=n_patch_mhmr) # [num_view * bs, h_nb_patches * w_nb_patches, 1] -> [num_view * bs, 1, h, w]
            if self.msk_head_flag:
                msks = self.downstream_head.segment(feat_mhmr_i) # low-res mask
                msks = rearrange(msks, "b (nh nw) c -> b c nh nw", nh=n_patch_mhmr, nw=n_patch_mhmr)
                msks = F.pixel_shuffle(msks, self.bb_patch_size)  # (num_view * bs, 1, h, w)
                msks = msks.permute((0, 2, 3, 1)) # (num_view * bs, h, w, 1)
            feat_mhmr_i = rearrange(feat_mhmr_i, "b (nh nw) c -> b nh nw c", nh=n_patch_mhmr, nw=n_patch_mhmr) # head token extraction: (num_view * bs, h, w, 1024)

            scores = nms(scores, kernel=3) # (num_view * bs, 1, h, w)
            scores = scores.permute((0, 2, 3, 1)) # (num_view * bs, h, w, 1)
            idx = apply_threshold(0.3, scores)
            img_id, h_id, w_id = idx[0], idx[1], idx[2]

            # Head token and offset
            feat_central_mhmr = feat_mhmr_i[img_id, h_id, w_id] # (nvh, 1024)
            offset = self.downstream_head.mlp_offset(feat_central_mhmr)# [nhv,2]
            # Distance for estimating the 3D location in 3D space
            loc = torch.stack([w_id, h_id]).permute(1,0) # x,y
            loc = (loc + 0.5 + offset) * self.bb_patch_size # Moving to higher res the location of the pelvis

            smpl_tk_mhmr = feat_central_mhmr.unsqueeze(0)   # use mhmr vit token

            # CUT3R smpl tokenizer
            n_patch_cut3r = shape[0] // self.croco_args['patch_size'] # H,W
            feat_cut3r_i = rearrange(
                feat_i, "b (nh nw) c -> b nh nw c", nh=n_patch_cut3r[0], nw=n_patch_cut3r[1]) # (num_view * bs, h, w, 1024)
            pos_cut3r_i = rearrange(
                pos_i, "b (nh nw) c -> b nh nw c", nh=n_patch_cut3r[0], nw=n_patch_cut3r[1]) # (num_view * bs, h, w, 2)
            
            loc_cut3r = unpad_uv(loc, self.mhmr_img_res, *shape[0])
            smpl_uv_cut3r = (loc_cut3r // self.croco_args['patch_size']).int()
            w_id_cut3r, h_id_cut3r = smpl_uv_cut3r.T
            feat_central_cut3r = feat_cut3r_i[img_id, h_id_cut3r, w_id_cut3r] # (nvh, 1024)
            pos_central_cut3r = pos_cut3r_i[img_id, h_id_cut3r, w_id_cut3r] # (nvh, 2)

            smpl_tk_cut3r = feat_central_cut3r.unsqueeze(0)
            smpl_pos_cut3r = pos_central_cut3r.unsqueeze(0)

            # fuse CUT3R and MHMR smpl tokens
            fused_tk = torch.cat([smpl_tk_mhmr, smpl_tk_cut3r], dim=-1) #(1, nvh, 2048)
            fused_tk = self.downstream_head.mlp_fuse(fused_tk) # (1, nvh, 768)

            smpl_feat_i = fused_tk # (1,nvh, 768)
            smpl_pos_i = smpl_pos_cut3r # (1,nvh, 2)

            ###############################
            # ここでマッチング+並び替えを実施する
            reset_prev = (
                bool(reset_mask.any().item())
                if torch.is_tensor(reset_mask)
                else bool(reset_mask)
            )

            if recurrent_human_slots is None or reset_prev:
                recurrent_human_slots = self._init_recurrent_human_slot_state(
                    device=device,
                    feat_dtype=fused_tk.dtype,
                    pos_dtype=smpl_pos_cut3r.dtype,
                    max_humans=max_humans,
                )
                if (
                    self.gs_mode in ["human", "both"]
                    and self.human_gs_head.num_human_memory_tokens > 0
                ):
                    human_memory_bank = self.human_gs_head.get_initial_human_memory(
                        batch_size=1,
                        max_humans=max_humans,
                        device=device,
                        dtype=fused_tk.dtype,
                    )
                else:
                    human_memory_bank = None

            (
                recurrent_human_slots,
                smpl_feat_i,
                smpl_pos_i,
                smpl_tk_mhmr_slots,
                smpl_loc_slots,
                frame_valid_slots,
                smpl_id_slots,
            ) = self._match_recurrent_humans_to_slots(
                recurrent_human_slots,
                curr_smpl_feat=fused_tk,
                curr_smpl_pos=smpl_pos_cut3r,
                curr_mhmr_feat=smpl_tk_mhmr,
                curr_smpl_loc=loc.unsqueeze(0),
                frame_index=i,
            )

            n_humans_i = smpl_feat_i.shape[1]
            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                # if i == 0 or reset_mask:
                if i == 0 or reset_prev:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, cross_attn_states = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                smpl_feat_i,
                smpl_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                use_ttt3r=use_ttt3r,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            if n_humans_i > 0:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:-n_humans_i].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:-n_humans_i].float(),
                    dec[self.dec_depth][:, :-n_humans_i].float(),
                ]
                smpl_token = dec[self.dec_depth][:, -n_humans_i:].float()
                smpl_token_cat = torch.cat([smpl_token, smpl_tk_mhmr_slots], dim=-1)
            else:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:].float(),
                    dec[self.dec_depth].float(),
                ]
                smpl_token = None
                smpl_token_cat = None
            res = self._downstream_head(
                head_input,
                shape,
                pos=pos_i,
                n_humans=n_humans_i,
                smpl_token=smpl_token_cat,
                image=view["img"],
            )
            ###  3DGSの追加をここで行う
            camera_pose_i = res["camera_pose"] # [B, 7]
            extrinsics_i = pose_encoding_to_camera(camera_pose_i).unsqueeze(1)  # [B, 1, 4, 4]
            intrinsics_gt_i = view["camera_intrinsics"].unsqueeze(1)  # [B, 1, 3, 3]
            # intrinsicsの正規化
            # In the lighter recurrent path, `shape` is [B, 2] for the current view.
            H = shape[0][0].item()
            W = shape[0][1].item()
            if self.gs_mode in ["bg", "both"]:
                gs_feats_i = res['gs_feats']  # [B, N, D]
                gs_pts_i, pts_conf_i = self._get_viewer_aligned_bg_gs_points_and_conf(
                    res,
                    extrinsics_i.squeeze(1),
                )
                gs_pts_i = self._apply_bg_point_offsets(gs_pts_i, gs_feats_i)

                target_hw = tuple(int(x) for x in shape[0].tolist()) # (H, W)
                msks_gs = unpad_image(msks.permute(0, 3, 1, 2), target_hw) # (B, 1, H, W)
                msks_gs = self._dilate_human_mask_for_bg(msks_gs)
                msks_gs = msks_gs.permute(0, 2, 3, 1).reshape(B, -1)  # [B, N]
                bg_valid_mask = msks_gs < getattr(self, "bg_mask_threshold", 0.05) # [B, N]
                # self._print_bg_filter_stats(i, msks_gs, pts_conf_i, bg_valid_mask)
                gs_feats_i, gs_pts_i, gs_conf_i = self._filter_background_gaussian_candidates(
                    gs_feats_i,
                    gs_pts_i,
                    pts_conf_i,
                    bg_valid_mask,
                )

                update_bg = view.get("update", None)
                if update_bg is not None:
                    update_bg = update_bg.reshape(B).to(device=gs_conf_i.device).bool()
                    gs_feats_i = gs_feats_i.masked_fill(~update_bg[:, None, None], -1e10)
                    gs_pts_i = gs_pts_i.masked_fill(~update_bg[:, None, None], -1e10)
                    gs_conf_i = gs_conf_i.masked_fill(~update_bg[:, None], -1e10)
                # Gaussiansの統合
                global_feats, global_pts, global_conf = self._merge_gaussians(global_feats=global_feats,
                                                                        global_pts=global_pts,
                                                                        global_conf=global_conf,
                                                                        new_feats=gs_feats_i,
                                                                        new_pts=gs_pts_i,
                                                                        new_conf=gs_conf_i,
                                                                        voxel_size=getattr(self, "bg_voxel_size", 0.05),
                                                                        N_max=N_max,
                                                                        frame_idx=i)
                

                if save_all_bg_gaussians or i == len(views) - 1:
                    depths = global_pts[..., -1] # (8, N)
                    densities = global_feats[..., 0].sigmoid() # (8, N)
                    opacity = densities

                    gaussians_i = self.downstream_head.gaussian_adapter.forward(
                        global_pts, # (8, N, 3)
                        depths, # (8, N)
                        opacity, # (8, N)
                        global_feats[..., 1:], # (8, N, 82)
                    )
                    res.update({'gaussians': gaussians_i,})

            ### 人間のGSを追加する
            if self.gs_mode in ["human", "both"] and n_humans_i > 0:
                human_smpl_query = smpl_token_cat.flatten(0, 1)  # (B*n_humans_i, 1792)

                human_extrinsics_i = extrinsics_i.repeat_interleave(n_humans_i, dim=0).squeeze(1)
                human_intrinsics_i = intrinsics_gt_i.repeat_interleave(n_humans_i, dim=0).squeeze(1)
                patch_size = self.croco_args["patch_size"]

                image_token_i = head_input[-1][:, 1:]
                image_token_i = image_token_i.repeat_interleave(n_humans_i, dim=0)

                smpl_rotmat = res["smpl_rotmat"]
                smpl_rotvec = roma.rotmat_to_rotvec(smpl_rotmat)

                cam_c2w_R = human_extrinsics_i[:, :3, :3]
                cam_c2w_t = human_extrinsics_i[:, :3, 3]

                root_rotmat_cam = smpl_rotmat[:, :, 0, :, :].reshape(-1, 3, 3)
                root_rotmat_world = torch.matmul(cam_c2w_R, root_rotmat_cam)
                root_pose_world = roma.rotmat_to_rotvec(root_rotmat_world)

                trans_cam = res["smpl_transl"].reshape(-1, 3)
                trans_world = (
                    torch.matmul(cam_c2w_R, trans_cam.unsqueeze(-1)).squeeze(-1)
                    + cam_c2w_t
                )

                smpl_params = {}
                smpl_params["betas"] = res["smpl_shape"].reshape(-1, res["smpl_shape"].shape[-1])
                smpl_params["root_pose"] = root_pose_world
                smpl_params["body_pose"] = smpl_rotvec[:, :, 1:22, :].reshape(-1, 21, 3)
                smpl_params["lhand_pose"] = smpl_rotvec[:, :, 22:37, :].reshape(-1, 15, 3)
                smpl_params["rhand_pose"] = smpl_rotvec[:, :, 37:52, :].reshape(-1, 15, 3)
                smpl_params["jaw_pose"] = smpl_rotvec[:, :, 52, :].reshape(-1, 3)
                smpl_params["expr"] = res["smpl_expression"].reshape(-1, res["smpl_expression"].shape[-1])
                smpl_params["leye_pose"] = torch.zeros_like(smpl_params["jaw_pose"])
                smpl_params["reye_pose"] = torch.zeros_like(smpl_params["jaw_pose"])
                smpl_params["trans"] = trans_world

                # current frame で見えている slot だけ HumanGS に流す
                valid_human_mask = frame_valid_slots.reshape(-1)

                skip_render_outputs = getattr(self, "skip_inference_render_outputs", False)
                human_rgb = None
                human_mask = None
                if not skip_render_outputs:
                    human_rgb = torch.zeros(
                        (human_smpl_query.shape[0], H, W, 3),
                        device=human_smpl_query.device,
                        dtype=human_smpl_query.dtype,
                    )
                    human_mask = torch.zeros(
                        (human_smpl_query.shape[0], H, W, 1),
                        device=human_smpl_query.device,
                        dtype=human_smpl_query.dtype,
                    )
                human_gaussians = None

                if torch.any(valid_human_mask):
                    smpl_params_valid = {
                        k: v[valid_human_mask] for k, v in smpl_params.items()
                    }
                    raw_images_i = view["img"].repeat_interleave(n_humans_i, dim=0)

                    human_memory_i = None
                    active_human_slots = human_smpl_query.shape[0]
                    if human_memory_bank is not None:
                        human_memory_i = human_memory_bank[:, :active_human_slots].reshape(
                            active_human_slots,
                            self.human_gs_head.num_human_memory_tokens,
                            self.human_gs_head.hidden_dim,
                        )

                    human_gs_result = self.human_gs_head(
                        smpl_query=human_smpl_query[valid_human_mask],
                        image_tokens=image_token_i[valid_human_mask],
                        smpl_params=smpl_params_valid,
                        extrinsics=human_extrinsics_i[valid_human_mask],
                        intrinsics=human_intrinsics_i[valid_human_mask],
                        height=H,
                        width=W,
                        debug_image=raw_images_i[valid_human_mask],
                        raw_image=raw_images_i[valid_human_mask],
                        patch_size=patch_size,
                        human_memory=(
                            human_memory_i[valid_human_mask]
                            if human_memory_i is not None
                            else None
                        ),
                        render=not skip_render_outputs,
                    )

                    if human_memory_i is not None:
                        memory_update_rate = 0.5
                        next_memory = human_memory_i.clone()
                        old_valid_memory = human_memory_i[valid_human_mask]
                        updated_valid_memory = human_gs_result["updated_human_memory"]
                        next_memory[valid_human_mask] = (
                            old_valid_memory
                            + memory_update_rate * (updated_valid_memory - old_valid_memory)
                        )
                        human_memory_bank[:, :active_human_slots] = next_memory.view(
                            1,
                            active_human_slots,
                            self.human_gs_head.num_human_memory_tokens,
                            self.human_gs_head.hidden_dim,
                        )

                    if not skip_render_outputs:
                        human_rgb[valid_human_mask] = human_gs_result["comp_rgb"]
                        human_mask[valid_human_mask] = human_gs_result["comp_mask"]
                    human_gaussians = human_gs_result["gaussians"]

                if skip_render_outputs:
                    res.update({"human_gaussians": human_gaussians})
                else:
                    # 外に出すときは current frame で visible な人だけに戻す
                    human_rgb = human_rgb[valid_human_mask].permute(0, 3, 1, 2)
                    human_mask = human_mask[valid_human_mask].permute(0, 3, 1, 2)

                    res.update(
                        {
                            "human_rgb": human_rgb,
                            "human_mask": human_mask,
                            "human_gaussians": human_gaussians,
                        }
                    )


            # # tracking
            # num_miss_match0 = 0
            # if last_smpl_tk is not None and smpl_token is not None:
            #     cost_mat = -torch.cdist(last_smpl_tk, smpl_token, p=2)
            #     cost_mat = log_optimal_transport(
            #         cost_mat, alpha=torch.tensor(-10.0, device=device), iters=20)
            #     matches = cost_mat[:, :-1, :-1]
            #     max0, max1 = matches.max(2), matches.max(1)
            #     indices0, indices1 = max0.indices, max1.indices
            #     mutual0 = torch.arange(
            #         indices0.shape[1], device=device
            #         )[None] == indices1.gather(1, indices0)
            #     mutual1 = torch.arange(
            #         indices1.shape[1], device=device
            #         )[None] == indices0.gather(1, indices1)
            #     zero = matches.new_tensor(0)
            #     mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            #     mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)

            #     match_threshold = 0.2
            #     valid0 = mutual0 & (mscores0 > match_threshold) # 1,n
            #     valid1 = mutual1 & valid0.gather(1, indices1) # 1,m
            #     # get the final matching indices, invalid matches set to -1
            #     indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1)) # [1, n] current frame matches for last frame
            #     indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1)) # [1, m] last frame matches for current frame

            #     smpl_id = indices1.new_full(indices1.shape, -1) # 1,m
            #     valid_match1 = indices1[valid1]
            #     if valid_match1.numel() > 0:
            #         smpl_id[valid1] = last_smpl_id.gather(1, valid_match1[None]).flatten()

            #     num_miss_match0 = int((~valid0).sum())
            #     num_new_persons = len(smpl_id[~valid1])
            #     if num_new_persons > 0:
            #         new_ids = torch.arange(
            #             max_smpl_id + 1,
            #             max_smpl_id + 1 + num_new_persons,
            #             device=device
            #         )
            #         smpl_id[~valid1] = new_ids
            #         max_smpl_id += num_new_persons
            # else:
            #     # first frame with humans
            #     if smpl_token is not None:
            #         smpl_id = torch.arange(n_humans_i, device=device)[None]  # (1, nvh)
            #         max_smpl_id = n_humans_i - 1
            #     else:
            #         smpl_id = None

            # if smpl_token is not None:
            #     if num_miss_match0 > 0:
            #         miss_match_id0 = last_smpl_id[~valid0][None]
            #         miss_match_tk0 = last_smpl_tk[~valid0][None]
            #         last_smpl_id = torch.cat([smpl_id, miss_match_id0], dim=1)
            #         last_smpl_tk = torch.cat([smpl_token, miss_match_tk0], dim=1)
            #     else:
            #         last_smpl_tk = smpl_token.clone()
            #         last_smpl_id = smpl_id.clone()

            # if smpl_id is not None:
            #     res['smpl_id'] = smpl_id

            res["smpl_id"] = smpl_id_slots
            res["smpl_loc"] = smpl_loc_slots

            res, visible_idx = self._compact_recurrent_visible_humans(
                res, frame_valid_slots
            )

            if self.msk_head_flag:
                res['msk'] = msks
            if output_callback is not None:
                output_callback(i, res, view)
            if keep_outputs:
                res_cpu = to_cpu({**res, 'smpl_scores': scores})
                ress.append(res_cpu)
            # ress.append(res)

            # updating the state and memory
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            if use_ttt3r and i != 0 and not reset_mask:
                cross_attn_states = rearrange(torch.cat(cross_attn_states, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)').mean(dim=(-1, -2))
                update_mask_state = update_mask * torch.sigmoid(cross_attn_states)[..., None]
            else:
                update_mask_state = update_mask

            state_feat = new_state_feat * update_mask_state + state_feat * (
                1 - update_mask_state
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
           
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward_recurrent_lighter_naive(self, views, device, ret_state=False, use_ttt3r=False):
        ress = []
        all_state_args = []
        last_smpl_tk = None
        last_smpl_id = None
        max_smpl_id = -1
        reset_mask = False
        for i, _view in enumerate(views):
            view = to_gpu(_view, device)
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            # ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None

            shape = shapes
            feat_i = img_out[-1]
            pos_i = img_pos
            
            # MHMR vit
            imgs_mhmr = view["img_mhmr"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            imgs_mhmr = imgs_mhmr.view(
                -1, *imgs_mhmr.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            selected_imgs_mhmr = imgs_mhmr[img_masks_flat]
            if selected_imgs_mhmr.size(0) > 0:
                mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
                selected_imgs_mhmr = (selected_imgs_mhmr * 0.5 + 0.5 - mean) / std
                mhmr_img_out = [self.backbone(selected_imgs_mhmr)] # image[bs, 3, h, w] -> image feature [bs, h_patches*w_patches, D]
            feat_mhmr_i = mhmr_img_out[-1]


            # MHMR smpl tokenizer
            n_patch_mhmr = self.bb_token_res
            scores = self.downstream_head.detect_mhmr(feat_mhmr_i) #(num_view * bs, h_patches*w_patches, 1)
            scores = rearrange(scores, "b (nh nw) c -> b c nh nw", nh=n_patch_mhmr, nw=n_patch_mhmr) # [num_view * bs, h_nb_patches * w_nb_patches, 1] -> [num_view * bs, 1, h, w]
            feat_mhmr_i = rearrange(feat_mhmr_i, "b (nh nw) c -> b nh nw c", nh=n_patch_mhmr, nw=n_patch_mhmr) # head token extraction: (num_view * bs, h, w, 1024)

            scores = nms(scores, kernel=3) # (num_view * bs, 1, h, w)
            scores = scores.permute((0, 2, 3, 1)) # (num_view * bs, h, w, 1)
            idx = apply_threshold(0.3, scores)
            img_id, h_id, w_id = idx[0], idx[1], idx[2]

            # Head token and offset
            feat_central_mhmr = feat_mhmr_i[img_id, h_id, w_id] # (nvh, 1024)
            offset = self.downstream_head.mlp_offset(feat_central_mhmr)# [nhv,2]
            # Distance for estimating the 3D location in 3D space
            loc = torch.stack([w_id, h_id]).permute(1,0) # x,y
            loc = (loc + 0.5 + offset) * self.bb_patch_size # Moving to higher res the location of the pelvis

            # Concat with camera embedding
            # K = get_camera_parameters(self.mhmr_img_res, device=device) # use pseudo K
            K = view["K_mhmr"] # use GT K
            feat_K = self.embedd_camera(K, [n_patch_mhmr, n_patch_mhmr]) # Embed viewing directions. [num_view * bs,h,w,99]
            feat_K_central = feat_K[img_id, h_id, w_id] # (nvh, 99)
            feat_central_mhmr = torch.cat([feat_central_mhmr, feat_K_central], 1) # feature + camera embedding for heads only to query tokens [nhv, 1123]
            feat_all = torch.cat([feat_mhmr_i, feat_K], -1).permute(0,3,1,2) # feature + camera embedding for full image for the cross-attention only. [bs,1123,nh,nw]

            # Get learned embeddings for queries, at positions with detected people.
            queries_xy = self.cross_queries_x[h_id] + self.cross_queries_y[w_id]
            # Add the embedding to the central features.
            feat_central_mhmr = feat_central_mhmr + queries_xy # [nhv, 1123]
            # Inject leared embeddings for key/values at detected locations. 
            values_xy = self.cross_values_x[h_id] + self.cross_values_y[w_id]
            feat_all[img_id, :, h_id, w_id] += values_xy  # [bs, 1123, nh, nw]
            feat_all = rearrange(feat_all, "b c h w -> b (h w) c") # (num_view * bs, nh*nw, 1024)

            # Get initial smpl token from MHMR
            expand = lambda x: x.expand(*feat_central_mhmr.shape[:-1] , -1)
            pred_body_pose, pred_betas, pred_cam, pred_expression = [expand(x) for x in
                    [self.downstream_head.init_body_pose, 
                    self.downstream_head.init_betas, 
                    self.downstream_head.init_cam, 
                    self.downstream_head.init_expression,
                    ]]
            feat_central_mhmr = torch.cat([
                feat_central_mhmr, pred_body_pose, pred_betas, pred_cam, 
                ], dim=-1)  # training: [bs, 10, 1454]; inference: [nhv, 1454]

            smpl_tk_mhmr = self.transformer(
                feat_central_mhmr.unsqueeze(0), 
                context=feat_all, 
                mask=None) # inference:[1, nhv, 1024]
            
            smpl_feat_i = smpl_tk_mhmr # (1,nvh, 768)

            n_humans_i = smpl_feat_i.shape[1]
            if i == 0 :
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0 or reset_mask:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, cross_attn_states = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                None,
                None,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                use_ttt3r=use_ttt3r,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            if n_humans_i > 0:
                smpl_token = smpl_feat_i
            else:
                smpl_token = None
            res = self._downstream_head(
                head_input,
                shape,
                pos=pos_i,
                n_humans=n_humans_i,
                smpl_token=smpl_token,
                image=view["img"],
            )

            # tracking
            num_miss_match0 = 0
            if last_smpl_tk is not None and smpl_token is not None:
                cost_mat = -torch.cdist(last_smpl_tk, smpl_token, p=2)
                cost_mat = log_optimal_transport(
                    cost_mat, alpha=torch.tensor(-10.0, device=device), iters=20)
                matches = cost_mat[:, :-1, :-1]
                max0, max1 = matches.max(2), matches.max(1)
                indices0, indices1 = max0.indices, max1.indices
                mutual0 = torch.arange(
                    indices0.shape[1], device=device
                    )[None] == indices1.gather(1, indices0)
                mutual1 = torch.arange(
                    indices1.shape[1], device=device
                    )[None] == indices0.gather(1, indices1)
                zero = matches.new_tensor(0)
                mscores0 = torch.where(mutual0, max0.values.exp(), zero)
                mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)

                match_threshold = 0.2
                valid0 = mutual0 & (mscores0 > match_threshold) # 1,n
                valid1 = mutual1 & valid0.gather(1, indices1) # 1,m
                # get the final matching indices, invalid matches set to -1
                indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1)) # [1, n] current frame matches for last frame
                indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1)) # [1, m] last frame matches for current frame

                smpl_id = indices1.new_full(indices1.shape, -1) # 1,m
                valid_match1 = indices1[valid1]
                if valid_match1.numel() > 0:
                    smpl_id[valid1] = last_smpl_id.gather(1, valid_match1[None]).flatten()

                num_miss_match0 = int((~valid0).sum())
                num_new_persons = len(smpl_id[~valid1])
                if num_new_persons > 0:
                    new_ids = torch.arange(
                        max_smpl_id + 1,
                        max_smpl_id + 1 + num_new_persons,
                        device=device
                    )
                    smpl_id[~valid1] = new_ids
                    max_smpl_id += num_new_persons
            else:
                # first frame with humans
                if smpl_token is not None:
                    smpl_id = torch.arange(n_humans_i, device=device)[None]  # (1, nvh)
                    max_smpl_id = n_humans_i - 1
                else:
                    smpl_id = None

            if smpl_token is not None:
                if num_miss_match0 > 0:
                    miss_match_id0 = last_smpl_id[~valid0][None]
                    miss_match_tk0 = last_smpl_tk[~valid0][None]
                    last_smpl_id = torch.cat([smpl_id, miss_match_id0], dim=1)
                    last_smpl_tk = torch.cat([smpl_token, miss_match_tk0], dim=1)
                else:
                    last_smpl_tk = smpl_token.clone()
                    last_smpl_id = smpl_id.clone()

            if smpl_id is not None:
                res['smpl_id'] = smpl_id

            res_cpu = to_cpu({**res, 'smpl_scores': scores, 'smpl_loc': loc[None]})
            ress.append(res_cpu)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()


            if use_ttt3r and i != 0 and not reset_mask:
                cross_attn_states = rearrange(torch.cat(cross_attn_states, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)').mean(dim=(-1, -2))
                update_mask_state = update_mask * torch.sigmoid(cross_attn_states)[..., None]
            else:
                update_mask_state = update_mask

            state_feat = new_state_feat * update_mask_state + state_feat * (
                1 - update_mask_state
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def _init_recurrent_human_slot_state(
        self,
        device,
        feat_dtype,
        pos_dtype,
        max_humans=10,
    ):
        # 推論時に「人ごとの固定スロット」を持つための初期状態を作る。
        # recurrent_lighter では毎フレーム detector の検出順になるので、
        # そのままだと「前フレームの人0」と「今フレームの先頭に来た人」が
        # 同一人物とは限らず、memory の対応が崩れる。
        #
        # そこで推論時にも 10 人ぶんの固定スロットを作り、
        # 毎フレームの検出結果をそのスロットへ再配置してから使う。

        return {
            # CUT3R + MHMR を fuse した human token。
            # shape: [1, max_humans, dec_embed_dim]
            "smpl_feat": torch.zeros(
                1, max_humans, self.dec_embed_dim, device=device, dtype=feat_dtype
            ),
            # 各 human token の 2D position。
            # shape: [1, max_humans, 2]
            "smpl_pos": torch.zeros(
                1, max_humans, 2, device=device, dtype=pos_dtype
            ),
            # MHMR 側の human token。
            # 後で smpl_token_cat = [decoder_token, mhmr_token] を作るときに使う。
            # shape: [1, max_humans, backbone_dim]
            "mhmr_feat": torch.zeros(
                1, max_humans, self.backbone_dim, device=device, dtype=feat_dtype
            ),
            # 人の 2D 位置（高解像度側の loc）。
            # shape: [1, max_humans, 2]
            "smpl_loc": torch.zeros(
                1, max_humans, 2, device=device, dtype=feat_dtype
            ),
            # 「このスロットは過去に一度でも使われたか」
            # 一度使われたスロットは、人が一時的に消えても残す。
            "slot_used": torch.zeros(
                1, max_humans, dtype=torch.bool, device=device
            ),
            # 各スロットに割り当てた人物ID。
            # shape: [1, max_humans]
            "slot_ids": torch.full(
                (1, max_humans), -1, dtype=torch.long, device=device
            ),
            # 次に新規人物へ振るID。
            "next_id": 0,
        }

    def _match_recurrent_humans_to_slots(
        self,
        slot_state,
        curr_smpl_feat,
        curr_smpl_pos,
        curr_mhmr_feat,
        curr_smpl_loc,
        loc_match_threshold=0.25,
        frame_index=None,
        debug=True,
    ):
        # 今回の recurrent_lighter 経路は実質 batch_size == 1 前提で動いている。
        # まずは安全のため、その前提を明示しておく。
        if curr_smpl_feat.shape[0] != 1:
            raise NotImplementedError(
                "forward_recurrent_lighter currently assumes batch_size == 1"
            )

        max_humans = slot_state["smpl_feat"].shape[1]

        # 累計 10 人までという前提なので、今フレームの検出数も 10 に切る。
        # 11人目以降はこの時点で捨てる。
        curr_count = min(curr_smpl_feat.shape[1], max_humans)

        curr_smpl_feat = curr_smpl_feat[:, :curr_count]
        curr_smpl_pos = curr_smpl_pos[:, :curr_count]
        curr_mhmr_feat = curr_mhmr_feat[:, :curr_count]
        curr_smpl_loc = curr_smpl_loc[:, :curr_count]

        # まずは「前の slot_state をそのままコピー」しておく。
        # これが重要で、今フレームで見えなかった人はこのまま残る。
        aligned_feat = slot_state["smpl_feat"].clone()
        aligned_pos = slot_state["smpl_pos"].clone()
        aligned_mhmr = slot_state["mhmr_feat"].clone()
        aligned_loc = slot_state["smpl_loc"].clone()

        # これは「今フレームで本当に見えている slot」だけ True にする mask。
        # slot_used と違って、一時的に消えた人は False になる。
        frame_valid = torch.zeros(
            (1, max_humans), dtype=torch.bool, device=curr_smpl_feat.device
        )

        # det_to_slot[d] = 今フレームの d 番目検出が、どの slot に入るか
        # 未割当ては -1。
        det_to_slot = torch.full(
            (curr_count,), -1, dtype=torch.long, device=curr_smpl_feat.device
        )

        # 既に過去に使われた slot だけ取り出す。
        # 新規人物はここには含まれない。
        used_slots = torch.where(slot_state["slot_used"][0])[0]
        used_slots_before = used_slots.clone()

        # 既存 slot があり、かつ今フレームでも人が検出されている場合だけ matching する。
        if used_slots.numel() > 0 and curr_count > 0:
            prev_loc = slot_state["smpl_loc"][:, used_slots] / float(self.mhmr_img_res)
            curr_loc = curr_smpl_loc / float(self.mhmr_img_res)
            loc_dist = torch.cdist(prev_loc, curr_loc, p=2)[0]
            candidates = [
                (float(loc_dist[p, d].item()), int(p), int(d))
                for p in range(loc_dist.shape[0])
                for d in range(loc_dist.shape[1])
            ]
            used_prev = set()
            used_det = set()
            for dist, prev_pos, det_idx in sorted(candidates):
                if dist > loc_match_threshold:
                    break
                if prev_pos in used_prev or det_idx in used_det:
                    continue
                det_to_slot[det_idx] = used_slots[prev_pos]
                used_prev.add(prev_pos)
                used_det.add(det_idx)

        # まだ slot に入っていない検出は、新規人物候補。
        new_det = torch.where(det_to_slot < 0)[0]

        # 一度も使っていない空き slot を探す。
        free_slots = torch.where(~slot_state["slot_used"][0])[0]

        # 新規人物を空き slot に後ろから詰める。
        if new_det.numel() > 0 and free_slots.numel() > 0:
            assign_count = min(int(new_det.numel()), int(free_slots.numel()))
            assigned_det = new_det[:assign_count]
            assigned_slots = free_slots[:assign_count]

            det_to_slot[assigned_det] = assigned_slots
            slot_state["slot_used"][:, assigned_slots] = True

            # 新規人物には新しいIDを振る。
            start_id = slot_state["next_id"]
            slot_state["slot_ids"][:, assigned_slots] = torch.arange(
                start_id,
                start_id + assign_count,
                device=curr_smpl_feat.device,
                dtype=torch.long,
            ).unsqueeze(0)
            slot_state["next_id"] += assign_count

        # slot が決まった検出だけ、今フレームの token / pos / loc で上書きする。
        assigned_det = torch.where(det_to_slot >= 0)[0]
        if assigned_det.numel() > 0:
            assigned_slots = det_to_slot[assigned_det]
            aligned_feat[:, assigned_slots] = curr_smpl_feat[:, assigned_det]
            aligned_pos[:, assigned_slots] = curr_smpl_pos[:, assigned_det]
            aligned_mhmr[:, assigned_slots] = curr_mhmr_feat[:, assigned_det]
            aligned_loc[:, assigned_slots] = curr_smpl_loc[:, assigned_det]
            frame_valid[:, assigned_slots] = True

        # 今フレームで見えなかった人の slot は上書きしない。
        # つまり「消えたけど slot と memory は残る」という挙動になる。
        slot_state["smpl_feat"] = aligned_feat
        slot_state["smpl_pos"] = aligned_pos
        slot_state["mhmr_feat"] = aligned_mhmr
        slot_state["smpl_loc"] = aligned_loc

        # next_id は「累計何人出たか」を表しているので、
        # 今までに使った slot 数は next_id 個でよい。
        num_slots = int(slot_state["next_id"])

        if debug:
            debug_msgs = []
            current_used_slots = torch.where(slot_state["slot_used"][0])[0]

            for slot in current_used_slots.tolist():
                pid = int(slot_state["slot_ids"][0, slot].item())

                if bool(frame_valid[0, slot].item()):
                    det_idx = torch.where(det_to_slot == slot)[0]
                    det_txt = int(det_idx[0].item()) if det_idx.numel() > 0 else -1

                    was_existing = bool((used_slots_before == slot).any().item())
                    status = "match" if was_existing else "new"

                    debug_msgs.append(
                        f"det{det_txt} -> slot{slot}(id={pid}, {status})"
                    )
                else:
                    debug_msgs.append(
                        f"slot{slot}(id={pid}) -> hidden"
                    )

            prefix = f"[recur-match frame={frame_index}]"
            print(prefix, " | ".join(debug_msgs))


        return (
            slot_state,
            aligned_feat[:, :num_slots],
            aligned_pos[:, :num_slots],
            aligned_mhmr[:, :num_slots],
            aligned_loc[:, :num_slots],
            frame_valid[:, :num_slots],
            slot_state["slot_ids"][:, :num_slots],
        )

    def _compact_recurrent_visible_humans(self, res, frame_valid_slots):
        # 内部では 10 枠の slot を持ち続けるが、
        # 外に返す出力は「今フレームで見えている人だけ」に戻す。
        #
        # これをしないと downstream の demo / render / save 側が
        # 「消えた人まで毎フレーム出てくる」と解釈して壊れやすい。
        if frame_valid_slots.shape[0] != 1:
            raise NotImplementedError(
                "forward_recurrent_lighter currently assumes batch_size == 1"
            )

        visible_idx = torch.where(frame_valid_slots[0])[0]
        for key in [
            "smpl_shape",
            "smpl_rotmat",
            "smpl_transl",
            "smpl_expression",
            "smpl_id",
            "smpl_loc",
        ]:
            if key in res:
                res[key] = res[key][:, visible_idx]
        return res, visible_idx

if __name__ == "__main__":
    print(ARCroco3DStereo.mro())
    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        msk_head=False,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )
    ARCroco3DStereo(cfg)
