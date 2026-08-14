#!/usr/bin/env python3
"""Run GUSH3R inference and render merged background+human Gaussians."""

import argparse
import glob
import math
import os
import shutil
import tempfile
import time
from copy import deepcopy

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from add_ckpt_path import add_path_to_dust3r
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


CHECKPOINT = "checkpoints/gush3r.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Run GUSH3R inference.")
    parser.add_argument("--seq_path", required=True, help="Input video path or image directory.")
    parser.add_argument("--output_dir", default="outputs", help="Output directory.")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_ttt3r", action="store_true")
    parser.add_argument("--gs_conf_threshold", type=float, default=1.0)
    parser.add_argument("--bg_mask_threshold", type=float, default=0.02)
    parser.add_argument("--bg_mask_dilation", type=int, default=3)
    parser.add_argument("--bg_voxel_size", type=float, default=0.005)
    parser.add_argument("--bg_gaussian_max", type=int, default=2_000_000)
    return parser.parse_args()


def parse_seq_path(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*"))), None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {path}")

    tmpdir = tempfile.mkdtemp()
    img_paths = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = os.path.join(tmpdir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        img_paths.append(frame_path)
        frame_idx += 1
    cap.release()
    return img_paths, tmpdir


def prepare_input(img_paths, size, img_res=None, reset_interval=10_000_000):
    from src.dust3r.utils.geometry import get_camera_parameters
    from src.dust3r.utils.image import load_images, pad_image

    images = load_images(img_paths, size=size)
    K_mhmr = get_camera_parameters(img_res, device="cpu") if img_res is not None else None

    def make_camera_intrinsics(img_tensor):
        h, w = img_tensor.shape[-2:]
        img_size = max(h, w)
        return get_camera_parameters(
            img_size,
            p_x=w / (2 * img_size),
            p_y=h / (2 * img_size),
            device="cpu",
        )

    views = []
    for i, image in enumerate(images):
        view = {
            "img": image["img"],
            "ray_map": torch.full(
                (image["img"].shape[0], 6, image["img"].shape[-2], image["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(image["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "camera_intrinsics": make_camera_intrinsics(image["img"]),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
        }
        if img_res is not None:
            view["img_mhmr"] = pad_image(view["img"], img_res)
            view["K_mhmr"] = K_mhmr
        views.append(view)
        if (i + 1) % reset_interval == 0:
            overlap_view = deepcopy(view)
            overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
            views.append(overlap_view)
    return views


def restore_sampling_cache(ckpt):
    assets = ckpt.get("assets", {})
    payload = assets.get("sampling_cache")
    filename = assets.get("sampling_cache_filename")
    if payload is None or filename is None:
        return
    cache_dir = "src/dust3r/heads/human_gs/smplx_mesh/sampling_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        torch.save(payload, cache_path)


def load_model(device, args):
    import src.dust3r.model as dust3r_model

    add_path_to_dust3r(CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    restore_sampling_cache(ckpt)

    model = eval(ckpt["model_args"], vars(dust3r_model)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.gs_point_conf_threshold = args.gs_conf_threshold
    model.bg_mask_threshold = args.bg_mask_threshold
    model.bg_mask_dilation = args.bg_mask_dilation
    model.bg_voxel_size = args.bg_voxel_size
    model.bg_gaussian_max = args.bg_gaussian_max
    model.eval()
    return model


def build_render_state(outputs):
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera

    preds = outputs["pred"]
    views = outputs["views"]
    pts3d = torch.cat([pred["pts3d_in_self_view"] for pred in preds], dim=0)
    num_frames, height, width, _ = pts3d.shape

    poses = [pose_encoding_to_camera(pred["camera_pose"].clone()).cpu() for pred in preds]
    R = torch.cat([pose[:, :3, :3] for pose in poses], dim=0).numpy()
    t = torch.cat([pose[:, :3, 3] for pose in poses], dim=0).numpy()

    gt_intrinsics = [view.get("camera_intrinsics") for view in views]
    pp = torch.tensor([width // 2, height // 2], device=pts3d.device).float().repeat(num_frames, 1)
    if all(K is not None for K in gt_intrinsics):
        intrinsics = torch.cat(gt_intrinsics, dim=0).to(pts3d.device).float()
    else:
        focal = estimate_focal_knowing_depth(pts3d, pp, focal_mode="weiszfeld")
        intrinsics = torch.eye(3, device=pts3d.device).unsqueeze(0).repeat(num_frames, 1, 1)
        intrinsics[:, 0, 0] = focal.detach()
        intrinsics[:, 1, 1] = focal.detach()
        intrinsics[:, 0, 2] = pp[:, 0]
        intrinsics[:, 1, 2] = pp[:, 1]

    return {
        "gaussians": [pred.get("gaussians") for pred in preds],
        "human_gaussians": [pred.get("human_gaussians") for pred in preds],
        "intrinsics": intrinsics.cpu().numpy(),
        "R": R,
        "t": t,
        "image_hw": (height, width),
    }


def select_background_gaussians(gaussians):
    return next((g for g in reversed(gaussians) if g is not None), None)


def background_tensors(gs):
    xyz = gs.means.view(-1, 3)
    scale = gs.scales.view(-1, 3)
    rot = gs.rotations.view(-1, 4)
    opac = gs.opacities.view(-1, 1)
    shs = gs.harmonics.transpose(-2, -1).contiguous().view(-1, gs.harmonics.shape[-1], 3)
    return xyz, scale, rot, opac, shs


def human_tensors(gs, sh_dim):
    if gs is None or gs.xyz.shape[0] == 0:
        return None
    rgb = gs.shs.view(-1, 3)
    shs = torch.zeros(rgb.shape[0], sh_dim, 3, device=rgb.device, dtype=rgb.dtype)
    shs[:, 0, :] = (rgb - 0.5) / 0.28209479177387814
    return (
        gs.xyz.view(-1, 3),
        gs.scaling.view(-1, 3),
        gs.rotation.view(-1, 4),
        gs.opacity.view(-1, 1),
        shs,
    )


def merged_tensors(bg, human):
    bg_t = background_tensors(bg) if bg is not None else None
    human_t = human_tensors(human, bg_t[-1].shape[1] if bg_t is not None else 1)
    if bg_t is None:
        return human_t
    if human_t is None:
        return bg_t

    bg_xyz, bg_scale, bg_rot, bg_opac, bg_shs = bg_t
    h_xyz, h_scale, h_rot, h_opac, h_shs = human_t
    max_sh = max(bg_shs.shape[1], h_shs.shape[1])
    if bg_shs.shape[1] < max_sh:
        bg_shs = F.pad(bg_shs, (0, 0, 0, max_sh - bg_shs.shape[1]))
    if h_shs.shape[1] < max_sh:
        h_shs = F.pad(h_shs, (0, 0, 0, max_sh - h_shs.shape[1]))
    return (
        torch.cat([bg_xyz, h_xyz], dim=0),
        torch.cat([bg_scale, h_scale], dim=0),
        torch.cat([bg_rot, h_rot], dim=0),
        torch.cat([bg_opac, h_opac], dim=0),
        torch.cat([bg_shs, h_shs], dim=0),
    )


def projection_matrix(znear, zfar, fov_x, fov_y, device):
    tan_y = math.tan(float(fov_y) / 2.0)
    tan_x = math.tan(float(fov_x) / 2.0)
    top = tan_y * znear
    right = tan_x * znear
    P = torch.zeros(4, 4, device=device, dtype=torch.float32)
    P[0, 0] = znear / right
    P[1, 1] = znear / top
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def render_frame(tensors, K, R, t, image_hw, device):
    height, width = image_hw
    xyz, scale, rot, opac, shs = tensors
    K = torch.as_tensor(K, device=device, dtype=torch.float32)
    R = torch.as_tensor(R, device=device, dtype=torch.float32)
    t = torch.as_tensor(t, device=device, dtype=torch.float32)

    c2w = torch.eye(4, device=device)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    w2c = torch.inverse(c2w)
    view = w2c.transpose(0, 1).contiguous()

    fov_x = 2.0 * torch.atan(torch.tensor(float(width), device=device) / (2.0 * K[0, 0]))
    fov_y = 2.0 * torch.atan(torch.tensor(float(height), device=device) / (2.0 * K[1, 1]))
    proj = projection_matrix(0.01, 100.0, fov_x, fov_y, device).transpose(0, 1).contiguous()
    full_proj = (view @ proj).contiguous()
    center = torch.inverse(view)[3, :3].contiguous()

    settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fov_x * 0.5),
        tanfovy=math.tan(fov_y * 0.5),
        bg=torch.tensor([1.0, 1.0, 1.0], device=device),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=full_proj,
        sh_degree=math.isqrt(shs.shape[1]) - 1,
        campos=center,
        prefiltered=False,
        debug=False,
    )
    image, _, _, _ = GaussianRasterizer(settings)(
        means3D=xyz.to(device).float().contiguous(),
        means2D=torch.zeros_like(xyz, device=device, requires_grad=True).float().contiguous(),
        shs=shs.to(device).float().contiguous(),
        colors_precomp=None,
        opacities=opac.to(device).float().contiguous(),
        scales=scale.to(device).float().contiguous(),
        rotations=rot.to(device).float().contiguous(),
        cov3D_precomp=None,
    )
    return np.clip(image.permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)


def render_video(state, output_dir, fps, device):
    frames_dir = os.path.join(output_dir, "merged_render")
    os.makedirs(frames_dir, exist_ok=True)
    bg = select_background_gaussians(state["gaussians"])
    frames = []
    for i, human in enumerate(state["human_gaussians"]):
        tensors = merged_tensors(bg, human)
        if tensors is None:
            height, width = state["image_hw"]
            frame = np.ones((height, width, 3), dtype=np.float32)
        else:
            frame = render_frame(
                tensors,
                state["intrinsics"][i],
                state["R"][i],
                state["t"][i],
                state["image_hw"],
                device,
            )
        frame_u8 = (frame * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:06d}.png"), cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
        frames.append(frame_u8)
    video_path = os.path.join(output_dir, "render.mp4")
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Saved merged Gaussian render: {video_path}")


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    img_paths, tmpdir = parse_seq_path(args.seq_path)
    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[:: args.subsample]
    if not img_paths:
        raise ValueError(f"No input frames found: {args.seq_path}")

    model = load_model(device, args)
    views = prepare_input(img_paths, size=args.size, img_res=getattr(model, "mhmr_img_res", None))

    from src.dust3r.inference import inference_recurrent_lighter

    start = time.time()
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, device, use_ttt3r=args.use_ttt3r)
    print(f"Inference finished in {time.time() - start:.2f}s")

    if tmpdir is not None:
        shutil.rmtree(tmpdir)

    render_video(build_render_state(outputs), args.output_dir, max(1, 30 // args.subsample), device)


if __name__ == "__main__":
    main()
