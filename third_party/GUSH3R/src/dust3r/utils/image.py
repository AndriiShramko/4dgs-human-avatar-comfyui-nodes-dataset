# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

try:
    from pillow_heif import register_heif_opener  # noqa

    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def img_to_arr(img):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
        W1, H1 = img.size
        if size == 224:

            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:

            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )

    assert imgs, "no images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs


def load_images_for_eval(
    folder_or_list, size, square_ok=False, verbose=True, crop=True
):
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            if crop:
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:  # resize
                img = img.resize((2 * half, 2 * half), PIL.Image.LANCZOS)
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            if crop:
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            else:  # resize
                img = img.resize((2 * halfw, 2 * halfh), PIL.Image.LANCZOS)
        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
                ori_shape=np.int32([[H1, W1]])
            )
        )

    assert imgs, "no images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs



def load_masks_for_eval(
    folder_or_list, size, square_ok=False, verbose=True, crop=True
):
    if isinstance(folder_or_list, str):
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            if crop:
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:  # resize
                img = img.resize((2 * half, 2 * half), PIL.Image.LANCZOS)
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            if crop:
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            else:  # resize
                img = img.resize((2 * halfw, 2 * halfh), PIL.Image.LANCZOS)

        img = tvf.ToTensor()(img)
        img = (img[0] > 0.5).float()
        imgs.append(img[None])

    assert imgs, "no images foud at " + root
    return imgs


def load_images_512(folder_or_list, size, square_ok=False, verbose=True):
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
        img = img.resize((512, 384))
        W1, H1 = img.size
        if size == 224:

            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:

            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )

    assert imgs, "no images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs


IMG_NORM_MEAN = [0.5, 0.5, 0.5]
IMG_NORM_STD = [0.5, 0.5, 0.5]


def normalize_rgb(img, imagenet_normalization=True):
    """
    Args:
        - img: np.array - (W,H,3) - np.uint8 - 0/255
    Return:
        - img: np.array - (3,W,H) - np.float - -3/3
    """
    img = img.astype(np.float32) / 255.
    img = np.transpose(img, (2,0,1))
    if imagenet_normalization:
        img = (img - np.asarray(IMG_NORM_MEAN).reshape(3,1,1)) / np.asarray(IMG_NORM_STD).reshape(3,1,1)
    img = img.astype(np.float32)
    return img


def denormalize_rgb(img, imagenet_normalization=True):
    """
    Args:
        - img: np.array - (3,W,H) - np.float - -3/3
    Return:
        - img: np.array - (W,H,3) - np.uint8 - 0/255
    """
    if imagenet_normalization:
        img = (img * np.asarray(IMG_NORM_STD).reshape(3,1,1)) + np.asarray(IMG_NORM_MEAN).reshape(3,1,1)
    img = np.transpose(img, (1,2,0)) * 255.
    img = img.astype(np.uint8)
    return img


import torch.nn.functional as F

def pad_image(img_tensor, target_size, pad_value=-1.0):
    """
    torch version of ImageOps.pad, equivalent to the combination of contain and pad
    
    Args:
        img_tensor: torch tensor, shape [C, H, W] or [B, C, H, W]
        target_size: int, target size (square)
    
    Returns:
        torch tensor, shape [C, target_size, target_size] or [B, C, target_size, target_size]
    """
    
    # process input dimension
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, channels, height, width = img_tensor.shape
    
    # calculate scale (contain function)
    scale = min(target_size / height, target_size / width)
    
    # resize image
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    img_resized = F.interpolate(
        img_tensor, 
        size=(new_height, new_width), 
        mode='bilinear',     # bicubic
        align_corners=False
    )
    
    # calculate padding (pad function)
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    
    # center padding
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # apply padding (left, right, top, bottom)
    img_padded = F.pad(
        img_resized, 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode='constant', 
        value=pad_value
    )
    
    if squeeze_output:
        img_padded = img_padded.squeeze(0)
    
    return img_padded


def unpad_image(img_tensor, target_size):
    """
    torch version of unpad, reverse operation of pad_image
    
    Args:
        img_tensor: torch tensor, shape [C, H, W] or [B, C, H, W], assumed to be square and padded
        target_size: tuple/list [H, W], target height and width
    
    Returns:
        torch tensor, shape [C, H, W] or [B, C, H, W] with target_size dimensions
    """
    
    # process input dimension
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    target_height, target_width = target_size
    max_target = max(target_height, target_width)
    
    # first resize to the larger dimension size (square)
    img_resized = F.interpolate(
        img_tensor, 
        size=(max_target, max_target), 
        mode='nearest',
        # align_corners=False
    )
    
    # then crop to target size (center crop)
    pad_height = max_target - target_height
    pad_width = max_target - target_width
    pad_top = pad_height // 2
    pad_left = pad_width // 2
    
    img_cropped = img_resized[
        :, :,
        pad_top:pad_top + target_height,
        pad_left:pad_left + target_width
    ]
    
    if squeeze_output:
        img_cropped = img_cropped.squeeze(0)
    
    return img_cropped


def unpad_uv(uv, original_size, target_height, target_width):
    """
    transform uv from padded image to unpadded image
    
    Args:
        uv: uv coordinates tensor, shape [batch_size, num_points, 2] or [num_points, 2]
        original_size: original size of the image (int)
        target_height: target height of the image (int)
        target_width: target width of the image (int)
    
    Returns:
        uv_transformed: transformed uv coordinates tensor, shape [batch_size, num_points, 2] or [num_points, 2]
    """
    # calculate the maximum size of the target
    max_target = max(target_height, target_width)
    
    # first, scale the uv from original_size to max_target
    scale_factor = max_target / original_size
    uv_scaled = uv * scale_factor
    
    # then, subtract the padding offset
    pad_left = (max_target - target_width) // 2
    pad_top = (max_target - target_height) // 2
    
    # create the offset tensor, shape [2]
    offset = torch.tensor([pad_left, pad_top], dtype=uv.dtype, device=uv.device)
    
    # broadcast subtraction
    uv_transformed = uv_scaled - offset
    uv_transformed[..., 0] = torch.clamp(uv_transformed[..., 0], 0, target_width - 1)   # u
    uv_transformed[..., 1] = torch.clamp(uv_transformed[..., 1], 0, target_height - 1)  # v
    return uv_transformed


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z