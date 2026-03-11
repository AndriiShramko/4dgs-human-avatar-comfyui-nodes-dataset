"""
ComfyUI node: generate 4DGS human avatar from image(s) or video.
"""

# ComfyUI imports (available when running inside ComfyUI)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class Generate4DGSAvatar:
    """Generate 4DGS human avatar from input image(s) using a loaded model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("4DGS_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "num_frames": ("INT", {"default": 30, "min": 1, "max": 300}),
            },
        }

    RETURN_TYPES = ("4DGS_SCENE", "IMAGE")
    RETURN_NAMES = ("scene", "preview")
    FUNCTION = "generate"
    CATEGORY = "4DGS"

    def generate(self, model, image, num_frames=30):
        if not TORCH_AVAILABLE or torch is None:
            return (None, image)
        # TODO: run feed-forward 4DGS model on image, output scene + preview frames
        # Placeholder: pass-through image as preview
        return ({"model": model, "frames": num_frames}, image)


NODE_CLASS_MAPPINGS = {
    "Generate4DGSAvatar": Generate4DGSAvatar,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Generate4DGSAvatar": "Generate 4DGS Avatar",
}
