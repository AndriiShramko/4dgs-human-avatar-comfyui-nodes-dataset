"""
ComfyUI node: load 4DGS human avatar model.
"""

import os
# ComfyUI imports (available when running inside ComfyUI)
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    folder_paths = None


class Load4DGSModel:
    """Load a 4D Gaussian Splatting human avatar model from disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "4dgs_avatar"}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("4DGS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "4DGS"

    def load_model(self, model_name: str):
        if not COMFYUI_AVAILABLE or folder_paths is None:
            return (None,)
        models_dir = folder_paths.get_folder_paths("4dgs") or []
        if not models_dir:
            models_dir = [os.path.join(folder_paths.models_dir, "4dgs")]
        path = None
        for d in models_dir:
            p = os.path.join(d, model_name)
            if os.path.exists(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(f"4DGS model not found: {model_name}. Place weights in ComfyUI models/4dgs/")
        # TODO: load actual 4DGS checkpoint and return wrapper object
        return ({"path": path, "name": model_name},)


NODE_CLASS_MAPPINGS = {
    "Load4DGSModel": Load4DGSModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Load4DGSModel": "Load 4DGS Model",
}
