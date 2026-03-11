# Model weights

Place 4DGS human avatar model checkpoints here (or in the path ComfyUI uses for 4DGS models).

## ComfyUI setup

When this repo is installed as a ComfyUI custom node pack (`4dgs_nodes`), the **Load 4DGS Model** node looks for models in:

- ComfyUI’s `models/4dgs/` folder, or  
- A custom path you have registered for the `4dgs` model type in ComfyUI/folder_paths.

So you can:

1. **Option A:** Put checkpoint files in `ComfyUI/models/4dgs/` (create the folder if needed).  
2. **Option B:** Configure ComfyUI to add another folder for 4DGS models and place files there.

## Checkpoint format

- Exact format (e.g. `.ckpt`, `.safetensors`, or framework-specific) will be documented when the first official model is released.
- Use the **model_name** input of the Load 4DGS Model node as the filename (without path), e.g. `4dgs_avatar` for `4dgs_avatar.ckpt`.

## Download

- Official weights will be linked here when available (Hugging Face or other host).
- Until then, train your own model or use weights from a compatible 4DGS implementation and place them in the 4DGS models folder.
