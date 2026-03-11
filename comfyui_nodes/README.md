# 4DGS ComfyUI Nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to load 4D Gaussian Splatting human avatar models and generate avatars from images.

## Nodes

- **Load 4DGS Model** — Load a 4DGS checkpoint from `models/4dgs/` (or configured path). Outputs a `4DGS_MODEL` for use in the generate node.
- **Generate 4DGS Avatar** — Takes an image and a loaded model, runs feed-forward 4DGS inference, and returns a scene plus preview frames.

## Setup

1. Clone the parent repo into ComfyUI `custom_nodes` as `4dgs_nodes` (see root [README](../README.md)).
2. Install dependencies: `pip install -r requirements.txt`
3. Add 4DGS model weights to ComfyUI’s model folder (see [models/README.md](../models/README.md)).
4. Restart ComfyUI.

## Extending

- Add new node classes in `nodes_4dgs_*.py` and register them in `__init__.py` via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
