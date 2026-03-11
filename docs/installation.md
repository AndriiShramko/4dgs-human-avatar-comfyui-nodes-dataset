# Installation

## ComfyUI nodes

1. **Install ComfyUI** (if not already):
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. **Install this custom nodes pack** as `4dgs_nodes`:
   ```bash
   cd custom_nodes
   git clone https://github.com/AndriiShramko/4dgs-human-avatar-comfyui-nodes-dataset.git 4dgs_nodes
   cd 4dgs_nodes
   pip install -r requirements.txt
   ```

3. **Add 4DGS model folder** (ComfyUI may use `models/4dgs/` or a path you configure). Place your checkpoint files there. See [models/README.md](../models/README.md).

4. **Restart ComfyUI** and look for the **4DGS** category and nodes: **Load 4DGS Model**, **Generate 4DGS Avatar**.

## Dataset scripts only

To only use the dataset download or request scripts (without ComfyUI):

```bash
git clone https://github.com/AndriiShramko/4dgs-human-avatar-comfyui-nodes-dataset.git
cd 4dgs-human-avatar-comfyui-nodes-dataset
pip install -r requirements.txt
python dataset/scripts/download_free.py --output_dir ./data/4dgs_free
```

Set `--repo_id` once the free Hugging Face dataset is published.
