# 4DGS Human Avatar — ComfyUI Nodes & Datasets

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Nodes-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Datasets-yellow.svg)](https://huggingface.co/)

ComfyUI nodes for **4D Gaussian Splatting** human avatars and curated **human scan datasets** for training and fine-tuning. Generate 360° avatars in sub-second time with feed-forward models.

---

## Overview

- **ComfyUI nodes** — Load 4DGS models and generate human avatars from images/video inside [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- **Free dataset** — Scanned human data for **non-commercial** use and research (CC BY-NC-4.0). Hosted on [Hugging Face](https://huggingface.co/) (links TBD).
- **Enterprise dataset** — Large-scale scans (thousands of subjects, hundreds of TB) for training your own AI models. Available under a commercial license.

| Tier        | Use case                    | License        | Access        |
|------------|-----------------------------|----------------|---------------|
| **Free**   | Research, non-commercial    | CC BY-NC-4.0   | Hugging Face  |
| **Enterprise** | Commercial training, custom models | Commercial     | On request    |

---

## Quick Start (ComfyUI)

1. **Install ComfyUI** (if not already):
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. **Install this custom nodes pack:**
   ```bash
   cd custom_nodes
   git clone https://github.com/AndriiShramko/4dgs-human-avatar-comfyui-nodes-dataset.git 4dgs_nodes
   cd 4dgs_nodes
   pip install -r requirements.txt
   ```

3. **Download model weights** (see [models/README.md](models/README.md)) and place them in ComfyUI’s `models` folder.

4. Restart ComfyUI and use the **4DGS Load Model** and **4DGS Generate Avatar** nodes in your workflow.

---

## Repository structure

```
├── comfyui_nodes/       # ComfyUI node implementations
├── models/              # Model download instructions
├── dataset/             # Dataset docs and download scripts
│   ├── FREE_DATASET.md
│   ├── ENTERPRISE_DATASET.md
│   └── scripts/
├── docs/                # Installation and usage guides
├── examples/            # Example ComfyUI workflows
├── LICENSE              # Apache 2.0 (code)
├── DATASET_LICENSE_FREE.md
└── DATASET_LICENSE_ENTERPRISE.md
```

---

## Dataset

- **Free (non-commercial):** See [dataset/FREE_DATASET.md](dataset/FREE_DATASET.md) and use [dataset/scripts/download_free.py](dataset/scripts/download_free.py) to fetch from Hugging Face.
- **Enterprise:** See [dataset/ENTERPRISE_DATASET.md](dataset/ENTERPRISE_DATASET.md) for access and licensing.

---

## Documentation

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Data format](dataset/DATA_FORMAT.md)

---

## License

- **Code (ComfyUI nodes):** [Apache 2.0](LICENSE).
- **Free dataset:** [CC BY-NC-4.0](DATASET_LICENSE_FREE.md) — non-commercial use and attribution.
- **Enterprise dataset:** See [DATASET_LICENSE_ENTERPRISE.md](DATASET_LICENSE_ENTERPRISE.md).
