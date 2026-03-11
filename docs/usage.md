# Usage

## ComfyUI workflow

1. **Load 4DGS Model** — Choose the model name (e.g. the checkpoint filename without extension). The node looks for it in ComfyUI’s 4DGS models folder.
2. **Generate 4DGS Avatar** — Connect the loaded model and an **IMAGE** input (e.g. from a Load Image node). Optionally set number of output frames. The node outputs a scene and a preview.
3. Use the **4DGS_SCENE** output in downstream nodes (e.g. render or export) when those nodes are implemented.

## Dataset

- **Free (non-commercial):** See [dataset/FREE_DATASET.md](../dataset/FREE_DATASET.md) and run `dataset/scripts/download_free.py` with the Hugging Face `--repo_id` when available.
- **Enterprise:** See [dataset/ENTERPRISE_DATASET.md](../dataset/ENTERPRISE_DATASET.md) and use `dataset/scripts/request_enterprise_access.py` to generate a request, then contact the maintainers.

## Example workflow

An example ComfyUI workflow JSON is in [examples/workflows/example_workflow.json](../examples/workflows/example_workflow.json). Import it in ComfyUI via the menu (Load / Import workflow) and adjust paths or model names as needed.
