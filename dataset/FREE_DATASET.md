# Free Dataset (Non-Commercial & Research)

A subset of human scan data is available **free** for **non-commercial** use and research under [CC BY-NC-4.0](../DATASET_LICENSE_FREE.md).

## Where it is hosted

The free dataset is (or will be) published on **Hugging Face Datasets**. Links will be added here once the first release is available.

- **Dataset page (TBD):** `https://huggingface.co/datasets/...`
- **Alternative:** Direct download script using the Hugging Face API: [scripts/download_free.py](scripts/download_free.py)

## Conditions

- **License:** [CC BY-NC-4.0](../DATASET_LICENSE_FREE.md) — attribution required, **no commercial use**.
- **Use cases:** Academic research, personal projects, open-source non-commercial tools.
- **Commercial use:** Not permitted under this license. For commercial use or larger data, see [ENTERPRISE_DATASET.md](ENTERPRISE_DATASET.md).

## How to download

1. Install dependencies (optional, for script):
   ```bash
   pip install huggingface_hub
   ```

2. Run the download script (once the HF dataset ID is set):
   ```bash
   python dataset/scripts/download_free.py --output_dir ./data/4dgs_free
   ```

3. Or use the Hugging Face CLI / web UI when the dataset is published.

## Citation and attribution

When using the free dataset, please cite the dataset and this repository and comply with the attribution requirements of CC BY-NC-4.0.
