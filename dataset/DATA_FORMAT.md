# Data Format

Description of the human scan data format used in the free and Enterprise datasets.

## Capture setup (reference)

- **Studio rig:** Multi-camera setup (e.g. 90 cameras), 4K resolution.
- **Output:** Synchronized multi-view video and/or preprocessed frames per camera.
- Exact calibration and layout may vary per release; metadata is provided with each subset.

## Directory layout (typical)

```
<scan_id>/
├── cameras.json          # or calibration file (intrinsics, extrinsics)
├── frames/               # or images/
│   ├── cam_00/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── cam_01/
│   └── ...
├── metadata.json         # optional: frame count, fps, subject info
└── ...
```

- **Per-camera folders** or **per-frame multi-view images** are common.
- Naming and structure may differ between the free (Hugging Face) and Enterprise releases; each release can ship a short `README` or `metadata.json` describing the layout.

## File formats

- **Images:** PNG or JPEG; 4K where available.
- **Calibration:** JSON or standard formats (e.g. COLMAP-style); fields include camera intrinsics and extrinsics.
- **Precomputed 4DGS / point clouds:** If provided, format will be documented in the release notes (e.g. `.ply`, custom binary, or framework-specific checkpoints).

## Coordinate system and conventions

- To be documented per release (e.g. Y-up, unit scale, color space).
- Check `metadata.json` or release README for details.

## Free vs Enterprise

- **Free dataset:** Subset of scans; same format conventions; may include fewer subjects and fewer cameras or resolution variants.
- **Enterprise dataset:** Full libraries; same or extended format; may add extra metadata or preprocessed assets. Exact specs are provided with the signed agreement and delivery package.
