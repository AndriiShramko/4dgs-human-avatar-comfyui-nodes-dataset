# Development & Research Documentation

This folder contains **research and implementation documentation** for the 4DGS avatar generator project: architecture, loss functions, sparse-view generation, kinematics, data preprocessing, hardware economics, compression, and legal strategy.

## Current version: v6

**Start here:** [v6/README.md](v6/README.md) — index and quick start for the implementation team.

| Section | Document | Description |
|--------|----------|-------------|
| Main plan | [6.0-overview.md](v6/6.0-overview.md) | MVP architecture, budget, timeline |
| Loss functions | [6.1-loss-functions.md](v6/6.1-loss-functions.md) | Rendering loss, pixel/voxel alignment |
| Sparse-view | [6.2-sparse-view.md](v6/6.2-sparse-view.md) | LRM + 1-step refinement, DiffusionGS |
| Kinematics | [6.3-kinematics.md](v6/6.3-kinematics.md) | DCT + GNN motion, cloth dynamics |
| Data preprocessing | [6.4-data-preprocessing.md](v6/6.4-data-preprocessing.md) | Squisher, FIM, pipeline |
| Hardware economics | [6.5-hardware-economics.md](v6/6.5-hardware-economics.md) | H100/FP8, cost analysis |
| Compression & deploy | [6.6-compression-deploy.md](v6/6.6-compression-deploy.md) | MEGA, P-4DGS, ComfyUI |
| Legal strategy | [6.7-legal-strategy.md](v6/6.7-legal-strategy.md) | Anny/MHR, gsplat, Clean-Room |

## Disclaimer

This documentation was produced with AI assistance and is shared for reference. It may contain inaccuracies or be outdated. Use it as a starting point and validate against primary sources and current code.
