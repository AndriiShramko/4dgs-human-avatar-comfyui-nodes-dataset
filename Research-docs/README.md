# Development & Research Documentation

This folder contains **research and implementation documentation** for the Shramko-Human-4D project (faithful 4DGS of real human performance): architecture, loss functions, sparse-view generation, kinematics, data preprocessing, hardware economics, compression, and legal strategy.

## Current version: v7

**Start here:** [v7/README.md](v7/README.md) — the audited, three-variant implementation plan. Every claim in v7 carries a verification tag; verdicts on v5/v6 claims live in [v7/CLAIMS_REGISTRY.md](v7/CLAIMS_REGISTRY.md), the audit narrative in [v7/AUDIT_V6.md](v7/AUDIT_V6.md).

v6 remains authoritative for the implementation detail it pioneered (notably the canonical execution prompt 6.21 §21.9 and agent-team mechanics 6.14); its index below. v5 is archived provenance.

## v6 index (audited by v7)

**v6 entry point:** [v6/README.md](v6/README.md) — full 25-section index.

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

## Intake (raw material awaiting a research pass)

| Date | Document | What it is |
|---|---|---|
| 2026-08-14 | [intake/2026-08-14-gush3r.md](intake/2026-08-14-gush3r.md) | GUSH3R (U. Tokyo, arXiv:2607.05243) — feed-forward monocular human+scene → 3DGS. Code preserved in `third_party/GUSH3R/`; license unclear (empty LICENSE + CC BY-NC components) → research-only. |

## Disclaimer

This documentation was produced with AI assistance and is shared for reference. v7 claims are individually verified against primary sources and tagged (see v7/CLAIMS_REGISTRY.md); earlier versions may contain inaccuracies or be outdated. Use them as a starting point and validate against primary sources and current code.
