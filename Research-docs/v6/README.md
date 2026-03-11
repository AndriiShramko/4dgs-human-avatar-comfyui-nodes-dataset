# 4DGS Avatar Generator — Research Documentation v6

> **Critical Audit & Corrected Architecture for MVP Development**

This documentation represents the **6th iteration** of the research and architectural plan for a feed-forward 4D Gaussian Splatting avatar generator. Version 6 is the result of a comprehensive critical audit of all v5 documents, identifying hallucinations, overstatements, and gaps against the actual SOTA 2025-2026 landscape.

## What Changed from v5

- **7 significant overstatements** corrected (e.g., "Chamfer Distance is absolutely impossible" → near-linear approximations exist)
- **3 critical architectural gaps** filled (1-step diffusion refinement, GNN cloth dynamics, hardware-agnostic training)
- **2 new commercially viable alternatives** integrated (Anny and MHR body models replace legally blocked SMPL)
- **$40,000+ in cost savings** identified (H100+FP8 vs mandatory Blackwell B200)

## Document Structure

| Document | Section | Description |
|----------|---------|-------------|
| [6.0-overview.md](6.0-overview.md) | **Main Plan** | Complete MVP architecture, budget, timeline, risk register |
| [6.1-loss-functions.md](6.1-loss-functions.md) | Loss Functions | Rendering loss imperative, pixel vs voxel alignment, approximate matching analysis |
| [6.2-sparse-view.md](6.2-sparse-view.md) | Sparse-View Generation | Hybrid LRM + 1-step refinement, DiffusionGS, GIFSplat analysis |
| [6.3-kinematics.md](6.3-kinematics.md) | Kinematics | Tiered motion (DCT + GNN), VF-NODE analysis, cloth dynamics |
| [6.4-data-preprocessing.md](6.4-data-preprocessing.md) | Data Preprocessing | Squisher-based filtering, FIM limitations, multi-stage pipeline |
| [6.5-hardware-economics.md](6.5-hardware-economics.md) | Hardware Economics | H100+FP8 as MVP path, NVFP4 reality check, cost analysis |
| [6.6-compression-deploy.md](6.6-compression-deploy.md) | Compression & Deploy | MEGA + P-4DGS hybrid, ComfyUI integration, VRAM budgeting |
| [6.7-legal-strategy.md](6.7-legal-strategy.md) | Legal Strategy | Anny/MHR as SMPL replacement, gsplat, Clean-Room timeline |

## Quick Start for Implementation Team

1. **Start here**: Read [6.0-overview.md](6.0-overview.md) for the full architectural picture
2. **Your module**: Find your section in the table above for detailed specifications
3. **Legal compliance**: Read [6.7-legal-strategy.md](6.7-legal-strategy.md) before writing any code
4. **Hardware setup**: Follow [6.5-hardware-economics.md](6.5-hardware-economics.md) for cluster configuration

## Key Architectural Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Generation paradigm | Hybrid: Feed-forward LRM + 1-step diffusion refinement | Pure LRM produces blurry occluded regions |
| Prediction alignment | Voxel-aligned (VolSplat-style) | Resolution-independent density, 360° consistency |
| Motion model | Tiered: Static / DCT-skeletal / GNN-cloth | DCT alone fails for loose clothing |
| Training loss | L1 + LPIPS + DISTS + depth priors + opacity entropy | SOTA consensus across all 2025-2026 methods |
| Training hardware | H100/H200 + FP8 | 2-3x cheaper than B200; NVFP4 unvalidated for 4DGS |
| Body model | MHR (Meta, Apache 2.0) | Free, rich parameterization, facial expressions, LOD |
| Rasterizer | gsplat (Apache 2.0) | Legally safe, 4x VRAM reduction vs Inria |
| Compression | MEGA DC-AC + P-4DGS temporal | ~10-20 MB per 10s animation clip |
| Code methodology | Clean-Room Design | Required for VC due diligence |

## Estimated Budget

| Item | Cost |
|------|------|
| Training compute (H100 cluster, 3 months) | $57,000 |
| Development compute (6 months) | $26,000 |
| Legal (FTO search + IP attorney) | $26,000 |
| Storage | $12,000 |
| **Total (excluding salaries)** | **~$121,000** |

## Estimated Timeline

| Phase | Duration |
|-------|----------|
| Legal setup + Clean-Room infrastructure | 2-3 weeks |
| Specification + prototyping | 6-8 weeks |
| Core implementation | 10-14 weeks |
| Integration + training | 8-12 weeks |
| Refinement + deployment | 6-8 weeks |
| Optimization + launch | 4-6 weeks |
| **Total** | **~9-13 months** |

---

*Previous versions: [v5](../v5/5.0.md)*
