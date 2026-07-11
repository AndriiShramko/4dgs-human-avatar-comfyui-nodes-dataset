# 4DGS Avatar Generator — Research Documentation v6

> **Critical Audit & Corrected Architecture for MVP Development**

This documentation represents the **6th iteration** of the research and architectural plan for a feed-forward 4D Gaussian Splatting avatar generator. Version 6 is the result of a comprehensive critical audit of all v5 documents, identifying hallucinations, overstatements, and gaps against the actual SOTA 2025-2026 landscape.

> **Fact-check audit (2026-06-12):** Sections 6.1-6.7 were verified claim-by-claim against primary sources (papers, repos, license texts) by a 7-agent team. All cited papers turned out to exist; ~30 factual corrections were applied in place (inline annotations marked ⚠️). All key architectural decisions **survived** verification. Most material corrections: B200 2026 pricing makes the H100 cost argument obsolete (FP8 maturity is now the rationale); the "5-15 dB occluded-region gap" figure was unsourced; MEGA is 161→20 params; Anny's SMPL-X-compat mode is non-commercial only; filtering stages compound to ~84%. Section 6.8 added — analysis of Flex4DHuman as a new research source.

> **Competitive landscape + ArtiFixer pass (2026-07-11):** Two new research sources added — §6.22 (4C4D and 4DV.ai/FreeTimeGS, sparse/few-camera 4DGS capture prior art) and §6.23 (NVIDIA ArtiFixer, a post-hoc 3DGS completion model). Both went through primary-source verification; several claims from an earlier informal vault note were retracted as unconfirmable (4C4D's coverage angle and sync mechanism, specific PSNR figures), and one existing correction was applied in place: §6.16's FreeTimeGS entry mischaracterized the AGPL-3.0 license as if it belonged to an official ZJU release — the official repo has no code at all; the license belongs only to an unofficial fork. Headline finding: 4C4D's MIT top license does not clear its vendored non-commercial dependencies (Inria rasterizer, MASt3R) — a live confirmation of exactly the license-trap pattern §6.7 already audits for. ArtiFixer's training/inference code is Apache-2.0, but its only released checkpoint is NVIDIA's non-commercial "research and development only" license — code and weights must be evaluated separately, every time.

## What Changed from v5

- **7 significant overstatements** corrected (e.g., "Chamfer Distance is absolutely impossible" → near-linear approximations exist)
- **3 critical architectural gaps** filled (1-step diffusion refinement, GNN cloth dynamics, hardware-agnostic training)
- **2 new commercially viable alternatives** integrated (Anny and MHR body models replace legally blocked SMPL)
- **Hardware de-risked**: H100+FP8 instead of mandatory Blackwell B200+NVFP4 (chosen for FP8 training maturity; the originally claimed "$40K+ savings" did not survive the 2026 price check — per-run costs are roughly comparable)

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
| [6.8-flex4dhuman-analysis.md](6.8-flex4dhuman-analysis.md) | **New Research Source** | Flex4DHuman (UW + World Labs): pipeline breakdown, license audit, integration strategy, replication plan |
| [6.9-acceleration-and-cost-reduction.md](6.9-acceleration-and-cost-reduction.md) | **Track A: Acceleration** | Verified 2026 cost/speed optimizations: LoRA-first fine-tune, few-step distillation (FastWan/Self-Forcing), 10-30x faster 4DGS fitting, backbone bake-off — MVP budget ~$62-70K |
| [6.10-data-strategy.md](6.10-data-strategy.md) | **Data Strategy** | Commercially clean training data at minimum cost: license-verified synthetic stack (CC0/Apache/MIT), 8-12 camera consumer rig, vendor license audit — ~$8-15K total |
| [6.11-gracia-collaboration-track.md](6.11-gracia-collaboration-track.md) | **Track B: Partner** | Gracia AI collaboration: verified company profile, distillation-from-reconstruction pipeline, IP firewall protecting Track A, deal options, first-call questions |
| [6.12-noncommercial-mvp-track.md](6.12-noncommercial-mvp-track.md) | **Track C: NC MVP** | Fast/cheap non-commercial open demo from released NC weights (MV-Performer / Diffuman4D + gsplat), 3-5 weeks ≈$700-1.6K, hard firewall from commercial tracks |
| [6.13-single-gpu-nc-training-plan.md](6.13-single-gpu-nc-training-plan.md) | **NC Training: Machine-Time** | Train our own NC model on ONE RTX 6000 Pro (96GB): three scenarios (red-team audited). S1 ~1 wk (smoke), S2 ~5-8 wk calendar (recommended), S3 ~2-3.5 mo. ⚠️ DNA-Rendering data gating is a hard day-0 blocker |
| [6.14-goal-mode-agent-team-plan.md](6.14-goal-mode-agent-team-plan.md) | **Goal-Mode Agent Team** | 6-agent autonomous build plan, milestones M0-M6 with numeric acceptance gates, single-GPU discipline, Obsidian reporting, owner-gates at data licensing + publishing |
| [6.15-world-tracing-analysis.md](6.15-world-tracing-analysis.md) | **New Research Source** | World Tracing (World Labs+UIUC, arXiv 2606.13652): occlusion-aware per-pixel geometry stacks. License CC BY-NC-ND 4.0 → NC track only; useful as 4DGS-init / back-side prior; does NOT close moving-camera gap |
| [6.16-zero-training-mvp-pipeline.md](6.16-zero-training-mvp-pipeline.md) | **Zero-Training MVP** | Faithful REAL-VIDEO→4DGS with no training: per-video optimization (Shape-of-Motion/MoSca/Instant4D; ExAvatar/GART for humans) — multi-cam=faithful 360°, mono=faithful front. Photo→avatar (LHM) excluded from product path |
| [6.17-obtainable-data-plan.md](6.17-obtainable-data-plan.md) | **Obtainable Data (no gating)** | Solo-downloadable data replacing DNA-Rendering: CMU Panoptic (open), AIST++ (CC BY), PKU-DyMVHumans (click-through, dense multi-cam) + synthetic. Volume clears the fine-tune floor. ETH-AIT/RenderMe-360 disqualified |
| [6.18-frontier-and-camera-frontend.md](6.18-frontier-and-camera-frontend.md) | **2026 Frontier + Camera** | Newest post-Dec-2025 releases (Forge4D, NoPo4D, TrackingWorld); moving-camera front-end: MegaSaM (NC) / VGGT-1B-Commercial (commercial) |
| [6.19-buildable-mvp-synthesis.md](6.19-buildable-mvp-synthesis.md) | **★ Buildable MVP (synthesis)** | The realizable plan on one RTX 6000 Pro: Layer 0 (zero-training demo this week) → Layer 1 (light fine-tune on obtainable data, §6.13 S2 now UNBLOCKED) → Layer 2 (watch). DNA-Rendering blocker resolved |
| [6.20-goal-mode-2frontal-build-plan.md](6.20-goal-mode-2frontal-build-plan.md) | **★ Goal-Mode Prompt (2-frontal)** | Copy-paste goal-mode master prompt + detailed capture spec for the 2-frontal-cameras→360° variant (front faithful, back hallucinated). Autonomous agent roles, M0-M6, single-GPU on RTX 6000 Pro |
| [6.21-2gopro-reality-adapted-plan.md](6.21-2gopro-reality-adapted-plan.md) | **★ 2× GoPro Reality Plan + Prompt** | §6.20 adapted to the owner's real kit: 2× GoPro HERO 13 (Labs QR/GPS sync, NOT flash), ~5 subjects, no green screen, varied locations. Verified sync facts, rigid-rig calibration, no-greenscreen matting, 360° turntable back-GT trick, open-data back-learning, revised goal-mode prompt |
| [6.22-4c4d-and-4dvai-competitive-landscape.md](6.22-4c4d-and-4dvai-competitive-landscape.md) | **New Research Source** | 4C4D (4×GoPro academic prior art): Neural Decaying Function technique, ⚠️ MIT-top-license/non-commercial-dependency trap (Inria rasterizer + MASt3R). 4DV.ai/FreeTimeGS: corrected hardware/pricing, official-repo-has-no-code correction, still-undisclosed commercial terms |
| [6.23-artifixer-3dgs-completion.md](6.23-artifixer-3dgs-completion.md) | **New Research Source** | NVIDIA ArtiFixer (SIGGRAPH 2026): post-hoc 3DGS completion via opacity-mixing diffusion; code Apache-2.0 but only released checkpoint is NC-licensed; analysis of whether it could complement/replace our Stage-A back-hallucination fine-tune (§6.13/§6.21) — open question, untested |

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
| Training hardware | H100/H200 + FP8 | FP8 training mature; NVFP4 unvalidated for geometric regression (per-run cost vs B200 roughly comparable at 2026 prices) |
| Body model | MHR (Meta, Apache 2.0) | Free, rich parameterization, facial expressions, LOD; Anny (Apache 2.0) backup — native topology only (SMPL-X-compat mode is non-commercial) |
| Rasterizer | gsplat (Apache 2.0) | Legally safe, up to 4x VRAM reduction vs Inria |
| Compression | MEGA DC-AC (161→20) + custom temporal layer (P-4DGS-inspired) | Target ~10-20 MB per 10s clip (extrapolated, not published) |
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
