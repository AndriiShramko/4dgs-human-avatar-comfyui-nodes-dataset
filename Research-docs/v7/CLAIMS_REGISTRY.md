# Claims Registry — v7 verification ledger

**Date of verification pass: 2026-07-12.** Every factual claim inherited from v5/v6 (and every new claim made in v7) carries a status tag. Verdicts from v6 itself were treated as `inherited-unverified` hypotheses and re-proven against primary sources — v6 is an AI-written audit of AI-written text and is not evidence.

## Tag taxonomy <!-- lint-skip -->
- `[verified | URL | as-of date | "verbatim quote"]` — checked against a primary source (LICENSE file, arXiv abs page, official README, conference site). <!-- lint-skip -->
- `[estimate | derivation]` — a number derived from verified baselines; derivation shown. <!-- lint-skip -->
- `[hypothesis | how to test]` — requires an experiment; no claim of truth. <!-- lint-skip -->
- `[not-found-as-of date | queries: ...]` — we searched and did not find it. This is NOT a claim of non-existence. <!-- lint-skip -->
- `[inherited-unverified]` — legacy v5/v6 claim quoted for audit purposes. <!-- lint-skip -->
- `[rumor | circulation]` — number circulating without a public primary source. <!-- lint-skip -->
- `[internal | path]` — fact about this repository itself, checkable by opening the file. <!-- lint-skip -->
- `[registry ID, ...]` — cross-reference: the full evidence lives in this registry's row(s) with those IDs; other v7 docs use this instead of duplicating long tags. <!-- lint-skip -->

Verdict column: TRUE / FALSE / PARTIAL (true with a material caveat) / INTERNAL-CONTRADICTION / RUMOR / NOT-FOUND / UNVERIFIED.

---

## Section LB1 — Load-bearing references: suspected phantom papers

**Outcome: no phantoms.** All eight works exist. The v7 planning audit itself suspected several of these were hallucinations — that suspicion was refuted by fetching primary sources, which is exactly why this registry re-proves everything, in both directions. Material distortions found instead: wrong acronym expansions, stage-time presented as pipeline-time, training-speedup presented as runtime-speedup.

| ID | Claim (short) | Verdict | Evidence |
|---|---|---|---|
| LB1-1 | "GIFSplat (2026)" arXiv 2602.22571 exists; feed-forward 3DGS with iterative refinement | TRUE | [verified \| https://arxiv.org/abs/2602.22571 \| as-of 2026-07-12 \| "GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views"] (CVPR 2026) |
| LB1-2 | "CUDA-APML" arXiv 2512.19743 claims −99.9% peak GPU memory | TRUE | [verified \| https://arxiv.org/abs/2512.19743 \| as-of 2026-07-12 \| "reducing peak GPU memory by 99.9%"] |
| LB1-2n | v6's expansion "approximate point matching loss" for APML | FALSE | [verified \| https://arxiv.org/abs/2509.08104 \| as-of 2026-07-12 \| "APML: Adaptive Probabilistic Matching Loss for Robust 3D Point Cloud Reconstruction"] — v6 misnames it |
| LB1-3 | "CloDS / Cloth Dynamics Splatting" has a primary source | PARTIAL | [verified \| https://arxiv.org/abs/2602.01844 \| as-of 2026-07-12 \| "CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions"] — ICLR 2026; "Cloth Dynamics Splatting" is only the GitHub repo tagline (https://github.com/whynot-zyl/CloDS), not the paper title |
| LB1-4 | "VF-NODE" (ICLR 2025) gives 10–1000x Neural ODE speedup | PARTIAL | [verified \| https://iclr.cc/virtual/2025/poster/28017 \| as-of 2026-07-12 \| "accelerates NODE training by 10 to 1000 times compared to existing NODE-based methods"] — speedup applies to TRAINING, not inference; any v6 runtime budget built on it is unsupported |
| LB1-5 | "Squisher" (ICML 2025): informativeness from Adam v_t as diag-FIM proxy | TRUE | [verified \| https://icml.cc/virtual/2025/poster/44175 \| as-of 2026-07-12 \| "the 'Squisher' (Squared gradient accumulator as an approximation of the Fisher)"] (ICML 2025 Spotlight) |
| LB1-6 | HumanSplat (NeurIPS 2024): 0.3s extraction, >150 FPS rendering | PARTIAL | [verified \| https://arxiv.org/html/2406.12459v1 \| as-of 2026-07-12 \| "latent reconstruction for 3DGS takes only 0.3s ... render novel views at a rate exceeding 150 FPS on a single NVIDIA A100"] — caveat: the same source gives ~9s for the 2D diffusion stage, so full inference is ≈9.3s [estimate \| 9s diffusion + 0.3s reconstruction per the verified quote]; NeurIPS listing: https://papers.nips.cc/paper_files/paper/2024/hash/87affd2029375d1be123ccdab5334c55-Abstract-Conference.html |
| LB1-7 | DiffusionGS: ~6s on A100, +2.20 dB PSNR | TRUE | [verified \| https://arxiv.org/abs/2411.14384 \| as-of 2026-07-12 \| "improvements of 2.20 dB/23.25 ... over 5× faster speed (~6s on an A100 GPU)"] (ICCV 2025) |
| LB1-8a | Difix3D+ exists: single-step diffusion refinement of rendered novel views | TRUE | [verified \| https://arxiv.org/abs/2503.01774 \| as-of 2026-07-12 \| "Difix, a single-step image diffusion model trained to enhance and remove artifacts in rendered novel views"] (CVPR 2025 Oral) |
| LB1-8b | Difix3D+ code/model is permissively licensed | FALSE | [verified \| https://github.com/nv-tlabs/Difix3D \| as-of 2026-07-12 \| "The use of the model and code is governed by the NVIDIA License."] — NVIDIA License, not Apache/MIT; commercial use requires reading LICENSE.txt restrictions |

---

## Section LB2 — Load-bearing references: licenses and the ComfyUI niche

| ID | Claim (short) | Verdict | Evidence |
|---|---|---|---|
| LB2-1 | SMPL-X model is research-only; commercial route via Meshcapade | TRUE | [verified \| https://smpl-x.is.tue.mpg.de/modellicense.html \| as-of 2026-07-12 \| "Any other use, in particular any use for commercial, pornographic, military, or surveillance, purposes is prohibited." + "The software/data is also available for commercial licensing through Meshcapade.com"] |
| LB2-2a | Apple ml-sharp = single image → static 3DGS (no video/4D) | TRUE | [verified \| https://github.com/apple/ml-sharp \| as-of 2026-07-12 \| "We present SHARP, an approach to photorealistic view synthesis from a single image."] — README contains no 4D/temporal/video capability |
| LB2-2b | ml-sharp CODE license bans commercial use | FALSE | [verified \| https://raw.githubusercontent.com/apple/ml-sharp/main/LICENSE \| as-of 2026-07-12 \| "Apple grants you a personal, non-exclusive license... to use, reproduce, modify and redistribute the Apple Software"] — custom permissive-style license; no commercial ban in the CODE license |
| LB2-2c | ml-sharp model WEIGHTS are research-only | TRUE | [verified \| https://raw.githubusercontent.com/apple/ml-sharp/main/LICENSE_MODEL \| as-of 2026-07-12 \| "exclusively for Research Purposes" excluding "any commercial exploitation, product development or use in any commercial product"] — HF card apple/Sharp: license tag `apple-amlr` |
| LB2-2d | Apple states "~100 GB per minute" for SHARP anywhere | NOT-FOUND | [not-found-as-of 2026-07-12 \| queries: 'Apple SHARP gaussian splatting "GB per minute" OR "100 GB"'; '"SHARP" Apple gaussians "per minute" OR "gigabytes" storage'; README github.com/apple/ml-sharp; HF card apple/Sharp] — the figure attributed to SHARP in this repo's README/landing has no traceable source |
| LB2-4 | QUEEN (NVlabs/queen) allows commercial use | FALSE | [verified \| https://raw.githubusercontent.com/NVlabs/queen/main/LICENSE.md \| as-of 2026-07-12 \| "The Work and any derivative works thereof only may be used or intended for use non-commercially."] — plus inherited Inria license "for research and/or evaluation purposes only" |
| LB2-5 | DualGS code exists; license status | PARTIAL | [verified \| https://github.com/HiFi-Human/DualGS \| as-of 2026-07-12 \| "code derived from the original Gaussian Splatting implementation are licensed under the Gaussian Splatting Research License... new components... under MIT License."] — training code released; MIT + Inria hybrid: commercial use requires a cleaned rasterizer |
| LB2-6 | VolHuMe (arXiv 2606.23062) dataset has a stated license / commercial-training terms | NOT-FOUND | [not-found-as-of 2026-07-12 \| queries: arxiv.org/abs/2606.23062; arxiv.org/html/2606.23062v1; 'VolHuMe dataset license terms download'] — CC BY 4.0 on the arXiv page covers the PAPER only; consent quote: "for research purposes and publication" → commercial training questionable; no download page found |
| LB2-7 | No 4DGS (temporal) nodes exist in the ComfyUI ecosystem | PARTIAL | [verified \| https://api.comfy.org/nodes/search?search=gaussian \| as-of 2026-07-12 \| registry 4D/temporal results: none] — queries: registry search gaussian/splat/4d; GitHub "comfyui 4dgs" / "comfyui temporal gaussian" / "comfyui dynamic gaussian splatting". Essentially true: sole exception is TiMarinov/ComfyUi-BrahmaGS (GPL-3.0, per-frame orchestration wrapper, not native temporal gaussians) plus this repository itself |
| LB2-8 | ComfyUI v0.23.0 added a native GAUSSIAN type (PR #14190); no 4D | TRUE | [verified \| https://github.com/comfyanonymous/ComfyUI/pull/14190 \| as-of 2026-07-12 \| "Adds a generic GAUSSIAN type and a set of utility nodes for working with 3D Gaussian splats"] — merged 2026-05-31; v0.23.0 adds .ply/.ksplat/.spz I/O, EWA render, mesh extraction, TripoSplat (#14210); no temporal support |
| LB2-9 | LHM: code Apache-2.0, weights Apache-2.0, official ComfyUI branch exists | TRUE | [verified \| https://github.com/aigc3d/LHM \| as-of 2026-07-12 \| repo license "Apache-2.0"; README: "See branch feat/comfyui for more information!"] — HF card 3DAIGC/LHM-1B license Apache-2.0; note: per the owner correction in v6/6.16, LHM stays OUT of the core product (image→avatar is the wrong tool); its clean license matters only for auxiliary experiments |
| LB2-10a | MHR (Meta) = Apache-2.0 | TRUE | [verified \| https://raw.githubusercontent.com/facebookresearch/MHR/main/LICENSE \| as-of 2026-07-12 \| "Apache License Version 2.0, January 2004"] |
| LB2-10b | Anny (NAVER) = Apache-2.0, with a non-commercial exception | PARTIAL | [verified \| https://github.com/naver/anny \| as-of 2026-07-12 \| "The code of Anny, Copyright (c) 2025 NAVER Corp., is licensed under the Apache License, Version 2.0"] — assets MakeHuman/MPFB2 = CC0 1.0; the optional SMPL-X-topology mode (v0.3, 2026-02-04) is non-commercial only |
| LB2-11 | Wan 2.1 (Alibaba) code and weights = Apache-2.0; commercial fine-tuning permitted | TRUE | [verified \| https://github.com/Wan-Video/Wan2.1 \| as-of 2026-07-12 \| "Apache-2.0 license" + "We claim no rights over the your generated contents, granting you the freedom to use them"] — load-bearing for Variant 2 (Flex4DHuman-replication backbone) |

---

## Section C1 — Facade claims: repository, scale, capture rig

The dominant failure mode here is **verb tense**: research docs (6.10, 6.21) honestly state that the 90-camera rig is not built and no data exists, while facade docs (root README, dataset/, 6.0-overview) describe the same things in the present tense as existing assets.

| ID | Claim (short) | Verdict | Evidence |
|---|---|---|---|
| C1-1 | "processing petabytes of volumetric data" (README) vs "hundreds of terabytes" (dataset docs) | INTERNAL-CONTRADICTION | [internal \| README.md L47 + dataset/ENTERPRISE_DATASET.md L7] — "processing petabytes of volumetric data" vs "Thousands of high-quality human scans, hundreds of terabytes of data"; 100s of TB = 0.1–0.9 PB, an order-of-magnitude apart |
| C1-2 | Enterprise dataset exists ("What you get": scans, S3/CDN delivery) | FALSE-AS-EXISTING | [internal \| dataset/ENTERPRISE_DATASET.md L7-10] — repo contains zero data artifacts (65 files, no .ply/.mp4/media); download_free.py default repo_id="" ("Set when dataset is published"); README itself admits "There is no downloadable build or dataset yet." |
| C1-3 | "Proprietary 90-camera studio rig, 4K, 60 FPS, hardware-synchronized" (as training data) | INTERNAL-CONTRADICTION | [internal \| v6/6.0-overview.md L15 vs v6/6.10 L5-8 + v6/6.21] — 6.10: "the 90-camera rig is not yet built"; 6.21: owner's actual kit = "2× GoPro HERO 13, no genlock"; sync table: "True genlock ❌ not on GoPro". As a future-rig requirement it is legitimate; as a description of existing training data it is fiction |
| C1-4 | "complete animated avatar in under 3 seconds" | PARTIAL | [internal \| v6/6.0 L13,L101 + v6/6.2 L119] — the timed sum covers only LRM (~0.5s) + refinement (~1-2s); pose estimation (MHR), DINOv2 features, motion system and compression have no time budget anywhere |
| C1-5 | "13M Gaussians per scene / 161 params" | PARTIAL | 161 params confirmed against the 4DGS paper [verified \| https://arxiv.org/abs/2410.13613 \| as-of 2026-07-12 \| "144 out of the total 161 parameters are 4D spherical harmonics (SH) coefficients"]; 13M for a single human is 5–100× above published avatar counts: GAvatar caps at ~2M [verified \| https://arxiv.org/abs/2312.11461 \| as-of 2026-07-12 \| "We stop densification when the total number of Gaussians exceeds 2 million"], relightable avatars use ~100K [verified \| https://arxiv.org/abs/2407.10707 \| as-of 2026-07-12 \| "about 100K Gaussians"]; v6 already softened to "1-13M" as a range |
| C1-6 | Data filtering "65-75% without quality loss" | RETRACTED-IN-V6 | [internal \| v6/6.4 L151-156] — the file's own audit: "the original '~65-75% total' figure was internally inconsistent with the per-stage numbers"; current text: "stages compound to up to ~84% reduction; start conservative" — still an unvalidated extrapolation [hypothesis \| run the filtering ablation of 6.4 on real capture data] |
| C1-7 | v6/README "What changed from v5" lists 7 overstatement corrections, ~30 inline fixes, retractions of 2026-07-11 | TRUE | [internal \| v6/README.md L7-16] — v6 demonstrates a real self-correction culture; the facade marketing docs have none |

---

## Section C2 — Methods and performance claims

| ID | Claim (short) | Verdict | Evidence |
|---|---|---|---|
| C2-1 | Near-linear (1+ε)-approx Chamfer line of work (NeurIPS 2023 + ICALP 2025) | TRUE | [verified \| https://arxiv.org/abs/2307.03043 \| as-of 2026-07-12 \| "our algorithm runs in time O(nd log(n)/ε²)"] + [verified \| https://arxiv.org/abs/2505.08957 \| as-of 2026-07-12 \| "we improve the running time further, to O(nd(loglog(n)+log(1/eps))/eps^2)"] (ICALP 2025, LIPIcs vol. 334) |
| C2-2 | SHARP ablation: removing perceptual loss → "+0.06 LPIPS" | FALSE | [verified \| https://arxiv.org/html/2512.10685 \| as-of 2026-07-12 \| Table 8: LPIPS 0.270 vs 0.143 (ScanNet++), 0.548 vs 0.421 (T&T)] — actual delta ≈ +0.13 on both datasets; direction correct, figure wrong; v6 flagged it "unconfirmed", upgrade to refuted |
| C2-3 | "5–15 dB PSNR gap visible vs occluded regions" | NOT-FOUND | [not-found-as-of 2026-07-12 \| queries: 'feed-forward 3D reconstruction PSNR gap visible versus occluded "dB"'; '"5-15 dB" PSNR occluded gaussian splatting human'] — nearest documented figure is an overall 5.7–6.2 dB generalization gap [verified \| https://arxiv.org/abs/2505.23481 \| as-of 2026-07-12 \| "A generalization gap of 5.7-6.2 dB is consistently observed"] |
| C2-4 | HumanSplat: "0.3s extraction, >150 FPS" | PARTIAL | [verified \| https://arxiv.org/html/2406.12459 \| as-of 2026-07-12 \| "latent reconstruction for 3DGS takes only about 0.3s" + "The video diffusion model takes about 9s" + "exceeding 150 FPS on a NVIDIA A100 GPU"] — component figures real; full inference ≈9.3s; headline misleading without the diffusion stage |
| C2-5 | GNN on 2M cloth Gaussians ≈ 5ms/frame (200 FPS) on RTX 4090 | HYPOTHESIS | [hypothesis \| napkin estimate, correctly disclosed in v6/6.3 L122; memory-bound analysis puts the realistic range at 5–50 ms/frame, so 200 FPS is a best case off by up to 3-10×; benchmark before relying on it] |
| C2-6 | Adam v_t ≈ diagonal empirical Fisher ("Squisher" idea) | TRUE | [verified \| https://arxiv.org/abs/2507.18807 \| as-of 2026-07-12 \| "an approximation of the Fisher diagonal can be obtained 'for free' by recycling the squared gradient accumulator"] — v6's caveats (mini-batch vs per-example, EMA staleness) match the paper's own clarifications |
| C2-7 | MEGA: 190× compression, "161→17 params", ~6% active Gaussians | PARTIAL | [verified \| https://arxiv.org/abs/2410.13613 \| as-of 2026-07-12 \| "~190×" (Technicolor), "only about 6% of Gaussians actively participate in rendering at any given time", "the DC color component requires only 3 parameters"] — correct value is 161→20 (17 base + 3 DC), already fixed in v6; 6% is a property of vanilla 4DGS, not of MEGA output |
| C2-8 | "600 frames ≈ 20 GB → ÷190 → ~105 MB" | FALSE (self-corrected) | [internal \| v6/6.6 L130-131] — category error: 190× is measured against the time-amortized 4DGS model, not a naive per-frame dump; the file's 2026-06-12 audit already retracted the arithmetic |
| C2-9 | P-4DGS: 40–90× compression, ~1 MB per scene | TRUE | [verified \| https://arxiv.org/abs/2510.10030 \| as-of 2026-07-12 \| "achieving up to 40× and 90× compression on synthetic and real-world scenes, respectively" + storage "around 1MB on average"] — the "10-20 MB per clip" figure next to it is labeled in-file as own extrapolation, consistent |

---

## Section C3 — Hardware and economics claims

| ID | Claim (short) | Verdict | Evidence |
|---|---|---|---|
| C3-1 | B200: 208B transistors, NV-HBI 10 TB/s, HBM3e 8 TB/s | TRUE | [verified \| https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing \| as-of 2026-07-12 \| "Packed with 208 billion transistors" + "10 TB/second chip-to-chip link"] — capacity nuance: shipping SKUs list 180 GB/GPU, not the announced 192 GB [verified \| https://www.nvidia.com/en-us/data-center/dgx-b200/ \| as-of 2026-07-12 \| "1,440 GB total" ÷ 8] |
| C3-2 | A product named "B200 Ultra" with 288 GB exists | FALSE | [verified \| https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ \| as-of 2026-07-12 \| "Blackwell Ultra GPUs contain up to 160 SMs and 288GB HBM3E Memory"] — 288 GB belongs to B300/Blackwell Ultra; v6's own audit already corrected the name |
| C3-3 | FP8: "~99% retention, 33% faster, 50% KV cache reduction" as evidence for 4DGS training | FALSE-BY-CONTEXT | [internal \| v6/6.5 L52-54, Ref 8 is an LLM-inference article] — a feed-forward geometric regressor has no KV cache (no autoregression); all three numbers are LLM-inference metrics; the valid argument is FP8 Transformer Engine maturity, which the file eventually makes |
| C3-4 | H100 rent $2–3/hr; 8×H100 ≈ $17K/mo; training ≈ $57K | PARTIAL | [verified \| https://www.runpod.io/pricing \| as-of 2026-07-12 \| H100 SXM $2.99/hr] — achievable at RunPod/Vast-class pricing; Lambda has drifted to $3.99–4.29/hr [verified \| https://lambda.ai/pricing \| as-of 2026-07-12 \| H100 tiers $3.99-4.29] so the $17K/mo figure is provider-dependent; $57K arithmetic internally consistent |
| C3-5 | "$40,000+ savings identified" (B200 vs H100) | RETRACTED-IN-V6 | [internal \| v6/README L16] — "the originally claimed '$40K+ savings' did not survive the 2026 price check — per-run costs are roughly comparable"; original figure compared equal wall-clock at stale B200 prices |
| C3-6 | Budget "$121K, 9–13 months" | PARTIAL (legacy) | [internal \| v6/6.0 L177-184 + v6/6.24 §24.1, §24.6] — internally consistent but excludes salaries and rig ops; 6.24 reconciles scopes explicitly: fine-tune MVP ≈ $5-10K, good-quality ≈ $30-60K, full Track A ≈ $62-70K (6.9); $121K is the legacy from-scratch upper bound, not the current plan |
| C3-7 | NVFP4: 1.59× throughput, selective BF16 layers | TRUE | [verified \| https://developer.nvidia.com/blog/using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy/ \| as-of 2026-07-12 \| "Gains in throughput can be up to 1.59x over BF16 baseline" + "maintaining the final four transformer layers in BF16 proved sufficient"] — "up to" qualifier applies; unvalidated for geometric regression |
| C3-8 | RTX 6000 Pro Blackwell: 96 GB, sm_120 toolchain pain | TRUE | [verified \| https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ \| as-of 2026-07-12 \| "96 GB GDDR7 with error-correction code (ECC)"] + [verified \| https://github.com/nerfstudio-project/gsplat/issues/855 \| as-of 2026-07-12 \| build emitted only "-gencode=arch=compute_90,code=sm_90"] — stock toolchains miss Blackwell CC 12.0; PyTorch nightly cu128/cu130 required |

---

## Section C4 — Legal and market claims

| ID | Claim (short) | Verdict | Evidence |
|---|---|---|---|
| C4-1 | Inria commercial license "~80K EUR" | RUMOR | [rumor \| v5/5.7.1 L21 "по неофициальным источникам"; v5/5.7.2 L35 escalates to "по подтвержденным неофициальным данным" — an oxymoron; v6/6.7 L15 correctly says "no public pricing exists; treat as unconfirmed rumor"] — [not-found-as-of 2026-07-12 \| queries: 'Inria diff-gaussian-rasterization commercial license price EUR'; '"gaussian splatting" license "80,000" OR "80 000" euros Inria SATT'] — only contact stip-sophia.transfert@inria.fr exists |
| C4-2 | Meshcapade "~150K EUR/yr" | RUMOR | [rumor \| v5/5.7.1 L57; v6/6.7 L18 labels it "Industry rumors"] — [not-found-as-of 2026-07-12 \| queries: 'Meshcapade SMPL commercial license pricing'; '"Meshcapade" "150,000" OR "150k" euros'] — meshcapade.com/infopages/licensing.html shows no figures |
| C4-3 | "SMPL-X body model is CC-BY-4.0" | FALSE (bare claim) | [verified \| https://smpl-x.is.tue.mpg.de/modellicense.html \| as-of 2026-07-12 \| "Any other use, in particular any use for commercial... purposes is prohibited"] — CC-BY-4.0 (bodylicense.html) covers only GENERATED body meshes; current v6/6.7 already carries the correct two-license distinction |
| C4-4 | Patent US10395411B2 (SMPL) exists | TRUE | [verified \| https://patents.google.com/patent/US10395411B2/en \| as-of 2026-07-12 \| "Skinned multi-person linear model", assignee Max Planck Gesellschaft, granted 2019-08-27] |
| C4-5 | MHR: Apache-2.0; 45 shape / 204 pose / 72 expression params, 7 LOD | TRUE | [verified \| https://github.com/facebookresearch/MHR \| as-of 2026-07-12 \| README: "45 shape parameters", "204 model parameters", "72 expression parameters", "7 levels of detail (LOD 0-6)"; LICENSE = Apache-2.0] |
| C4-6 | Anny v0.3 (2026-02-04) SMPL-X topology compat; NC-only for that mode | TRUE | [verified \| https://github.com/naver/anny \| as-of 2026-07-12 \| "2026-02-04: v0.3: 'smplx' topology available for interoperability with SMPL-X" + "downloaded for non-commercial use only"] |
| C4-7 | gsplat: "up to 4x less GPU memory with up to 15% less time" vs Inria | TRUE | [verified \| https://github.com/nerfstudio-project/gsplat \| as-of 2026-07-12 \| "up to 4x less GPU memory with up to 15% less time to finish than the official implementation"] — v6 quotes it exactly; v5's "10%" and "15-20%" variants are both inaccurate |
| C4-8 | Anthropic $1.5B settlement as training-data precedent | PARTIAL | [verified \| https://www.npr.org/2025/09/05/nx-s1-5529404 \| as-of 2026-07-12 \| settlement announced 2025-09-05, ~$3,000/work, ~482,460 works] — preliminary approval 2025-09-25; final fairness hearing was scheduled for 2026-05-14 and its outcome was NOT verified here; cite as "settlement, preliminary approval granted", not a finished precedent |
| C4-9 | Clean-room design does not protect against patents | TRUE | [verified \| https://en.wikipedia.org/wiki/Clean-room_design \| as-of 2026-07-12 \| "independent invention is not a defense against patents, clean-room designs typically cannot be used to circumvent patent restrictions"] |
| C4-10a | RenderPeople "4D People" (Volucap) is a live commercial 4D-scan vendor | TRUE | [verified \| https://renderpeople.com/4d-people/ \| as-of 2026-07-12 \| "captured in one of world's leading volumetric studios operated by Volucap"] — sold at EUR 79/model; ML/CV licensing gated behind quote |
| C4-10b | HumanDataset: 35,000 scans, 330-camera rig | TRUE | [verified \| https://humandataset.com/ \| as-of 2026-07-12 \| "Over 35,000 scans and more than 5,500 refined 3D models" + "330-camera setup"] — claims GDPR Art. 6(1)/9(2) compliance; direct competitor benchmark for the dataset asset |
| C4-10c | Twindom sells people scans; public AI-training terms | PARTIAL | [verified \| https://web.twindom.com/ \| as-of 2026-07-12 \| "scanned over 100,000 people"] — no public AI/ML dataset licensing terms on the site; matches v6/6.10's "no public terms" |

---

## Section N — New v6 docs (6.8–6.24): key claims carried into v7

These entered v6 after the earlier audits and are the basis of the v7 variants. Statuses are honest: `[internal]` = a fact about what the repo's own docs decide/say (checkable by opening the file); outward claims not re-fetched in this pass stay `[inherited-unverified]`.

| ID | Claim (short) | Verdict | Evidence |
|---|---|---|---|
| N-1 | OWNER DECISION (2026-06-12): no purchased third-party datasets; commercial track = own rig data only | BINDING | [internal \| Research-docs/v6/6.10-data-strategy.md L8] — "OWNER DECISION (2026-06-12): no purchased third-party datasets. The commercial track trains on our own rig data only" |
| N-2 | OWNER CORRECTION (2026-06-16): goal = faithful 4DGS of REAL VIDEO, not avatar-from-photo; LHM/IDOL/AniGS/PSHuman are the wrong tool for the core product | BINDING | [internal \| Research-docs/v6/6.16-zero-training-mvp-pipeline.md L11-12] — "the goal is FAITHFUL 4DGS OF REAL VIDEO — not an avatar generated from a photo" |
| N-3 | Owner ruling: DNA-Rendering and ActorsHQ are OUT, permanently | BINDING | [internal \| Research-docs/v6/6.17-obtainable-data-plan.md preamble] — "the owner's standing ruling: DNA-Rendering and ActorsHQ are OUT, permanently" |
| N-4 | Owner directive: plan must be realizable solo on ONE RTX 6000 Pro, no institutional gating | BINDING | [internal \| Research-docs/v6/6.19-buildable-mvp-synthesis.md preamble] |
| N-5 | Actual capture kit: 2× GoPro HERO 13, no genlock, ~5 subjects, no green screen | BINDING | [internal \| Research-docs/v6/6.21-2gopro-reality-adapted-plan.md header + §21.1 sync table "True genlock ❌ not on GoPro"] |
| N-6 | Canonical goal prompt = v3 (6.21 §21.9); v1/v2 superseded; NONE executed yet | TRUE | [internal \| Research-docs/v6/6.21 §21.4/§21.7/§21.9] — repo has no train/, state/, eval/; comfyui_nodes are 113-line placeholders |
| N-7 | Flex4DHuman (arXiv 2606.13655): Wan 2.1 1.3B backbone + PRoPE, trained on 32×H100, DNA-Rendering only, 25.44 PSNR | INHERITED | [inherited-unverified] — extracted from the paper PDF in 6.8 §8.1.1 with repro-score 7.5/10; re-verify the PDF quotes before building (M0.5 gate does this) |
| N-8 | Cloud training budget (6.24): Layer-0 ≈ $0-500; fine-tune MVP ≈ $5-10K; good-quality ≈ $30-60K; #1 cost driver = number of failed training campaigns | ESTIMATE | [estimate \| v6/6.24 TL;DR — honest arithmetic over H100 $2-3/hr baselines with stated unknowns (batch size from the paper unknown → 10× spread)] |
| N-9 | $0-license synthetic stack: MakeHuman CC0, Poly Haven CC0, Rocketbox MIT, CMU mocap commercial-OK | INHERITED | [inherited-unverified] — licenses were read first-hand per 6.10 §10.1; spot-check before commercial reliance |
| N-10 | MV-Performer repo has NO license file; Sapiens = CC BY-NC (hard NC anchor for that path) | INHERITED | [inherited-unverified \| v6/6.12 §12.1, file listing checked directly at the time] |
| N-11 | VGGT-1B-Commercial = only camera front-end with an explicit commercial license; MegaSaM = NC-track | INHERITED | [inherited-unverified \| v6/6.18 — marked [U] in-file: exact license text not fetched (403); must be read before commercial use] |
| N-12 | 4C4D: root MIT does not clear vendored NC deps (Inria rasterizer, simple-knn, MASt3R CC BY-NC-SA) | TRUE | [internal \| v6/6.22 §22.6 — source headers were read directly in the 2026-07-11 verification pass; consistent with LB2/C4 findings on Inria inheritance] |
| N-13 | ArtiFixer: code Apache-2.0, sole checkpoint = NVIDIA OneWay Noncommercial; cheap zero-shot back-fill experiment proposed, NOT run | INHERITED | [inherited-unverified \| v6/6.23 §23.6, §23.9.4] |
| N-14 | PKU-DyMVHumans (56-60 cams) is the DNA-Rendering replacement; HF tag = `c-uda`, license question open | INHERITED | [inherited-unverified \| v6/6.17 + 6.21 §21.8 — resolving the PKU license is gate M1 of the v3 prompt] |
| N-15 | Gracia AI (Track B): $2.9M funding, .mint format, ~1 GB/min compression — "verified as claims"; contract with owner since 2025-08-05; never on the critical path | INHERITED | [inherited-unverified \| v6/6.11 §11.1 — press-release-level sourcing; kill-criteria: written ML rights or no deal] |
