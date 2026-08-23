# 4DAnyone (Ant Research) — assessment for the capture → 4DGS pipeline

> Part of Research-docs v7. Research pass 2026-08-23, six days after the model weights appeared on Hugging Face (repo created 2026-08-17) [registry AN-3]. Unlike the FLUX 3 pass, evidence here is thick: code, weights, per-file license map and paper are all public. Every claim below carries a status tag; verdicts live in [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) section AN.

## 1. Verdict in one table

| Pipeline stage | 4DAnyone today | Why |
|---|---|---|
| Reference recipe for Variant 2 (monocular → multi-view → 4DGS fine-tune track) | **YES — the strongest published one** | released Wan-family multi-view human generator with code + weights, exactly the CAT4D-shaped pipeline v7 planned around [registry AN-1, AN-4, F-22] |
| Back-fill of the subject's unseen side (the clearly-labeled research track from the README) | **YES, as research track only** | pose-conditioned generation of unseen views is its core function; generated views remain invention, so the faithful-capture wall applies [registry AN-13, N-2] |
| Drop-in commercial component behind our license wall | **NO — not as bundled** | GVHMR motion stage is research/non-profit, YOLOv8x detector is AGPL-3.0, SMPL-X body model is research-only [registry AN-5, AN-7, LB2-1] |
| Faithful surround replacement for multi-camera capture (Variant 1) | **NO** | it invents unseen views; Variant 1 is per-capture fitting of real footage with no generation by construction [registry AN-13, N-2] |
| Cheap MVP-0-class experiment on our own GoPro footage | **YES, now** | local inference, automatic model download, nerfstudio export script shipped [registry AN-10, AN-12] |
| ComfyUI wrapper-node candidate | **YES, non-commercial tier first** | Apache-2.0 code is wrappable; the bundled motion stack forces the non-commercial label until swapped [registry AN-6, AN-5] |

The short version: **4DAnyone is the open, runnable version of what FLUX 3 only appeared to be** — camera-pose-conditioned multi-view video from one casual clip, with weights on disk and no vendor licence over our inputs [registry AN-10, F-17]. Its limits are ours to verify, not a vendor's to hide.

## 2. What 4DAnyone actually is (source-verified)

- One sentence, theirs: "4DAnyone turns a casual monocular video into multi-view videos, enabling downstream 4DGS reconstruction" [verified | https://github.com/ant-research/4DAnyone | as-of 2026-08-23 | "4DAnyone turns a casual monocular video into multi-view videos, enabling downstream 4DGS reconstruction."].
- Paper: "4DAnyone: Create Anyone in 4D from a Casual Monocular Video", Jin, Xie, Zhang, Shen, Xu, Shen, Bao, Zhou, Xu — arXiv preprint [verified | https://arxiv.org/abs/2608.20335 | as-of 2026-08-23 | bibtex in the model card names arXiv:2608.20335, year 2026]. Author list includes Hujun Bao and Xiaowei Zhou — the zju3dv orbit that also produced GVHMR and Diffuman4D [registry AN-2, F-21].
- Code: Apache-2.0 [verified | https://github.com/ant-research/4DAnyone/blob/main/LICENSE | as-of 2026-08-23 | "Apache License / Version 2.0, January 2004"] [registry AN-6].
- Weights: published at AntResearch/4DAnyone on Hugging Face — `model.safetensors` plus the full inference stack; two community demo Spaces already exist [verified | https://huggingface.co/AntResearch/4DAnyone | as-of 2026-08-23 | model card file tree lists 4danyone/model.safetensors, Wan2.2_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth, gvhmr/*] [registry AN-3].
- Backbone: the shipped bundle is Wan-family — the VAE is taken from Wan2.2-TI2V-5B and the UMT5-XXL text encoder + tokenizer from Wan2.1-T2V-1.3B, per the vendor's own per-file license map [registry AN-4]. This is the same Wan lineage v7's Variant 2 already builds on [registry N-7, LB2-11].
- Traction snapshot: 389 stars, 32 forks on GitHub; 44 likes on Hugging Face [verified | https://github.com/ant-research/4DAnyone | as-of 2026-08-23 | GitHub sidebar counters, read 2026-08-23] [registry AN-14].

## 3. The pipeline, concretely

1. **Motion recovery** — GVHMR (git submodule) recovers world-grounded SMPL-X body motion from the input clip; checkpoints for GVHMR, HMR2.0a, ViTPose and YOLOv8x ship in the model repo [registry AN-7, AN-3].
2. **Multi-view generation** — the Wan-family generator, conditioned on the recovered skeleton and user-specified camera layout, produces sparse anchor views and then dense views [registry AN-4, AN-9].
3. **Reconstruction export** — `scripts/export_nerfstudio.py` writes a nerfstudio dataset per frame; the documented command is per-frame 3DGS: "ns-train splatfacto --data data/nerfstudio/<clip>/frame_000" [verified | https://github.com/ant-research/4DAnyone | as-of 2026-08-23 | "ns-train splatfacto --data data/nerfstudio/<clip>/frame_000 --pipeline.model.background-color random"] [registry AN-12].

Camera layouts are explicit and scriptable — full orbits of 6 or 24 views, three-pitch-layer 48-view rigs, or an 8-view frontal arc via `--start_yaw -90 --yaw_span 180`; "views_per_layer ... must be divisible by 4 or 6" [verified | https://github.com/ant-research/4DAnyone | as-of 2026-08-23 | "views_per_layer: number of evenly spaced views per pitch layer; must be divisible by 4 or 6."] [registry AN-9]. That the tool speaks in ring-and-pitch rig vocabulary matters to us: its synthetic layouts can be set to mimic our real GoPro rig geometry, which makes real-vs-generated comparisons well-posed [hypothesis | configure a layout matching the physical rig, capture the same performance, compare fitted splats].

Input constraints, verbatim: the input video "is 720p or higher, with 1080p recommended; uses a 9:16 portrait aspect ratio; shows one person in a full-body or upper-body shot; has at least 121 frames; contains only mild camera motion" [verified | https://github.com/ant-research/4DAnyone | as-of 2026-08-23 | "is 720p or higher, with 1080p recommended; uses a 9:16 portrait aspect ratio; shows one person in a full-body or upper-body shot; has at least 121 frames; contains only mild camera motion."] [registry AN-8].

Hardware reality: the Todos list "Low-memory inference (<32 GB)" as unshipped, so current inference exceeds 32 GB VRAM — H100-class or the free RTX 6000 Pro, consistent with v7's compute contingency [verified | https://github.com/ant-research/4DAnyone | as-of 2026-08-23 | "- [ ] Low-memory inference (<32 GB)"] [registry AN-11, C3-4].

## 4. The license wall, per file — the decisive section

The vendor is unusually honest here: the model repo's LICENSE opens "This repository is a collection of separately licensed model and example assets. No single license applies to every file" [verified | https://huggingface.co/AntResearch/4DAnyone/blob/main/LICENSE | as-of 2026-08-23 | "This repository is a collection of separately licensed model and example assets. No single license applies to every file."] [registry AN-15]. Their own asset map, re-sorted by what it means for us:

| Asset | Their license | Our wall |
|---|---|---|
| 4DAnyone generation checkpoint (`model.safetensors`) | Apache-2.0 [registry AN-6] | clean side |
| Wan2.2 VAE + Wan2.1 UMT5-XXL encoder/tokenizer | Apache-2.0 [registry AN-4] | clean side |
| ViTPose checkpoint | Apache-2.0 [registry AN-5] | clean side |
| HMR2.0a checkpoint (4DHumans) | MIT license [registry AN-5] | clean side |
| GVHMR checkpoint + code | research/non-profit only [registry AN-7] | **dirty side** |
| YOLOv8x detector | AGPL-3.0 [registry AN-5] | **dirty side** for a closed product; viral |
| SMPL-X body model (required, fetched by `download_smplx.py`) | research-only; commercial via Meshcapade [registry LB2-1] | **dirty side** |
| Sapiens2-derived keypoint schema inside `smplx_to_goliath70.pt` | Sapiens2 license [registry AN-5] | read before commercial use |

GVHMR's license is explicit both ways: use is granted "for educational, research and non-profit purposes only" and "For commercial uses of this software, please send email to xwzhou@zju.edu.cn" [verified | https://github.com/zju3dv/GVHMR/blob/main/LICENSE | as-of 2026-08-23 | "educational, research and non-profit purposes only" + "For commercial uses of this software, please send email to xwzhou@zju.edu.cn"] [registry AN-7].

**Consequence.** The generator itself sits on the clean side; the *motion-recovery front end* is what drags the bundle to the dirty side. A commercial path exists on paper: swap GVHMR+SMPL-X+YOLOv8x for a commercially clean pose stack (or license them — Meshcapade for SMPL-X, ZJU by email, Ultralytics commercial license), keep the Apache-2.0 generator. Whether the generator works acceptably with a different SMPL-X-compatible motion source is an experiment, not a fact [hypothesis | feed the generator poses from a clean HMR stack and compare output quality against the GVHMR path]. Until then: **non-commercial research track only** — same shelf as Variant 2a, whose outputs are knowledge, never product [registry N-1, AN-5].

## 5. Where it plugs into our variants

- **MVP-0 (now, ≈ $0 plus GPU time).** Run 4DAnyone unmodified on one of our own casual clips, export frame_000, fit splatfacto, and measure what v7 already defines: held-out-camera PSNR/LPIPS and flicker across frames [registry AN-12]. This is the cheapest possible answer to "how far does monocular-generated multi-view actually get" — the question FLUX 3 could not even be asked [registry F-5]. Deliverable: numbers in DATASET-STATUS.md, not adjectives.
- **Variant 1 (budget replay): no role in the render path.** Variant 1 is faithful per-capture fitting of real multi-camera footage; nothing generated enters it [registry N-2]. Reusable anyway: the `export_nerfstudio.py` camera/dataset conventions and the ring-and-pitch layout vocabulary for our own rig metadata [registry AN-9, AN-12].
- **Variant 2 (fine-tune track): the main event.** v7 planned Variant 2 around a Wan 2.1 1.3B fine-tune in the CAT4D shape — monocular video to multi-view video to 4D fit [registry N-7, F-22]. 4DAnyone is that shape, published, with weights, on the same model family. Three uses in ascending ambition: (a) baseline to beat — any fine-tune we pay for must outperform a free checkpoint; (b) recipe donor — conditioning design (skeleton + camera layout) transferable to our own training; (c) starting checkpoint for continued fine-tuning on own-rig data, legally reviewable since the checkpoint is Apache-2.0 [registry AN-6]. The 2a/2b split is untouched: anything touching research-licensed data or the GVHMR path stays 2a (never commercial) [registry N-1].
- **Back-fill research track.** The public README promises learned back-fill as "a separate, clearly-labeled research track" [internal | README.md]; 4DAnyone is the first credible open engine for it. Labeling rule stands: generated views are invention and are never sold inside a faithful-capture deliverable [registry N-2, AN-13].
- **ComfyUI nodes.** An Apache-2.0 code base with a scripted CLI wraps cleanly into nodes for the ComfyUI-CustomNodePacks repo (video in → multi-view videos + cameras.json out, feeding our 4DGS nodes). VRAM (>32 GB today) and the license label are the gating facts to surface in the node UI [registry AN-11, AN-5].

## 6. Contrast with the FLUX 3 pass — why this one passes where that one failed

| Question | FLUX 3 (2026-07-27 pass) | 4DAnyone (this pass) |
|---|---|---|
| Camera-pose conditioning exists | no — a prompt, not a feature [registry F-7, F-10] | yes — explicit yaw/pitch/layout arguments [registry AN-9] |
| Weights available | no [registry F-4] | yes [registry AN-3] |
| License text exists | no [registry F-14] | yes, per-file [registry AN-15] |
| Vendor licence over our inputs | perpetual, sublicensable, incl. training — hard stop [registry F-17] | none: local inference, Apache-2.0 code [registry AN-10] |
| Geometric consistency independently testable | no, access-gated [registry F-5] | yes, today, on our own footage [registry AN-12] |

The structural objection to generated views as 4DGS input — independent hallucination instead of one constrained 3D scene [registry F-20] — is exactly what 4DAnyone's skeleton-and-camera conditioning is built to answer. Whether it succeeds to our acceptance thresholds is measurable now; run the COLMAP/held-out-camera test from the FLUX 3 doc against real 4DAnyone output [hypothesis | SfM + held-out PSNR/LPIPS on generated dense views; the test FLUX 3's gating made impossible].

## 7. Known gaps (theirs and ours)

1. **No temporal 4DGS reconstruction shipped.** The documented path is per-frame 3DGS via splatfacto; open-source 4DGS reconstruction is an unshipped Todo — "Support 4DGS reconstruction with an open-source method" [verified | https://github.com/ant-research/4DAnyone | as-of 2026-08-23 | "- [ ] Support 4DGS reconstruction with an open-source method"] [registry AN-12]. Same gap Diffuman4D left [registry F-21]; our own temporal-fit work remains the differentiator, not a duplicated effort.
2. **Single person, portrait, ≥121 frames, mild camera motion** — a narrow input envelope; our capture protocol can satisfy it deliberately, arbitrary client footage may not [registry AN-8].
3. **VRAM above 32 GB** until their low-memory Todo lands [registry AN-11].
4. **Quality is unmeasured by us.** Everything above is provenance and plumbing; no fidelity claim is made here until MVP-0 numbers exist [hypothesis | §5 MVP-0 experiment].

## 8. Actions

1. Add the MVP-0 experiment from §5 to the experiment queue (cost: GPU hours only) and record results in DATASET-STATUS.md.
2. Legal note in LEGAL-LICENSING.md when the experiment is worth pursuing commercially: generator = clean side, motion front end = dirty side, three named licensing routes [registry AN-5, AN-7, LB2-1].
3. If MVP-0 numbers are competitive: scope a ComfyUI wrapper node (non-commercial label) and a clean-pose-stack substitution experiment before any Variant 2b money [registry AN-5, N-1].
