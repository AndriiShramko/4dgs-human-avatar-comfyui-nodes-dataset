# POINT4D — assessment: a long-range 3D tracker, and where it may touch this pipeline

**Date of this pass: 2026-09-13.** Point4D (CMU, arXiv 2609.09145, announced publicly by its authors in early September 2026) is the first feed-forward 4D method with published evidence at the length of a real performance take. This page says what it actually is, what it verifiably does, which of this project's stages it can and cannot touch, and what its licence lets us do. Verdict up front, so nobody has to read to the end:

> **The code is clean and the checkpoint is not.** Point4D's code is MIT; the released weights are a derivative of non-commercial training data with no licence of their own. So Point4D is a **non-commercial-sub-path instrument** under the existing wall rule [registry N-12] — the same shelf as MegaSaM and ArtiFixer [registry N-11, N-13]. It is a *tracking and measurement* front-end, never a renderer, and it does **not** solve back-fill.

## 0. Provenance of the evidence on this page <!-- lint-skip -->

The authoring session's egress proxy blocked `arxiv.org`, `huggingface.co` and `point-4d.github.io`. Paper text was therefore read through the Hugging Face paper-text API (`hf://papers/2609.09145/paper.md`, whose own header names `https://arxiv.org/html/2609.09145` as its source), and repository/licence text through `raw.githubusercontent.com`, which was reachable. Every verified-tag below citing an `arxiv.org` or `huggingface.co` URL therefore names the **canonical** source, not a URL this session opened directly; the `raw.githubusercontent.com` citations were fetched first-hand. **Confirmation step for the owner:** open the arXiv abstract once on your own machine and confirm the title/authors line — everything else on this page is either first-hand or quoted verbatim from the paper text above. This note exists because a registry that cites sources it did not reach is the exact failure v7 was built to stop.

## 1. What it is (verified)

| Fact | Evidence |
|---|---|
| Title / authors / venue | [verified \| https://raw.githubusercontent.com/point-4d/Point4D/main/README.md \| as-of 2026-09-13 \| "Point4D: Long-range 4D Motion Reconstruction — Minsik Jeon · Jay Karhade · Deva Ramanan · Shubham Tulsiani — Carnegie Mellon University"] |
| The core idea: query 3D points, not pixels | [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "our flexible 3D query-based motion decoder that decouples trajectory prediction from image-plane visibility. The predicted 3D endpoints are then directly re-queried in the next chunk without re-projection or matching"] |
| The problem it solves — prior 4D methods cap out at a few dozen frames | [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "unlike existing 4D methods that are limited to short input windows of at most a few dozen frames"] |
| Second contribution: the visual descriptor may come from any frame where the point is visible, not the source patch | [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "extracting and reusing a visual descriptor from an arbitrary frame where the point is visible leads to better performance than relying solely on the source patch"] |
| What it outputs | 3D trajectories + per-frame depth + camera poses. Not appearance, not gaussians. [verified \| https://raw.githubusercontent.com/point-4d/Point4D/main/README.md \| as-of 2026-09-13 \| "traj: (T, N, 3) in frame 0 camera coordinates, conf: (T, N)" and "per-frame scene geometry the tracker computes anyway — depth clouds, camera poses and intrinsics"] |
| Backbone | Depth Anything 3 encoder + geometry heads, decoder trained from scratch [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "We initialize the encoder and geometry heads from Depth Anything 3 pretrained weights, while the decoder is trained from scratch."] |
| Training cost, as a cost anchor | 150 epochs on 8 H100 GPUs [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "We use AdamW with a peak learning rate of 1e-4 for 150 epochs ... Training is done on 8 H100 GPUs"] — wall-clock is not stated [not-found-as-of 2026-09-13 \| queries: paper §3.4 and Appendix A, repo README — no hours/days figure given] |
| Measured at | single A6000 48 GB, input 294×518 [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "All measurements are taken on a single A6000 (48 GB) at an input resolution of 294x518."] |

### The numbers that matter to us

Long-video chaining, 200-frame sequences cut into 48-frame chunks with 8-frame overlap — EPE (lower better) / Survival (higher better) [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| Table 1, caption: "We report EPE, APD, and survival rate on sequences of 200 frames, partitioned into chunks of 48 frames with 8-frame overlap."]:

| Method | PointOdyssey EPE / Surv. | DynamicReplica EPE / Surv. | PStudio EPE / Surv. |
|---|---|---|---|
| 4RC (2D-query feed-forward) | 0.789 / 0.463 | 0.336 / 0.654 | 0.379 / 0.534 |
| VDPM (2D-query feed-forward) | 0.736 / 0.461 | 0.386 / 0.587 | 0.280 / 0.634 |
| SpatialTrackerV2 (iterative) | 0.498 / 0.477 | 0.218 / 0.687 | 0.234 / 0.623 |
| **Point4D** | **0.616 / 0.514** | **0.155 / 0.812** | 0.236 / **0.664** |

<!-- lint-skip: the table body above is a verbatim transcription of Table 1; the tag is on the sentence introducing it. -->

Three further results decide whether this is useful here rather than merely impressive:

- **It holds at real take length.** Extended to 500 frames (PointOdyssey) and 300 frames (DynamicReplica) — 12 and 7 handoffs respectively — Point4D ranks first on every metric, ahead of the iterative trackers too [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "Point4D ranks first on every metric, surpassing not only feed-forward baselines but also iterative trackers. Chunking follows the same setting (48-frame chunks with 8-frame overlap), so longer sequences require more handoffs: 12 for PointOdyssey and 7 for Dynamic Replica."]
- **The whole gain is chaining, not a better decoder.** On single-chunk (≤64-frame) tracking it is merely comparable to 4RC/VDPM/Any4D [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "Point4D performs comparably to existing methods on short sequences that do not require trajectory chaining, which confirms that the long-video gains in Table 1 stem from the proposed 3D query formulation, not a stronger single-chunk decoder."] — so if our takes were short, Point4D would buy us nothing.
- **The margin is largest exactly on occluded points.** Broken down by visibility at the handoff frame, Point4D's biggest lead is on Occluded (PointOdyssey EPE 0.597 vs 4RC 0.915; DynamicReplica 0.202 vs 0.574), and it is *beaten* by 4RC on out-of-frame points on PointOdyssey [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| Table 7 caption: "Point4D achieves the lowest EPE in nearly every visibility, with the largest margins on Occluded points, where 2D re-querying fails but 3D re-querying carries the point through directly." + "For out-of-frame points the advantage narrows, and on PointOdyssey 4RC is better."]

## 2. Why this lands on *our* desk and not just on a reading list

Four of this project's recorded constraints line up with what Point4D is built for. That is the whole reason it is worth a page.

1. **Our unit of work is a take, not a clip.** Variant 1 fits performances of seconds, not a handful of frames [registry LB2-8 \| VARIANT-1 §3 sizes replay at 5–30 s clips]. At GoPro HERO 13 capture rates that is hundreds to thousands of frames — precisely the regime where every other feed-forward 4D method is documented to fail and Point4D is documented to hold.
2. **Two frontal GoPros means constant self-occlusion** [registry N-5]. An arm crossing the torso, a half-turn, a prop: the tracked point disappears and comes back. That is the exact failure mode 3D re-querying is designed to survive, and the ablation above shows it is where the method's margin actually lives.
3. **Our acceptance gate for temporal stability is currently subjective.** VARIANT-1 §5.3 asks for "no visible gaussian flicker in an eyes-on review" plus frame-to-frame LPIPS. Dense long-range 3D tracks are a *geometric* instrument for the same question — see §3.3.
4. **The 4D-fit tools we named consume tracks.** VARIANT-1 §2 names the Instant4D / Shape-of-Motion / MoSca class for the per-video fit, and Shape-of-Motion's own README states it depends on third-party preprocessing "to generate depth maps, object masks, camera estimates, and 2D tracks" [verified \| https://raw.githubusercontent.com/vye16/shape-of-motion/main/README.md \| as-of 2026-09-13 \| "We depend on the third-party libraries in `preproc` to generate depth maps, object masks, camera estimates, and 2D tracks."] Point4D emits depth, poses and 3D tracks from one forward pass — the same shopping list, one model, at take length.

## 3. Where it plugs in — stage by stage

### 3.1 Motion front-end for the per-video 4D fit — the strongest fit, NC-track only

Replace (or add to) the tracks+depth+pose preprocessing that a Shape-of-Motion/MoSca-class fit needs with a single Point4D pass over the whole take. Upside: one model instead of a preproc zoo, and 3D tracks that survive the self-occlusion in §2.2 instead of 2D tracks that break on it. **Status: [hypothesis | run E-06, then feed the tracks to a Layer-0 fit and compare held-out-camera PSNR against the current preproc stack on the same take].** Blocked from the commercial sub-path by §4.

### 3.2 Camera poses and video depth — real, but not a reason to adopt it

Point4D's geometry output beats the other feed-forward 4D methods on KITTI/Bonn/Sintel and matches its own DA3 backbone [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "Point4D performs on par with DA3, while outperforming other feed-forward 4D reconstruction methods on most metrics."] — which is the point: **if all you want is poses and depth, use DA3 directly.** DA3's small checkpoints are Apache-2.0 (§4), so this is also the licence-cheaper route. Point4D's geometry is a free byproduct of tracking, not an argument for it. This does not displace COLMAP for multicam or the VGGT-1B-Commercial question for the commercial monocular front-end [registry N-11].

### 3.3 A measurable temporal-stability metric — the most interesting non-obvious use

VARIANT-1 §5 needs flicker to be a number. Proposal: lay a query grid on frame 0 of the **captured** take and on frame 0 of a **render** of the fitted 4DGS from the same camera; track both with the same Point4D pass; a stable fit means the two trajectory sets agree, and gaussian flicker shows up as divergence long before it is describable in an eyes-on review. This turns §5.3 from a subjective gate into an EPE-style number on our own data, using an instrument that was never trained on it. **Status: [hypothesis | E-06 first establishes that tracks survive our footage at all; only then is the metric meaningful].** Licence caveat in §4 applies with full force here: an NC instrument may measure NC-track work only.

### 3.4 ComfyUI node surface — possible later, deliberately not now

A `Track4D` node over the `GAUSSIAN_SEQUENCE` type [registry LB2-8] is an obvious shape, and the niche is still empty [registry LB2-7]. It is **out of scope until Variant 1 ships**: a node that wraps a checkpoint we cannot ship commercially is a liability, not a feature, and VARIANT-1 §3 already fixed the node package as a thin adapter. Revisit only if §4's clean-retrain route is ever taken.

## 4. The licence wall — where Point4D actually stands

| Component | Status | Evidence |
|---|---|---|
| Point4D **code** | MIT license — clean, crosses the wall | [verified \| https://raw.githubusercontent.com/point-4d/Point4D/main/LICENSE \| as-of 2026-09-13 \| "MIT License / Copyright (c) 2026 Minsik Jeon ... without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies"] |
| Point4D **weights** | **No licence declared at all** — the HF repo holds exactly two files (`.gitattributes`, `point4d_final.pt`), with no model card and no licence tag | [verified \| https://huggingface.co/minsikj/point4d \| as-of 2026-09-13 \| Hub API file listing returns `.gitattributes` and `point4d_final.pt` only; repo tags: `region:us` — no `license:` tag, `README.md` returns not-found] |
| Encoder lineage | Initialised from Depth Anything 3, whose LARGE/GIANT/NESTED checkpoints are CC BY-NC 4.0 and only BASE/SMALL are Apache-2.0 | [verified \| https://raw.githubusercontent.com/ByteDance-Seed/Depth-Anything-3/main/README.md \| as-of 2026-09-13 \| model-card table: DA3-GIANT "CC BY-NC 4.0", DA3-LARGE "CC BY-NC 4.0", DA3-BASE "Apache 2.0", DA3-SMALL "Apache 2.0"] — **which** DA3 variant Point4D used is not stated [not-found-as-of 2026-09-13 \| queries: paper §3.4, Appendix A, repo README — the size is never named]; the 5.80 GB checkpoint implies ≈1.45B params in fp32, i.e. the GIANT class rather than BASE/SMALL [estimate \| 5,797,376,562 bytes ÷ 4 bytes/param; DA3-GIANT is 1.15B params per the same table] |
| Training data | Includes at least two non-commercial datasets: PointOdyssey (19.8% of samples) and Dynamic Replica (19.8%) | [verified \| https://raw.githubusercontent.com/y-zheng18/point_odyssey/main/README.md \| as-of 2026-09-13 \| "The full dataset can be downloaded from the project page, under CC BY-NC-SA 4.0 license"] + [verified \| https://raw.githubusercontent.com/facebookresearch/dynamic_stereo/main/LICENSE \| as-of 2026-09-13 \| "Attribution-NonCommercial 4.0 International"]; sampling ratios from the paper's Table 4 [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "PointOdyssey 19.8% / Dynamic Replica 19.8% / BEDLAM2 19.8%"] |

**Verdict.** Whatever the authors intended, the released checkpoint is a derivative of CC BY-NC data on a probably-CC-BY-NC encoder, published with no licence grant. Under this project's own rule — outputs of NC tools never migrate [registry N-12] — the checkpoint is **non-commercial sub-path only**, and nothing it produces may cross into commercial work, *including* a stability number used to sign off a commercial deliverable. The MIT code is separately clean and may be read, modified and shipped.

**The clean route exists and has a price.** MIT code + an Apache-2.0 DA3-BASE/SMALL encoder + the Apache-2.0 slices of the training mix (Kubric is Apache-2.0 [verified \| https://raw.githubusercontent.com/google-research/kubric/main/LICENSE \| as-of 2026-09-13 \| "Apache License Version 2.0, January 2004"]) would give a commercially clean tracker. Cost anchor: the published recipe is 150 epochs on 8 H100 GPUs; at H100 $2–3/hr [registry C3-4] a 3–10 day run is **$1.2–5.8K** [estimate | 8 GPUs × 72–240 h × $2–3/hr], plus a smaller encoder and a thinner data mix, so quality would land below the released checkpoint by an unknown margin. That is Variant-2-scale spending for a *preprocessing* component — **do not take this route before Variant 1 has shipped and shown it needs one.**

**One free action, worth taking now:** email the authors asking them to put a licence on the HF weights, exactly as E-04 does for MV-Performer [registry N-10]. Costs 15 minutes; a permissive answer would change the verdict above, and silence costs us nothing since the NC use stands either way.

## 5. What it does not do — the rails

- **It is not a renderer and produces no gaussians.** Trajectories, depth, poses. It cannot make a playable volumetric clip and cannot be put on the critical path to one.
- **It does not do back-fill.** The paper's own limitation: "a point that remains occluded or out of frame for an entire chunk has no supporting evidence in that chunk, and its predicted position becomes unreliable" [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "Since the encoder represents only what is observed within the current chunk, a point that remains occluded or out of frame for an entire chunk has no supporting evidence in that chunk, and its predicted position becomes unreliable."]. A performer who turns away hides their front for many consecutive chunks. The unseen back stays a separate research track exactly as the README says — Point4D does not touch it.
- **It is monocular as released.** The demo and inference path take one video and chunk it; we shoot two cameras [registry N-5]. The DA3 backbone is an any-view model, so multi-view input is architecturally plausible, but nothing published or released demonstrates or exposes it [hypothesis | inspect `point4d/inference.py` for whether the encoder accepts a multi-view batch before assuming this is a small change].
- **It does not scale to per-gaussian tracking.** Query-decoding cost grows with the number of queries, and the published measurements stop at 5,000 queries [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "when scaling the number of queries (100 to 5,000) the number of frame is fixed to 48" + "with dense queries it falls behind the dense DPT-head methods"]. Our avatars run 0.1–2M splats [registry C1-5 \| VARIANT-1 §3]; that is 20–400× beyond the measured range [estimate | 100,000–2,000,000 ÷ 5,000]. Treat Point4D as a **sparse-to-moderate** query instrument and design any integration around a grid of queries, not around one query per gaussian.
- **Its geometry depends on predicted depth and Sim(3) chunk alignment** [verified \| https://arxiv.org/abs/2609.09145 \| as-of 2026-09-13 \| "Our formulation also relies on predicted depth and on the Sim(3) alignment between consecutive chunks, so errors in depth prediction can compound across chunks."] — so on our own footage it inherits whatever the depth head does with a human filling the frame. That is what E-06 is for.

## 6. Decision

**Adopt as an NC-track instrument, evaluate via E-06, do not put it on the commercial critical path, do not build a node for it yet.** Nothing here changes Variant 1's plan; it adds one candidate to the 4D-fit front-end and one candidate instrument for the §5 stability gate. If E-06 fails on our footage, this page is the record of why we stopped, and the cost was hours.
