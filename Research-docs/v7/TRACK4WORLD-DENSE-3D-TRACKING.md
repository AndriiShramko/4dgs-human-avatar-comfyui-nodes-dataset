# Track4World — feedforward dense 3D tracking of every pixel: what it is and where it plugs into Shramko-Human-4D

> Part of Research-docs v7. Research pass 2026-09-17, five days after the authors' ECCV 2026 poster [registry T-19]. Trigger: the owner forwarded the authors' LinkedIn announcement. Evidence base: the official repository (cloned; LICENSE.txt, README, demo.py, config read directly), the paper PDF shipped inside that repository, and the Hugging Face weights card. arXiv, the project page and Springer were unreachable from the authoring sandbox, so every quote below comes from the repository copy of the paper. Verdicts live in [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) section T.

## 1. Verdict in one table

| Pipeline stage | Track4World today | Why |
|---|---|---|
| Variant 1 replay, **default (NC) sub-path** — motion/geometry prep for the 4D fit | **YES — strongest candidate, experiment E-06 first** | one forward pass returns camera poses + point maps + dense world-centric 3D trajectories of every pixel; that is exactly the scaffold MoSca/Shape-of-Motion-class fitters reconstruct today from three separate tools [registry T-15, N-6] |
| Variant 1 replay, **clean (commercial) sub-path** | **NO — hard stop** | licence: "only for academic purposes … refrain from using it for any commercial or production purposes under any circumstances", and the scope covers weights, not just code [registry T-3, T-4] |
| Monocular camera front-end (NC track) | **YES, alongside MegaSaM** | poses + metric-scale point maps from the DA3 backbone; best Bonn ATE among compared methods, behind Pi3 on Sintel [registry T-10, T-15] |
| Temporal-stability measurement for the Variant-1 acceptance gate | **YES, as a measuring tool** | dense 3D tracks on input vs. on rendered output give a flicker score that does not depend on held-out cameras [hypothesis \| §5.4; legal question on "tool outputs" recorded in §8] |
| Variant 2a (research-data fine-tune) — motion pseudo-labels | **YES, 2a only** | NC-tainted by both the licence and the DA3 CC-BY-NC weights; 2a artifacts never cross the wall anyway [registry T-3, T-5, N-14] |
| Variant 2b / Variant 3 (commercial weights, dataset) | **NO** | same wall; the *design lessons* (§6) are usable clean-room, the model is not |
| Back-fill of the unseen side | **NO** | it tracks what the camera saw; it does not invent the back — consistent with the product definition [registry N-2] |
| Human-specific quality | **UNPROVEN** | no human dataset in training; on the human-centric PStudio subset it is beaten by V-DPM (paper) and by OpenD4RT (README) [registry T-7, T-8, T-14] |

Short version: **Track4World is the feedforward successor of TrackingWorld (same first author) — the tool v6/6.18 wanted, minus the GPT call, plus a Tencent academic-only licence.** It belongs on the NC side of the wall as a preprocessing and measurement tool, and it changes nothing on the commercial side.

## 2. What it is (verified)

- **Paper.** "Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels", Jiahao Lu, Jiayi Xu, Wenbo Hu, Ruijie Zhu, Chengfeng Zhao, Sai-Kit Yeung, Ying Shan, Yuan Liu — HKUST + ARC Lab, Tencent PCG; ECCV 2026; arXiv 2603.02573 first posted 2026-03-03 [registry T-1, T-2].
- **Task.** "Track4World estimates dense 3D scene flow of every pixel between arbitrary frame pairs from a monocular video in a global feedforward manner, enabling efficient and dense 3D tracking of every pixel in the world-centric coordinate system" [verified | https://github.com/TencentARC/Track4World | as-of 2026-09-17 | "estimates dense 3D scene flow of every pixel between arbitrary frame pairs from a monocular video in a global feedforward manner"].
- **Architecture.** A fine-tuned feedforward geometry encoder (initialised from Depth Anything 3, Pi3 or MoGe) produces camera-centric point maps, camera poses and geometry features; a scene-flow decoder works on sparse anchors at 1/8 resolution, iteratively updates 2D and 3D flow with a "2D-to-3D correlation" that avoids k-NN search in 3D, then recovers dense flow by learned upsampling; pairwise flows are chained into world-centric trajectories [registry T-15 | paper §3].
- **The trick that matters for us.** 3D flow updates are anchored to image-plane correlations, so abundant 2D optical-flow / 2D-tracking datasets supervise the 3D task. The ablation is brutal: training on 3D data only collapses EPE3D from 0.2056 to 0.6511 [registry T-12].
- **Outputs (demo.py).** per-frame `points` and `world_points` (T×H×W×3), `camera_poses`, `traj_3d`, `traj_2d`, visibility/confidence maps, dynamic-object masks, PLY sequences for viewing; metric scale only with the DA3 backbone [registry T-15].
- **Release state.** Code + three checkpoints on Hugging Face (DA3 5.5 GB, Pi3 5.3 GB, MoGe 1.8 GB), ungated, plus a 31.5 GB evaluation bundle; last repository commit 2026-08-24; 320 stars at fetch time [registry T-6, T-16].

## 3. The numbers that matter for our decision

| Quantity | Value | Comparison | Tag |
|---|---|---|---|
| 3D tracking APD, world-centric, 4-benchmark mean | 0.5636 (16-frame) / 0.5323 (50-frame) | V-DPM 0.5142 / 0.4668; SpatialTrackerV2 0.2470 / 0.2335 | [registry T-8] |
| … on PStudio (CMU Panoptic — humans) | 0.5946 / 0.5422 | **V-DPM 0.6084 / 0.5795 is better** | [registry T-8] |
| README WorldTrack cross-check vs OpenD4RT, mean APD | 77.14 vs 74.42 (16f) | on PStudio **78.51 vs 81.42 — OpenD4RT better** | [registry T-14] |
| Dense tracking of a 16-frame ADT clip | 3.4 s, 14 GB VRAM, 26.06M decoder params | STV2 dense = OOM; POMATO dense 4.8 s / 16 GB | [registry T-9] |
| Video point-map Abs Rel, 7-dataset mean | 0.0552 | DA3 0.0743, Pi3 0.0738 (both are its own backbones) | [registry T-10] |
| Camera pose ATE | Bonn 0.009 (best); Sintel 0.119 (Pi3 0.088) | — | [registry T-10] |
| 2D tracking Kinetics AJ | 59.1 | CoTracker3 55.8 | [registry T-11] |
| Training compute | 8× 40 GB GPUs, ~1 week + ~5 days, 2 stages | no human dataset in either stage | [registry T-7] |

Reading for a solo operator: the 16-frame figure is the only published memory point [registry T-9]; a 30 s GoPro take at 60 fps is 1800 frames handled by a sliding window whose peak memory is **unpublished** — measure it, do not assume [hypothesis | E-06 measures VRAM at 5 s / 15 s / 30 s on the RTX 6000 Pro].

## 4. Licence and the wall

- **Governing licence:** "License Term of Track4World" — MIT-shaped grant with one added condition: "You agree to use the Track4World only for academic purposes, and refrain from using it for any commercial or production purposes under any circumstances." [registry T-3]. The definition sweeps in the model: "Track4World refers solely to inference code, parameters, and weights made publicly available by Tencent" [registry T-4].
- **Dependency chain (from the LICENSE.txt attribution list):** Depth Anything 3 nested giant-large weights = CC-BY-NC-4.0 (the default, metric-scale backbone); Pi3 = academic BSD-2, "For commercial use, please contact the authors"; MoGe-vitl = Apache-2.0; AllTracker = MIT; Wan2.1-T2V-1.3B = Apache-2.0 is also listed, oddly, for a tracker [registry T-5]. Even a clean-room re-implementation of the decoder would sit on an NC geometry backbone unless MoGe (Apache-2.0) is chosen — and MoGe is the weakest backbone in the ablation (3D-tracking APD 0.5447 vs 0.5712 with DA3) [registry T-5, T-10].
- **Hugging Face card:** `license: other`; the README badge links to a `LICENSE.txt` that does not exist in the HF repo — the GitHub file is the only licence text [registry T-6].
- **Consequence under LEGAL-LICENSING §2:** Track4World joins the "NC-track experiments only" row. No pose, point map, trajectory, mask or checkpoint produced with it enters a commercial artifact. Whether *measurements* (a stability score) count as artifacts is an open legal question (§8) — until answered, treat scores as internal QA that ships nowhere.
- **Versus its predecessor:** the v6/6.21 audit banned a TrackingWorld swap because it "chains an OpenAI API call → external egress + biometric data" [registry N-11 | 6.21 §21.8]. Track4World has no external API in its inference path (repository grep; the only `requests.post` is an optional Depth-Anything-3 service client) [registry T-15]. That objection is gone; the licence objection replaces it.

## 5. Where it plugs into the pipeline

### 5.1 Variant 1, default sub-path: one pass instead of three tools

Today's zero-training recipe (v6/6.16 Path B, VARIANT-1 §2) prepares a monocular clip with MegaSaM (poses + depth) + a 2D tracker + masks, then hands them to a per-video 4D fitter of the Shape-of-Motion / MoSca class, which *infers* a 3D motion scaffold from those 2D tracks and depths [registry N-6]. Track4World emits the scaffold directly: world-centric 3D trajectories of every pixel plus the poses and point maps that generated them [registry T-15]. Expected effects, all to be measured in E-06:

- Prep wall-clock drops from a chain of per-video optimisations to one feedforward pass [hypothesis | E-06 times both recipes on the same take].
- The 4D fit starts from consistent 3D motion, so the flicker term of the acceptance gate (VARIANT-1 §5.3) should be lower at initialisation [hypothesis | E-06 compares temporal-LPIPS spikes, same fitter, two inits].
- Metric scale from the DA3 backbone removes the manual scale alignment between two GoPro cameras [hypothesis | E-06 checks the recovered inter-camera baseline against the tape-measured rig].

### 5.2 Two GoPro cameras: per-camera runs, then one alignment

The model is monocular; a 2-camera take is two runs. Their world frames differ, so align them with the same rigid transform we already compute for the rig (COLMAP on a static calibration frame) [registry N-5]. Once aligned, the *same physical point tracked from two cameras* gives two independent 3D trajectories — disagreement between them is a direct measure of (a) reconstruction error and (b) residual temporal offset between the cameras. Minimising trajectory disagreement over a time shift is a software estimate of the sub-frame sync residual the owner's genlock project measures with hardware today [hypothesis | GoPro Genlock project: compare the shift-fitted offset with the QR/LED-strobe measurement on the same take; a fast-moving marker makes the fit well-conditioned].

### 5.3 Monocular front-end for the NC track

v6/6.18 chose MegaSaM (NC) and VGGT-1B-Commercial (commercial) for camera poses [registry N-11]. Track4World is a third option on the NC side: its pose accuracy is best on Bonn and mid-pack on Sintel [registry T-10], and it delivers motion in the same pass. Recommendation: run E-06 with MegaSaM as the control; adopt Track4World as the NC-track default only if poses are at least as good on our own takes.

### 5.4 A flicker score that needs no held-out camera

Variant 1 §5.3 defines temporal stability by eyes-on review plus frame-to-frame LPIPS. A stronger, camera-free score: track every pixel on the input video and on the rendered replay from the same viewpoint; the per-pixel 3D trajectory discrepancy (and its temporal spikes) is a stability number that any monocular capture can produce [hypothesis | validate against the held-out-camera PSNR on the first three multicam takes; keep only if the two rank captures the same way].

### 5.5 Variant 2a: motion pseudo-labels, knowledge only

The Wan 2.1 + PRoPE replication (VARIANT-2 §0) needs per-frame geometry and motion targets. Track4World can label the obtainable research mix (PKU-DyMVHumans, Panoptic, HUMBI) with dense 3D trajectories at scale — a cheaper supervision signal than optimisation-based labelling [hypothesis | label one subject, verify against the multicam GT before labelling the mix]. Everything produced this way is 2a knowledge: the licence and the DA3 weights both forbid a commercial descendant [registry T-3, T-5, N-14].

### 5.6 What it does not do

No Gaussians, no appearance, no rendering — it is a front-end, not a 4D reconstructor. No human prior and no human data in training [registry T-7]; the authors name "extreme poses or complex topological changes" as the failure mode [registry T-13] — loose clothing and hair are the topological changes our captures are full of. Long clips run through a sliding window with unpublished memory behaviour [hypothesis | E-06].

## 6. Design lessons we may use clean-room (ideas are not licensed)

1. **2D supervision carries the 3D task.** If we ever train our own motion module (Variant 2), anchor 3D motion updates to image-plane correlations and pour in 2D optical-flow / 2D-tracking data; the ablation says this is the difference between a working model and a collapsed one [registry T-12].
2. **Pairwise arbitrary-frame flow, chained, beats predicting every trajectory.** Memory scales with anchors, not with pixels × frames — STV2-style dense trajectories OOM at 16 frames where this runs in 14 GB [registry T-9].
3. **Backbone-agnostic geometry encoder.** The decoder was trained on three different geometry backbones with modest quality spread [registry T-10]; a commercially clean rebuild could sit on VGGT-1B-Commercial or MoGe (Apache-2.0) rather than DA3 [hypothesis | cost is a training run of the order the paper reports: 8× 40 GB GPUs × ~12 days, registry T-7].
4. **Hybrid unprojection.** Final 3D flow = precise 2D flow + predicted Z displacement, projected with the recovered intrinsics — better than pure 3D regression (EPE3D 0.2056 vs 0.2815) [registry T-12].

## 7. Commercial-clean substitute for the same function

No released model was found that is (a) feedforward, (b) dense 3D tracking in world coordinates and (c) explicitly commercial [not-found-as-of 2026-09-17 | queries: v6/6.18 front-end survey; Track4World README related-work list; OpenD4RT repository licence not fetched]. The nearest clean assembly is VGGT-1B-Commercial (poses, point maps, sparse point tracks; commercial OK, no-military) [registry N-11] + AllTracker (dense 2D tracks, MIT — the same 2D tracker Track4World builds on) [registry T-5] + our own lifting of 2D tracks through the VGGT point maps — i.e. the "2D flow + depth" baseline that Track4World's own ablation shows to be four times worse in EPE3D than the learned hybrid [registry T-12]. Honest reading: clean and cheap exists, clean and *good* does not yet; E-06 quantifies the gap on our footage.

## 8. Open questions (each closes with a written answer, never an assumption)

1. **Licence — do measurement outputs count?** If Track4World only scores a commercial replay and nothing it produced is stored or shipped, is that "commercial or production" use? Counsel question, filed with the LEGAL-LICENSING review [registry T-3].
2. **Memory vs clip length** on the RTX 6000 Pro at `--image_size 640`: 5 s / 15 s / 30 s at 60 fps [hypothesis | E-06].
3. **Human clothing and hair.** Track quality on a subject in a loose garment, measured against the second GoPro [hypothesis | E-06 held-out-camera reprojection of tracked points].
4. **Segmentation dependency.** World-centric visualisation recommends Grounded-SAM-2 masks by text prompt ("person."); confirm whether the *tracking* result changes without masks or only the visualisation does [hypothesis | run 3d_efep with and without the mask step, diff `traj_3d`].
5. **OpenD4RT** (the README's comparison target, better on PStudio) — licence and human performance unknown [registry T-18]; check before E-06 so both run on the same take.
6. **V-DPM and Any4D** — concurrent feedforward methods that beat it on the human subset in the paper's own table [registry T-8]; licences not checked in this pass.

## 9. Sources

- Repository (LICENSE.txt, README, demo.py, config, paper PDF at `assets/arxiv_Track4World.pdf`): https://github.com/TencentARC/Track4World
- Weights and evaluation bundle: https://huggingface.co/TencentARC/Track4World
- Paper index entry: https://huggingface.co/papers/2603.02573 · arXiv: https://arxiv.org/abs/2603.02573 · project page: https://jiah-cloud.github.io/Track4World.github.io/
- Predecessor and context: v6/6.18 (TrackingWorld, front-end survey), v6/6.15 (World Tracing, not the same thing), v6/6.16 (zero-training recipe), VARIANT-1 §2, MVP-0 E-06.
