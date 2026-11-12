# Solv3R — multi-view solver assessment for the capture → 4DGS pipeline

> Part of Research-docs v7. Research pass 2026-11-12, ten days after the paper appeared on arXiv and one day after the last push to its code repository [registry I-1, I-9]. Read first-hand: the arXiv HTML, the official repository README and GitHub licence metadata, the weights licence text, and the parent project's README. Verdicts live in [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) section I. **Solv3R outputs cameras, depth maps and ray maps — no appearance, no Gaussians, no motion model.** It is an upstream stage for our pipeline, not an avatar or 4D method, and its weights are non-commercial.

## 1. Verdict in one table

| Possible use | Verdict | Why |
|---|---|---|
| Use the Solv3R weights anywhere on a commercial track | **NO** | NVIDIA License: research or evaluation use only, derivative works included [registry I-6, I-7] |
| NC-track front-end for the 2-camera rig: relative pose, per-view depth, dense start for the fit | **YES, as experiment E-07** | multi-view-consistent output that a single-image depth prior cannot give [registry I-11, I-12] |
| Make it the permanent rig-calibration stage | **NO** | a fixed rig is calibrated once per session; the paper's novelty (refine, extend, cache) targets moving or growing image sets [registry I-2, I-3] |
| Rig growth (more cameras) or handheld monocular capture, later | **WATCH** | tested up to 64 views at 504-pixel input; needs a permissively licensed solver first [registry I-4, I-14] |
| Retrain a clean version ourselves | **NO at solo scale** | starts from an NC parent checkpoint, 25-dataset mixture, no training code in the release [registry I-4, I-5, I-10] |
| Treat it as a human or 4D method | **N/A** | people appear only as moving content in two streaming benchmarks [registry I-13] |

## 2. What it is (vendor-verified)

- Paper: Solv3R, arXiv 2611.02857; Kwak, Tagliasacchi, Kagami, Ono, Yi at UBC, Wayve Labs, SFU, Toronto and Sony [registry I-1].
- Idea: a looped transformer "that learns to continue solving multi-view geometry from its current state", with an update operator independent of the iteration count — so views at different refinement stages can be updated together, new views can join, and solved views can be cached once as key-value context [registry I-2].
- One model for four settings: offline reconstruction, incremental reconstruction, relocalization and streaming, with no task-specific module or training [registry I-3].
- Built by fine-tuning NVIDIA's Déjà View checkpoint; 117M parameters; input images are 504 pixels on the long side [registry I-4].

## 3. What the evidence supports — and what it does not

- **Compact:** against VGGT's 1.26B parameters the paper lists 117M, about 10.8× fewer, and 76.5 against 190.0 TFLOPs for 24 views [registry I-11].
- **Accuracy is mixed, not dominant:** on DTU pose AUC@3 it scores 84.5 against VGGT's 96.5, on ETH3D 65.4 against 35.4 [registry I-11].
- **Streaming:** 95 ms per frame and 5.0 GB peak memory, against 52 ms and 6.1 GB for CUT3R and for TTT3R; Solv3R's numbers are better than both on every pose, depth and reconstruction column of that table [registry I-12 \| estimate: column-by-column comparison of the quoted rows]. That is roughly 10 frames per second [estimate \| 1000 ms / 95 ms], below the rate of a 30 fps clip.
- **Dynamic content appears only as a benchmark label:** the streaming set includes Sintel, TUM dynamics and Bonn, scored on camera pose and depth; no human is reconstructed, tracked or deformed [registry I-13].
- **Scale limits:** training draws 4–64 views per step and the incremental test grows from 8 to 64 views; beyond that is untested — relevant to a 90-camera rig [registry I-14].
- **Resolution:** a 504-pixel input means GoPro frames are downsampled roughly 8× for the solver [estimate | 4K long side 3840 px over 504 px]; depth maps are coarse, poses are the useful output.

## 4. Licensing — the second Apache-code, NC-weights case in two days

Same pattern as the AffineGS depth prior [AFFINEGS-GEOMETRY-PRIOR.md §4 \| registry H-8]: LEGAL-LICENSING §1 detection rule 2 again.

| Component | License | Commercial track? | Evidence |
|---|---|---|---|
| Solv3R code | Apache-2.0 | ✅ | [registry I-6] |
| Solv3R weights (fine-tuned from the Déjà View checkpoint) | NVIDIA License: use "non-commercially", meaning research or evaluation only; derivative works included | ❌ | [registry I-6, I-7] |
| Parent Déjà View code | "mostly" Apache-2.0; VGGT-derived files under a separate VGGT License | ⚠ vendored-licence check before reuse | [registry I-8] |
| Parent Déjà View weights | NVIDIA License | ❌ | [registry I-8] |
| The paper's text | CC BY 4.0 — attribution only, unlike AffineGS's NC-SA text | ✅ | [registry I-1] |
| Third-party demo that turns solves into splats | Apache-2.0 demo code, but it pulls Apple SHARP weights, research-only | ❌ on that path | [registry I-15 \| LB2-2c] |

**Policy:** Solv3R weights are an NC-track diagnostic, exactly like MegaSaM [registry N-11]. Nothing produced with them — poses, depth maps, point clouds — enters the commercial dataset or a commercial checkpoint [LEGAL-LICENSING §2 \| registry N-12]. The commercial camera front-end candidate remains VGGT-1B-Commercial, whose licence text is still unread [registry N-11]. Whether a Solv3R-derived *pass/fail* calibration check counts as "evaluation" under the licence is a legal question [hypothesis | add to the single counsel review of LEGAL-LICENSING §4].

## 5. Where it could help us (hypotheses, ranked)

1. **Front-end bake-off on feature-poor studio takes.** Plain studio walls without a green screen are the classic failure case for feature-based SfM [hypothesis | E-07]; a learned solver may hold where COLMAP does not. The reference is cheap: a calibration board gives the true relative pose of the two GoPros.
2. **Dense, multi-view-consistent start for the sparse-view fit.** AffineGS-style supervision seeds its Gaussians from sensor depth, which two cameras cannot supply [registry H-14]; per-view depth and rays from a multi-view solver are a candidate substitute [hypothesis | E-06 init ablation, NC-track only].
3. **Calibration QA gate for later multi-rig sessions:** compare the production calibration with an independent solve and flag a bumped tripod. Output is a number, not an artifact — but see the counsel question in §4 [hypothesis | after E-07].
4. **Rig growth and handheld monocular capture:** the genuinely new capability, and the reason to keep watching. Not usable until a permissively licensed solver exists [registry I-4, I-14].

## 6. What it cannot do for us

- No appearance, no Gaussians, no deformation: a per-frame depth map of a moving person is not a 4D reconstruction [registry I-13].
- The fixed-rig case is the one where classical calibration is already strong; the paper's headline advantages are not our bottleneck today [hypothesis | E-07 decides].
- Clean retraining is out of reach for one person: the model starts from an NC checkpoint and trains on a mixture of 25 datasets whose individual terms we have not read [registry I-4, I-5].
- No training code is listed in the release [registry I-10].

## 7. Next step

Experiment **E-07** in [MVP-0-EXPERIMENT.md](MVP-0-EXPERIMENT.md): board-referenced pose comparison of COLMAP, VGGT and Solv3R on the E-01 take. Until it reports, no README, landing or dataset text may mention Solv3R-derived geometry [LANDING-SYNC.md \| registry N-6].
