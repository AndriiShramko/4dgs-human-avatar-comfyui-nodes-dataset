# VARIANT 1 — Budget track (≤ €1000): maximally stable 4DGS replay

> Part of Research-docs v7 (2026-07-12). Inherits and supersedes the *planning* role of 6.16 (zero-training MVP) and 6.19 Layer 0; those files stay authoritative for tool-level detail. Commercial cleanliness is NOT required on this track (owner's framing), but a clean sub-path is defined so nothing produced here poisons the commercial tracks.

## 1. What this variant is — and is deliberately not

**It is:** a per-capture optimization ("replay") pipeline — a real captured performance becomes a faithful 4D Gaussian-splat clip, playable in ComfyUI and on the web. No model training, no learned prior. This matches the owner's product definition: faithful 4DGS of REAL video, not avatar-from-photo [registry N-2].

**It is not:** a drivable/riggable avatar, and not a 360° promise from monocular input. Physics of observation, stated honestly [registry N-2, C1-3]:
- **Multi-camera capture (2 cams today — the owner's actual kit [registry N-5]; more as further rigs are brought into this project): faithful coverage** of what the cameras saw.
- **Monocular capture: faithful FRONT only.** The unseen back is either absent or visibly hallucinated — this variant does not hallucinate; back-fill belongs to Variant 2's learned prior.

Stability is the acceptance criterion, so replay-not-generative is a design decision, not a budget compromise: per-video optimization has no domain gap and no cherry-picked generalization to fail on.

## 2. Pipeline (tools verified for existence/licensing in the registry)

| Stage | Tool (primary / fallback) | Notes |
|---|---|---|
| Capture | 2× GoPro HERO 13, QR-timecode resync [registry N-5] | true genlock does not exist on GoPro [registry C1-3]; document residual sub-frame offset per take [estimate \| owner's gopro-genlock project measured σ 0.3-0.5 ms class offsets with QR+rolling-shutter methods] |
| Camera poses | COLMAP (multicam) / MegaSaM (monocular; NC-track only) [registry N-11] | VGGT-1B-Commercial is the commercial monocular front-end candidate — license text must be read first [registry N-11] |
| 4D fit | gsplat-based per-video optimization (Apache-2.0) [registry C4-7]; method refs: Instant4D / Shape-of-Motion / MoSca class [registry N-6 \| tool list inherited from 6.16] | STG/4DGaussians usable on the NC sub-path only — their rasterizer inherits Inria non-commercial [registry LB2-5, N-12] |
| ComfyUI delivery | Wrapper nodes over the native `GAUSSIAN` type (core since v0.23.0) [registry LB2-8] | see §3 — the time axis is ours to design; niche is open [registry LB2-7] |
| Web delivery | PLY-sequence → SOG/SPZ static frames [registry LB2-8] | no industry temporal format exists yet — an opportunity, not a blocker [registry LB2-7] |

## 3. Time over a static GAUSSIAN type — the load-bearing design question

ComfyUI v0.23.0 ships a *static* `GAUSSIAN` type; nothing temporal exists in the registry, and the only GitHub adjacency is a GPL-3.0 per-frame orchestration wrapper [registry LB2-7, LB2-8]. Variant 1 must pick one of:

- **(a) GAUSSIAN-sequence (chosen default):** a `GAUSSIAN_SEQUENCE` list-of-frames type + `Load4DGS` / `ExtractFrame → GAUSSIAN` / `Play4DGS` / `Export PLY-seq` nodes. Pros: zero coupling to core internals, every existing GAUSSIAN node works on any extracted frame, survives ComfyUI churn. Cons: memory-naive (per-frame copies) — acceptable at replay scale [estimate \| 5-30s clips × 0.1-2M splats, per C1-5 avatar counts].
- **(b) Native temporal type PR into core:** higher payoff, higher coupling; revisit only after (a) ships and if Comfy core signals temporal interest.

Decision (a) is recorded here as the v7 default; the node package stays a thin adapter over a standalone library so a future core temporal type is a migration, not a rewrite.

## 4. Budget (target ≤ €1000 total)

| Line | Amount | Basis |
|---|---|---|
| Storage (workspace + archive HDD) | €250–450 | [estimate \| one multicam subject ≈ 300-400 GB raw at 100-cam scale; at 2-8 cams today €250 covers 20+ TB] |
| Electricity, local RTX 6000 Pro runs | €50–150 | [estimate \| 0.8-1 kW under load × 200-500 h × PL tariff ~€0.25/kWh] |
| Cloud fallback when sm_120 fights back | €150–350 | [estimate \| H100 at $2-3/hr, 50-120 h; RunPod-class pricing verified in registry C3-4] |
| **Total** | **€450–950** | fits ≤ €1000; public-demo hosting would add €100-600 [estimate \| 6.12 ranged $700-1600 incl. hosting] |

The free RTX 6000 Pro (96 GB) carries the main compute [registry C3-8]. The sm_120 tax is real: stock toolchains emit sm_90 kernels; budget +1–2 weeks of build time and treat the cloud line as the escape hatch [registry C3-8].

## 5. Acceptance = "maximally stable result", made measurable

1. **Pipeline stability:** ≥90% of new captures pass end-to-end with zero manual interventions, demonstrated on 10 consecutive captures [hypothesis | run the 10-capture gauntlet after the pipeline freezes].
2. **Quality on held-out cameras** (multicam takes; 1–2 cams excluded from the fit): PSNR ≥ 28 dB / SSIM ≥ 0.92 / LPIPS ≤ 0.10 [hypothesis | thresholds inherited from STG-class literature as starting points, to be re-baselined on the first 3 own captures].
3. **Temporal stability:** no visible gaussian flicker in an eyes-on review, plus frame-to-frame LPIPS with no outlier spikes [hypothesis | define the spike threshold from the first stable capture].
4. **Reproducibility:** 3 runs, same seed → PSNR within ±0.3 dB [hypothesis | catches nondeterministic build breakage on sm_120].

## 6. Commercial vs non-commercial sub-paths

| | Default sub-path (research tools, fastest) | Clean sub-path (commerce-ready) |
|---|---|---|
| 4D fit | any tool incl. STG/Inria-derived [registry LB2-5] | **gsplat-only stack** [registry C4-7, N-12] |
| Monocular poses | MegaSaM [registry N-11] | VGGT-1B-Commercial after reading its license [registry N-11] |
| Artifacts | free to publish as demos | only artifacts produced by the clean stack cross into commercial use — **the wall rule**: outputs of NC tools never migrate [registry N-12 \| same principle that catches 4C4D's MIT-over-NC-deps trap] |
| Cost delta | baseline | +2–4 weeks porting effort [estimate \| replacing Inria-derived rasterizer calls with gsplat APIs] |

## 7. Kill / exit criteria

- Acceptance §5 not reached in 3 months [estimate | planning window, owner-adjustable] → fix capture/calibration; do NOT escalate to training — a learned prior multiplies bad data, it does not repair it.
- Success + external interest in back-fill/generative quality → open Variant 2. Layer-0 artifacts (calibrated captures, held-out eval protocol) are exactly Variant 2's prerequisites, so nothing is wasted.
