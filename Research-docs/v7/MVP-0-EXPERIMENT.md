# MVP-0 — the first experiments (cheap, decisive, none run yet)

> Part of Research-docs v7 (2026-07-27). v6 accumulated four cheap experiments that were designed but never executed [registry N-6, N-13]. They are the correct first moves because each one either kills or funds a much larger spend. Order matters; total cost of the whole page ≈ €0–200 [estimate | electricity + hours on the free RTX 6000 Pro; zero cloud required].

## E-01 — Layer-0 end-to-end replay (funds/kills Variant 1)

One subject, one take, both GoPro HERO 13 (QR-resync) → poses → gsplat-based per-video 4D fit → export PLY-sequence → view in SuperSplat + a first `Load4DGS/Play4DGS` node stub in ComfyUI over the native GAUSSIAN type [registry LB2-8].
- **Measures:** wall-clock per stage, VRAM peak, held-out-camera PSNR (fit on 1 cam + turntable, eval on the other), visible flicker.
- **Gate:** produces *any* faithful playable clip → Variant 1 is real; proceed to the 10-capture stability gauntlet (VARIANT-1 §5).
- **sm_120 note:** this experiment doubles as the build-viability probe — if gsplat CUDA kernels need patching beyond 2 days, invoke the cloud fallback line [registry C3-8].

## E-02 — M0.5 PRoPE overfit pre-gate (funds/kills Variant 2)

From 6.14: overfit the Wan 2.1 1.3B + PRoPE architecture on ONE subject's multicam clip until it memorizes it [registry N-7].
- **Gate:** if the architecture cannot even overfit a single subject on the 96 GB card [registry C3-8], the Flex4DHuman replication premise is broken → do not spend E1's money (VARIANT-2 §4 fallback applies).
- Cost: hours [estimate | single-subject overfit, no dataset assembly].

## E-03 — ArtiFixer zero-shot back-fill probe (half a day)

Designed in 6.23 §23.9.4, never run [registry N-13]. Feed a Layer-0 front-faithful reconstruction to ArtiFixer's checkpoint and see whether post-hoc completion is even directionally useful on humans (its domain is static scenes).
- **License rail:** checkpoint is NVIDIA OneWay Noncommercial → NC-track experiment only; results inform architecture choices, never ship [registry N-13].
- **Gate:** if zero-shot completion is embarrassing on humans, drop the "post-hoc fixer" idea from all roadmaps and let Variant 2's trained prior carry back-fill alone.

## E-04 — MV-Performer license email (15 minutes, async)

The strongest zero-training multiview candidate has NO license file [registry N-10]. Email the authors asking for one. Costs nothing, may unlock a better Layer-0 component; until answered, the repo stays unusable even for demos-with-attribution.

## E-05 — PKU-DyMVHumans license resolution (gate M1 of the v3 prompt)

The HF tag says `c-uda`, the site implies click-through research terms [registry N-14]. Resolve in writing before any 2a fine-tune ingests it — this is the datum that decides whether 2a's main course exists or the mix falls back to Panoptic+HUMBI+scans.

## E-06 — Affine-rectified depth prior probe (funds/kills the geometry-prior idea; needs the E-01 capture)

Assessment: [AFFINEGS-GEOMETRY-PRIOR.md](AFFINEGS-GEOMETRY-PRIOR.md). The AffineGS supervision is **ported as a loss** into the gsplat-based per-video fit of E-01 — the upstream repository is never run as a pipeline stage [registry H-4, H-7].
- **Setup:** one E-01 take, three static frames (subject holding still), E-01's camera split (fit on one camera plus turntable, evaluate on the other) — no extra hardware. Three seeds per arm [hypothesis | enough to see the ±0.3 dB band of VARIANT-1 §5.4].
- **Arms:** (0) no depth prior; (1) one global scale-and-shift alignment; (2) spatially varying affine rectification (the three terms).
- **Priors:** Depth-Anything-V2-Large as an NC-track diagnostic only — it measures the ceiling [registry H-8]; Depth-Anything-V2-Small and MoGe-2 as the clean candidates [registry H-9]. The large-versus-clean gap is the price of commercial cleanliness.
- **Init ablation:** SfM-cloud start versus a denser multi-view start, because two cameras give exactly the sparse cloud the paper says breaks competing methods [registry H-14].
- **Measures:** held-out PSNR and LPIPS, visible floaters, and the number of per-take hand retunes an arm needs (guided-filter radius and epsilon, loss weights) [registry H-18].
- **Gate (keep):** arm 2 beats arm 1 by more than 0.3 dB held-out PSNR on at least two of the three frames, with no hand retune, using a clean prior → integrate into Variant 1's clean sub-path [hypothesis | threshold borrowed from VARIANT-1 §5.4; re-set after the first run]. Otherwise drop the idea and say so in the result note.
- **Temporal probe (half a day):** a short consecutive run of frames, per-image prior versus Video-Depth-Anything-Small [registry H-9]; frame-to-frame LPIPS spikes as defined in VARIANT-1 §5.3. A prior that flickers rules arm 2 out of any 4D use, whatever its static score.
- **Port check first:** reproduce one MuSHRoom scene with the ported loss against the authors' published numbers (hours on a 3090-class GPU [registry H-16]) before trusting any own-capture result; the no-prior arm is the negative control and must NOT reproduce them.
- **Cost:** hours on the free RTX 6000 Pro, or a few cloud hours if sm_120 blocks the port [estimate | registry C3-8, H-7].

## E-07 — Rig front-end bake-off: COLMAP vs VGGT vs Solv3R (NC-track diagnostic; needs the E-01 capture)

Assessment: [SOLV3R-MULTIVIEW-SOLVER.md](SOLV3R-MULTIVIEW-SOLVER.md). The question is narrow: does a learned multi-view solver hold where classical SfM fails on our own studio takes?
- **Setup:** the two cameras of the E-01 take, plus a calibration-board shot of the same session from the same tripod positions. The board gives the reference; compare relative rotation and translation direction only (scale-free, since the solvers do not return metric scale).
- **Arms:** COLMAP; VGGT — the commercial-licensed checkpoint if its licence text clears, otherwise the research checkpoint as a diagnostic [registry N-11]; Solv3R weights as an NC-track diagnostic only [registry I-6, I-7].
- **Scenes:** one feature-rich background and one deliberately feature-poor one (plain studio wall, no green screen) — the case the hypothesis is about [hypothesis | feature-based SfM degrades on plain walls; E-07 measures whether it does on ours].
- **Measures:** relative-rotation and translation-direction error in degrees against the board reference, per arm and scene, plus the count of failed solves.
- **Camera-bump test:** after the first solve, move one camera by hand and re-solve from the previous state versus from scratch — the one property of the paper that maps onto a real session problem [registry I-2].
- **Also record:** the density of per-view depth and rays at the solver's 504-pixel input, as input to the E-06 init ablation [registry I-4].
- **Gate:** Solv3R earns further effort only if it succeeds where COLMAP fails on the feature-poor scene. If COLMAP and VGGT both succeed, drop it from the roadmap and watch for a permissively licensed solver instead [hypothesis | decision rule fixed before the run].
- **Licence rail:** nothing produced by the NC arms enters a commercial artifact [registry N-12, I-7].
- **Cost:** hours on the free RTX 6000 Pro for a 117M-parameter model [registry I-4]; sm_120 compatibility is untested, with the cloud fallback ready [registry C3-8].

## Reporting rule

Each experiment ends with a one-page result note added to v7 (tagged claims, linter-clean) and a row update in DATASET-STATUS.md §1. No result note → the experiment did not happen — this page is the antidote to the repo's historical pattern of plans quietly becoming "facts".
