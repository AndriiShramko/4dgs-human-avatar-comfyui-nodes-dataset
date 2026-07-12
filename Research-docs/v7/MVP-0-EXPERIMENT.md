# MVP-0 — the first experiments (cheap, decisive, none run yet)

> Part of Research-docs v7 (2026-07-12). v6 accumulated four cheap experiments that were designed but never executed [registry N-6, N-13]. They are the correct first moves because each one either kills or funds a much larger spend. Order matters; total cost of the whole page ≈ €0–200 [estimate | electricity + hours on the free RTX 6000 Pro; zero cloud required].

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

## Reporting rule

Each experiment ends with a one-page result note added to v7 (tagged claims, linter-clean) and a row update in DATASET-STATUS.md §1. No result note → the experiment did not happen — this page is the antidote to the repo's historical pattern of plans quietly becoming "facts".
