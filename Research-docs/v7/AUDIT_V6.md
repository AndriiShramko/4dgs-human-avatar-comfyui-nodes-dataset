# AUDIT_V6 — what v5/v6 got wrong, what they got right, and what v7 does about it

> Status: part of Research-docs **v7** (2026-07-12). Every verdict referenced here is backed by a row in [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) with a primary-source URL, an as-of date and a verbatim quote. v6's own verdicts were treated as hypotheses, not evidence, and re-proven in both directions.

## 1. The headline result: the failure is not where we expected

The v7 planning pass suspected v6 of citing phantom papers. **Web verification refuted that suspicion**: GIFSplat, CUDA-APML, CloDS, VF-NODE and Squisher all exist (registry LB1-1..LB1-5). The real failure modes are subtler and live in two different layers:

**Layer 1 — the research docs (Research-docs/v5, v6).** Failure mode: *distortion in transfer* — a real paper gets a wrong acronym expansion [registry LB1-2n], a stage-time becomes a pipeline-time [registry C2-4]: HumanSplat "0.3s" is one stage of a ≈9.3s pipeline), a training speedup becomes a runtime argument [registry LB1-4], an ablation number is off by 2× (C2-2: "+0.06 LPIPS" vs the paper's ≈+0.13), LLM-inference metrics get pasted into a geometry-training argument [registry C3-3: "KV cache" does not exist in a feed-forward regressor]. Notably, **v6's own audit culture is real**: its 2026-06-12 and 2026-07-11 passes retracted "$40K savings", fixed "B200 Ultra", fixed MEGA 161→20, fixed the ÷190 double-count, and correctly downgraded the "80K EUR"/"150K EUR" prices to rumors [registry C3-5, C3-2, C2-7, C2-8, C4-1, C4-2]. v7 inherits those corrections instead of re-litigating them.

**Layer 2 — the facade docs (root README, dataset/, parts of 6.0-overview).** Failure mode: *verb tense* — aspirations stated as existing assets. "Petabytes" vs "hundreds of TB" in the same repo [registry C1-1]; an enterprise dataset "delivered via S3/CDN" while the repo itself admits "There is no downloadable build or dataset yet" [registry C1-2]; a "proprietary 90-camera rig, hardware-synchronized" as *training data* while 6.10 says the rig "is not yet built" and 6.21 documents the actual kit as 2× GoPro HERO 13 with "True genlock ❌ not on GoPro" [registry C1-3]. The public README still carries a comparison table with a fabricated Apple SHARP figure — "~100 GB per minute" has **no traceable source** [registry LB2-2d] — and claims ("Absolute temporal stability", "100% legal for commercial use", "Accurate back-side reconstruction") that v6's own red-team explicitly banned in 6.8 §8.5.2. This is the single most damaging item in the repo and is fixed together with this v7 release.

## 2. Scorecard by original suspicion

| Original suspicion (v7 planning seed) | Outcome after verification |
|---|---|
| "GIFSplat / CUDA-APML / CloDS are probably hallucinated" | REFUTED — all exist (LB1-1, LB1-2, LB1-3); distortions are in names/scope, not existence |
| "SMPL-X = CC-BY-4.0 is a legal error" | CONFIRMED as a bare claim — model license is research-only [registry C4-3, LB2-1]; current 6.7 already carries the correct two-license wording |
| "13M Gaussians per avatar is 1-2 orders too high" | CONFIRMED as an avatar figure — published avatar counts are ~0.1-2.5M (C1-5); v6 had already softened to a "1-13M" range |
| "SHARP comparison table is fabricated" | CONFIRMED — the "~100 GB/min" figure is untraceable [registry LB2-2d]; SHARP is single-image → *static* 3DGS [registry LB2-2a], so the comparison axis itself is wrong; SHARP weights are research-only, code license is permissive [registry LB2-2c, LB2-2b] |
| "The <3 seconds avatar claim is a partial sum" | CONFIRMED — only 2 of ~6 inference stages are timed (C1-4) |
| "80K/150K EUR license prices are facts" | CONFIRMED as rumors — no public primary source for either (C4-1, C4-2) |

## 3. What v7 inherits from v6 unchanged (the healthy core)

- **Owner decisions (binding, registry N-1..N-5):** no purchased datasets; commercial track = own rig data only; the product is *faithful 4DGS of real video*, not avatar-from-photo (LHM/IDOL out of the core); DNA-Rendering and ActorsHQ permanently out; solo-realizable on one RTX 6000 Pro; actual kit = 2× GoPro HERO 13.
- **Stack decisions that survived audit:** gsplat as the only commercially clean rasterizer; MHR/Anny as free body priors with Anny's smplx-mode NC exception; H100+FP8 over B200+NVFP4 for training economics; composite rendering loss without Chamfer/Hungarian in the training loop; tiered motion; clean-room with the honest "does not protect against patents" caveat [registry C4-7, C4-5, C4-6, C3-5, C3-7, C2-1, C4-9].
- **The document set 6.16→6.24**: Layer-0 zero-training pipeline, obtainable-data plan, the canonical v3 goal prompt (6.21 §21.9, registry N-6), and the 6.24 budget reconciliation.

## 4. What v7 changes

1. **One apex document set.** 6.19 (synthesis), 6.20/6.21 (capture+prompt), 6.24 (budget) partially overlap. v7 declares itself the apex plan: the three variants in [VARIANT-1](VARIANT-1-BUDGET.md) / [VARIANT-2](VARIANT-2-SCALING.md) / [VARIANT-3](VARIANT-3-FLAGSHIP.md) supersede the *planning* role of those files; they remain authoritative for their implementation detail (6.21 §21.9 stays the canonical execution prompt for Variant 2a; 6.14 stays the agent-team mechanics).
2. **Facade brought down to research truth.** Root README rewritten in this release: the SHARP table is replaced by target characteristics derived from verified baselines; dataset docs get honest status via [DATASET-STATUS.md](DATASET-STATUS.md).
3. **Every claim tagged.** The registry taxonomy + `tools/claims_linter.py` (URL-liveness included) become the gate for any future doc added to v7.
4. **Deprecation notices.** v5 files are marked as unverified AI-generated prompt/essay archives (kept for provenance); v6 files get a pointer notice: audited by v7, claim statuses live in the registry, owner decisions remain binding.

## 5. Standing corrections that must not regress

- Never cite "~100 GB/min" as an Apple SHARP figure [registry LB2-2d]. Real compression baselines: QUEEN ≈0.7 MB/frame and DualGS ≈350 KB/frame — both non-commercial code paths [registry LB2-4, LB2-5], usable as *benchmarks*, not as components.
- Never present the 90-camera rig, enterprise dataset volumes, or "petabytes" as existing (C1-1..C1-3).
- Never argue FP8 for this workload via KV-cache/LLM numbers (C3-3).
- Keep prices "80K EUR" / "150K EUR/yr" tagged as rumors until a written quote exists (C4-1, C4-2).
