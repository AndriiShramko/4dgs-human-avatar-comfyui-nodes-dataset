# LEGAL-LICENSING — the license map and the wall between tracks

> **DRAFT — not legal advice, pending counsel review.** Part of Research-docs v7 (2026-07-12). Licenses change: every entry carries an as-of date; re-check the primary source before relying on it. Deeper analysis inherited from v6/6.7 (whose current text survived audit) and 6.22.

## 1. The license map (entries verified against primary sources on 2026-07-12, except rows explicitly tagged inherited)

| Component | License | Commercial track? | Evidence |
|---|---|---|---|
| gsplat rasterizer | Apache-2.0 | ✅ the backbone of every commercial path | [registry C4-7] |
| Wan 2.1 (code + weights) | Apache-2.0 | ✅ | [registry LB2-11] |
| MHR body prior (Meta) | Apache-2.0 | ✅ | [registry C4-5] |
| Anny body prior (NAVER) | Apache-2.0; its optional smplx-topology mode is NC-only | ✅ native mode only | [registry C4-6, LB2-10b] |
| ComfyUI native GAUSSIAN nodes (core) | part of ComfyUI (GPL-3.0 ecosystem) | ✅ as runtime; see §3 for node licensing | [registry LB2-8] |
| SMPL-X model | research-only; commercial via Meshcapade | ❌ out of commercial builds | [registry LB2-1, C4-3] |
| Inria diff-gaussian-rasterization + forks (STG thirdparty, 4DGaussians submodule, DualGS inherited parts, QUEEN) | non-commercial | ❌ research/reference only | [registry LB2-4, LB2-5, N-12] |
| Apple SHARP | code: permissive custom; weights: research-only | ❌ weights; code irrelevant to this product | [registry LB2-2b, LB2-2c] |
| Difix3D+ / ArtiFixer / World Tracing / Sapiens | NVIDIA License / NC checkpoints / CC BY-NC(-ND) | ❌ NC-track experiments only | [registry LB1-8b, N-13] |
| Academic 4D-human datasets (ActorsHQ, DNA-R, MVHumanNet, THuman, HuMMan, HuGe100K) | all non-commercial or gated | ❌ — and DNA-R/ActorsHQ are owner-banned outright | [registry LB2-12a..LB2-12f, N-3] |
| VolHuMe (2026) | dataset license NOT PUBLISHED | ❓ treat as NC until authors answer | [registry LB2-6] |
| $0-license synthetic stack (MakeHuman, Poly Haven, Rocketbox, CMU mocap) | CC0 / MIT / commercial-OK | ✅ | [registry N-9 \| spot-check each before first commercial use] |
| Rumored prices: Inria "~80K EUR", Meshcapade "~150K EUR/yr" | — | order-of-magnitude only | [registry C4-1, C4-2 \| RUMOR — no public primary source; get written quotes] |

**Two standing detection rules** (they caught every trap above): (1) check the *vendored rasterizer* first — a repo's root MIT does not relicense its `thirdparty/` Inria code (SpacetimeGaussians, 4C4D) [registry N-12]; (2) check CODE and WEIGHTS licenses separately — they diverge (Apple SHARP, ArtiFixer, Lyra) [registry LB2-2b, LB2-2c, N-13].

## 2. The wall between tracks

- Variant 1 default sub-path and Variant 2a may use any research tool.
- **Artifacts produced by NC tools or trained on NC data never cross into commercial tracks** — no scans, no checkpoints, no LoRAs, no distilled students [registry N-14 \| weights inherit training-data restrictions]. Firewall mechanics inherited from v6/6.12 §12.3.
- Every checkpoint stored in this project carries a provenance field: `training data → commercial status`.

## 3. ComfyUI nodes and GPL — the business model consequence

Local custom nodes import ComfyUI (GPL-3.0 [registry LB2-14]), making them derivative works: **closed-source paid local nodes are not viable**; a paid node marketplace was not found as of 2026-07 [registry LB2-14]. Weights, datasets and remote APIs are NOT code derivatives — GPL does not reach them. Therefore:

- Nodes = GPL-3.0, free, the distribution funnel (and the niche is currently empty [registry LB2-7]).
- Monetization = dataset licensing, model weights, hosted API. Heavy proprietary logic can live behind a remote API outside GPL scope.
- Node package stays a ~100-line adapter over a standalone library (our license) → survives ComfyUI churn and enables CLI/Blender re-targeting.

## 4. The GDPR / likeness package (blocker before the FIRST commercial capture day)

A photorealistic human scan is personal data; where identifiable, biometric-class data under GDPR Art. 9 → explicit consent, separate from the commercial release. Polish law adds written consent for likeness distribution (art. 81 ustawy o prawie autorskim). The package v7 requires:

1. **Two documents per subject:** commercial model release (sublicensing, derivatives, AI-training explicitly named) + GDPR Art. 9 explicit consent. Templates drafted in-house, one counsel review [estimate | €1-2K PL counsel, budgeted in VARIANT-2 §2].
2. **DPIA** before the first commercial capture.
3. **Dataset versioning + recall mechanism:** consent withdrawal must propagate to sold copies — buyer EULA obliges applying recall updates.
4. **Buyer EULA:** bans re-identification, deepfake use, resale; includes recall clause; EU AI Act transparency noted for generative uses. EULA draft is a Variant-2b deliverable so pre-sales start with paper.
5. No minors. No captures before releases are signed — retrofitting consent does not work.

Competitive note: HumanDataset markets GDPR Art. 6(1)/9(2) compliance as a selling point [registry C4-10b] — confirming that "clean + documented" is the differentiator worth paying for.

## 5. Patents (unchanged from v6, verified)

Clean-room protects against copyright, not patents — independent invention is not a defense [registry C4-9]. SMPL patent US10395411B2 (Max Planck) exists [registry C4-4]; avoiding the SMPL family entirely (MHR/Anny path) also reduces exposure. FTO search remains a Variant-3 line item before productization; the Anthropic settlement (preliminary approval; cite carefully) illustrates training-data liability scale [registry C4-8].
