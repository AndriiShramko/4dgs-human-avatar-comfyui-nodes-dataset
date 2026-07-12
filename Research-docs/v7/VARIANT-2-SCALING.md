# VARIANT 2 — The money→quality curve: does paying for training buy a better result?

> Part of Research-docs v7 (2026-07-12). Purpose: answer ONE question with controlled experiments — *does each additional dollar of fine-tuning measurably improve faithful-4DGS quality (above all: back-side fill and multi-view consistency), and where does the curve flatten?* Execution detail lives in 6.21 §21.9 (the canonical v3 prompt) and 6.14 (agent-team mechanics); budgets reconcile with 6.24. This document is the decision layer on top of them.

## 0. The pivot recorded

Earlier planning drafted this variant around LHM fine-tuning. That is dead on arrival against the owner correction: the product is faithful 4DGS of REAL video, and single-image→avatar models (LHM, IDOL, AniGS, PSHuman) are the wrong tool for the core product [registry N-2]. Variant 2 therefore fine-tunes the **video→4D path: Flex4DHuman-style replication — Wan 2.1 1.3B backbone + PRoPE positional encoding** [registry N-7], which is exactly what 6.13/6.21 already engineered. Wan 2.1 code and weights are Apache-2.0 [registry LB2-11], so the backbone itself is commercially clean; what taints or cleans the result is the *data* — hence the two sub-tracks.

## 1. Sub-track 2a — research data → knowledge (never a product)

Fine-tune on obtainable public human datasets (PKU-DyMVHumans as the DNA-Rendering replacement, CMU Panoptic, HUMBI, THuman2.1 scans; AIST++ contributes annotations only — zero pixels) [registry N-14, N-3]. Nearly all such data is non-commercial or license-ambiguous, so **everything 2a produces — weights, LoRAs, distilled students — is knowledge, not an asset** [registry N-14 \| plus the standing rule: weights fine-tuned on NC data inherit NC]. Firewall rules from 6.12 §12.3 apply: 2a artifacts never cross into commercial tracks.

**Experiment ladder (each rung = one point on the curve):**

| Rung | Data scale | Compute | Cost | What it answers |
|---|---|---|---|---|
| E0 | 0 (baseline) | Layer-0 replay + pretrained Wan inference | €0 [estimate \| runs on the free RTX 6000 Pro] | the floor: how bad is zero-training back-fill |
| M0.5 | 1 subject | PRoPE overfit pre-gate from 6.14 | ~€0-50 [estimate \| hours, not days] | does the architecture learn AT ALL before money is spent [registry N-7] |
| E1 | ~25% of obtainable mix | reduced fine-tune | $1-2.5K [estimate \| 6.24 MVP arithmetic scaled down] |first curve point |
| E2 | ~50% | reduced fine-tune | $2-5K [estimate \| ditto] | second point |
| E3 | 100% of obtainable mix | full "MVP (adequate)" recipe | $5-10K [registry N-8] | third point; = 6.24 MVP scenario |
| E4 (only if E1-E3 trend holds) | 100% + curriculum, bigger/longer | "good-quality" recipe | $30-60K [registry N-8] | is the flat part still above the bar |

The #1 cost driver is the number of *failed* campaigns [registry N-8] — hence M0.5 and the rule that every rung must produce a signed eval before the next is funded (6.14 mechanics).

**Metrics — fixed BEFORE E1, one page, frozen:** PSNR/SSIM/LPIPS split (a) on held-out *views* of trained subjects and (b) on held-out *subjects* — conflating these turns overfitting into fake progress; plus a dedicated **back-side score** (render the unobserved hemisphere against multicam GT — the owner's rigs make this measurable in-house, which the original Flex4DHuman evaluation could not do [registry N-7]); plus temporal-LPIPS stability. The published 25.44 PSNR is not reproducible as an external anchor — eval re-anchors on our own held-out set (6.19's conclusion) [registry N-7].

**Go / No-Go to Variant 3:** GO only if held-out-*subject* quality improves monotonically with data scale across E1→E2→E3 (working threshold: ≥ +0.5 dB PSNR or −10% LPIPS per doubling [hypothesis | threshold to be sanity-checked against E1-E2 variance before being enforced]) AND the back-side score improves visibly in eyes-on review. NO-GO → the generative track closes; Variant 1 remains the product basis and the capture asset keeps its value.

## 2. Sub-track 2b — own data → asset (the commercial mirror)

Re-run the best curve point using **only** owner-captured data (2× GoPro sessions per 6.21, mini-rig later) with signed releases, plus the $0-license synthetic stack (MakeHuman/Poly Haven CC0, Rocketbox MIT, CMU mocap) [registry N-1, N-5, N-9]. Same recipe, clean provenance → the weights become sellable/deployable. 2b starts only after 2a's GO signal, because 2b pays the same compute *plus* capture costs for every experiment.

| Line | Cost | Basis |
|---|---|---|
| Capture sessions (subjects, releases, time) | €2-6K | [estimate \| ~20-60 subjects × modest session fees + legal paperwork; PL market rates] |
| Compute re-run of best rung | $5-10K (MVP) … $30-60K (good) | [registry N-8] |
| Legal package (GDPR consents, DPIA, release templates, one counsel review) | €1-2K | [estimate \| single PL counsel engagement; templates drafted in-house — see LEGAL-LICENSING.md] |

**2b deliverable beyond weights:** the eval protocol + capture SOP become the seed of Variant 3's dataset production line, and the EULA draft ships here (not in Variant 3) so pre-sales conversations can start with paper in hand.

## 3. Commercial vs non-commercial summary

| | 2a (research data) | 2b (own data) |
|---|---|---|
| Backbone | Wan 2.1 Apache-2.0 — clean [registry LB2-11] | same |
| Data | NC / ambiguous → output NC [registry N-14] | owner-captured + CC0/MIT synthetic → clean [registry N-1, N-9] |
| Output usable in product | NO — knowledge only, firewalled | YES — this is the asset |
| Body prior if needed | MHR Apache-2.0; Anny native mode Apache-2.0 (its smplx-mode is NC) [registry C4-5, C4-6] — SMPL-X itself stays out of commercial builds [registry LB2-1] | same |
| Cost for the same curve point | ~2-3× cheaper (no capture, no legal) | full price, but produces something you own |

## 4. Kill criteria and honesty rails

- Any rung that fails its signed eval twice → stop the ladder, write the post-mortem, do not average away the failure [registry N-8 \| failed campaigns are the top cost driver].
- If M0.5 (PRoPE pre-gate) fails → the replication premise itself is broken; fall back to Variant 1 + watch Layer-2 frontier models (Forge4D/NoPo4D per 6.18) instead of spending on E1.
- No metric from 2a may be quoted in marketing or sales material — 2a exists to steer money, not to make claims.
