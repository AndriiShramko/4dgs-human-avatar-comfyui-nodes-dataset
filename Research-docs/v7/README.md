# Research-docs v7 — the audited, three-variant implementation plan

**Date: 2026-07-27.** v7 is the apex planning document set for Shramko-Human-4D (faithful 4DGS of real human performance → ComfyUI nodes + a commercially clean dataset). It supersedes the *planning* role of v5 entirely and of v6's overlapping plan files (6.19/6.20/6.24 summaries), while v6 remains authoritative for implementation detail it pioneered — above all the canonical execution prompt 6.21 §21.9 and the agent-team mechanics 6.14 [registry N-6].

## Why v7 exists

v5/v6 were written by AI agents; the owner suspected hallucinations. v7 re-verified every load-bearing claim against primary sources — in BOTH directions: suspected phantom papers turned out to exist, while several "facts" (a fabricated Apple SHARP figure, present-tense descriptions of an unbuilt rig and a non-existent dataset, rumor prices) failed verification. Full verdicts: [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) (84 claims, each with URL + as-of date + verbatim quote or an honest not-found). Narrative: [AUDIT_V6.md](AUDIT_V6.md).

## Methodology and trust levels <!-- lint-skip -->

Every factual claim in v7 carries a status tag — `[verified]` (primary source + quote + date), `[estimate]` (derivation shown), `[hypothesis]` (needs an experiment), `[not-found-as-of]` (searched, not found — NOT a claim of non-existence), `[inherited-unverified]`, `[rumor]`, `[internal]`, `[registry ID]` (cross-reference). <!-- lint-skip -->
`tools/claims_linter.py` enforces the tags and checks URL liveness; it must exit 0 for every v7 change. Owner decisions recorded in v6 are binding constraints [registry N-1..N-5]: faithful 4DGS of real video (not avatar-from-photo), own-rig data only for commerce, DNA-Rendering/ActorsHQ never, solo-realizable on one RTX 6000 Pro, actual kit = 2× GoPro HERO 13.

## The three variants

| | Variant 1 — [Budget replay](VARIANT-1-BUDGET.md) | Variant 2 — [Money→quality curve](VARIANT-2-SCALING.md) | Variant 3 — [Flagship own-data](VARIANT-3-FLAGSHIP.md) |
|---|---|---|---|
| Question it answers | can we ship stable faithful replay now? | does paying for fine-tuning buy quality, and where does it flatten? | is the dataset+model business worth building? |
| Approach | per-capture 4D fit (no training), ComfyUI nodes over native GAUSSIAN [registry LB2-8] | Flex4DHuman-style fine-tune (Wan 2.1 1.3B + PRoPE) [registry N-7, LB2-11]; 2a research-data (knowledge) / 2b own-data (asset) | scale capture + training on owner data with releases |
| Result stability | **highest** — no domain gap by construction | experiment-dependent; gated per rung | depends on Variant-2 curve; gated |
| Budget | €450–950 [estimate \| VARIANT-1 §4] | 2a: ~€0 + $1-10K cloud; 2b: +€3-8K capture/legal; E4 option $30-60K [registry N-8] | ≈ €130–330K, 9–15 mo [estimate \| VARIANT-3 §3] |
| Commercial track | clean sub-path = gsplat-only [registry C4-7, N-12] | 2a NEVER commercial (data taint); 2b clean | the entire point; 3 gates first |
| Non-commercial track | default sub-path, fastest | 2a is it | pointless except as a grant project [registry N-1] |
| Starts | now | after Variant-1 acceptance + M0.5 pre-gate | only when all 3 gates green |

Cross-cutting docs: [LEGAL-LICENSING.md](LEGAL-LICENSING.md) (license map, the wall, GPL business model, GDPR package — DRAFT, not legal advice) · [DATASET-STATUS.md](DATASET-STATUS.md) (what actually exists: currently zero data, placeholder nodes) · [MVP-0-EXPERIMENT.md](MVP-0-EXPERIMENT.md) (the five cheap experiments that come before any real spend) · [LANDING-SYNC.md](LANDING-SYNC.md) (work order for the contradicting landing page) · [FLUX3-SPLIT-VIEW.md](FLUX3-SPLIT-VIEW.md) (2026-07-27 assessment: why the FLUX 3 "split view" is a prompt, not a feature, and where a general video generator may and may not touch this pipeline) · [BYTEDANCE-4D-MEDIA-SYSTEM.md](BYTEDANCE-4D-MEDIA-SYSTEM.md) (2026-09-05 competitive pass: ByteDance's LiveGS/multi-camera-4DGS/monocular-conversion stack, the DualGS/TaoGS/Director research lineage behind it, and its open-source status).

## Sequencing

MVP-0 experiments → Variant 1 to acceptance → Variant 2a curve (M0.5 first) → 2b clean recipe + EULA → three gates → Variant 3. Strictly sequential for a solo operator. Compute contingency: everything must also run on rented H100-class cloud at $2-3/hr [registry C3-4] — the free RTX 6000 Pro is a cost saver, not a dependency [registry C3-8 \| sm_120 build tax is real; cloud is the escape hatch].

## Change rules for v7

1. New claims → tagged, linter-clean, registry row if load-bearing.
2. Reality changes (capture, checkpoint, release) → DATASET-STATUS.md updated in the same PR.
3. No number in any public-facing text without a derivation from a verified baseline.
4. v5 is archived provenance (do not cite as evidence); v6 files carry pointer notices and stay citable for their audited implementation detail.
