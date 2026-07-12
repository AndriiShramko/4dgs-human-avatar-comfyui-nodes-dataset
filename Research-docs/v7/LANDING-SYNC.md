# LANDING-SYNC — contradictions between the live landing page and v7

> Part of Research-docs v7. The landing https://comfyui-4dgs-volumetric-node.eu was fetched live on 2026-07-12 [internal | fetch performed during the v7 audit run]. Fixing the landing is a SEPARATE task outside this PR — this file is the work order for it. Until fixed, the landing contradicts the repository's own research conclusions and red-team rules (v6/6.8 §8.5.2-8.5.3).

## Contradiction table (landing, verbatim → v7 status)

| # | Landing claim (verbatim, as fetched 2026-07-12) | v7 status | Required change |
|---|---|---|---|
| 1 | Apple SHARP: "~100 GB per minute of video" | fabricated figure, no traceable source [registry LB2-2d] | delete; if a comparison stays, compare against published codecs (QUEEN/DualGS per-frame sizes) with citations [registry LB2-4, LB2-5] |
| 2 | "Ultra-optimized lightweight files — less than 1GB per minute" | product does not exist; number has no derivation [registry C1-2; LB2-2d] | replace with a *target* derived from verified baselines, e.g. "target: research codecs already demonstrate ≈0.6-1.3 GB/min at 30fps (DualGS ~350 KB/frame, QUEEN ~0.7 MB/frame)" [registry LB2-13a, LB2-13b] — or drop the number |
| 3 | SHARP: "Limited 10–20° parallax only" | unverified characterization of someone else's product; SHARP is single-image → static 3DGS, so the axis is wrong anyway [registry LB2-2a] | delete the SHARP column entirely |
| 4 | SHARP: "Temporal 'boiling' in animations" | SHARP does not do animations (static output) [registry LB2-2a] | delete |
| 5 | "Absolute temporal stability" | banned by the repo's own red-team (6.8 §8.5.2: do not market "absolute temporal stability") [internal \| v6/6.8] | replace with measurable phrasing tied to the VARIANT-1 acceptance metrics |
| 6 | "100% legal for commercial use" | banned phrasing per the repo's own red-team rule [internal \| v6/6.8 §8.5.2]; legal status is track-dependent (see LEGAL-LICENSING.md) | replace with "commercially licensed, consent-documented" once the dataset exists — not before |
| 7 | SHARP: "Commercial use strictly forbidden" | HALF-true: weights research-only [registry LB2-2c], but the CODE license is permissive [registry LB2-2b] | if any SHARP mention survives, state it precisely — sloppy claims about Apple's licensing are a legal own-goal |
| 8 | "Full 360° volumetric view" / "Accurate back-side reconstruction" | requires the multicam disclaimer per 6.8 §8.5.3; monocular back-side is hallucinated by definition [registry N-2] | add capture-dependency wording: "surround coverage from multi-camera capture; single-camera captures are front-faithful" |
| 9 | "Only 53 slots left" (waitlist) | marketing scarcity; no capacity mechanism exists in the repo [registry C1-2] | either implement a real capacity constraint or drop the counter (UOKiK-style fake-scarcity risk for an EU-facing page) |
| 10 | "physically correct fabric simulation, micro-expressions, bullet-time slow-motion" (Technology section) | none of these exist; cloth-GNN is a design sketch with napkin-level timing [registry C2-5]; no code [registry N-6] | move to a clearly-labeled roadmap section |
| 11 | Page exists only in EN | (observation, not a contradiction) | fine as-is; note the landing's ES/PL variants mentioned in older project notes were not detected on the live page |

## Alignment rule going forward

The landing may only claim what DATASET-STATUS.md §1 lists as existing, plus clearly-labeled targets whose numbers carry derivations from the registry. Every future landing edit cites this file's table row it resolves.
