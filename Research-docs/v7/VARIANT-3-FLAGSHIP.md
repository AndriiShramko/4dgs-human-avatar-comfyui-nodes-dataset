# VARIANT 3 — Flagship: large-budget training on our own captured data

> Part of Research-docs v7 (2026-07-12). This variant does NOT start until three gates are green. It converts the owner's real advantage — capture expertise and rigs — into the one asset this market lacks: a commercially clean, GDPR-documented multi-view human 4D dataset [registry N-1, LB2-6], with a fine-tuned model and ComfyUI distribution on top.

## 1. The three gates (all mandatory, in any order)

1. **Curve gate:** Variant 2a returned GO — held-out-subject quality scales with data and has not plateaued (see VARIANT-2 §1).
2. **Demand gate:** 3–5 written LOIs / pre-sale commitments from named buyers (GS-pipeline startups, synthetic-data vendors, labs with commercial budgets) with a price bracket. The competitive shelf already exists — RenderPeople 4D People at €79/model, HumanDataset with "over 35,000 scans" on a "330-camera setup" [registry C4-10a, C4-10b] — so demand must be proven, not assumed.
3. **Legal gate:** the GDPR/likeness package from LEGAL-LICENSING.md §4 is complete (release + Art. 9 consent templates counsel-reviewed, DPIA done, recall mechanism designed) BEFORE the first commercial capture day.

No gate → no Variant 3. Money spent here without gates is the "failed campaign" cost driver at its largest scale [registry N-8].

## 2. What gets built

- **The dataset (primary asset).** Owner-captured multi-view human performance data with full releases. All verified academic 4D-human datasets are non-commercial (ActorsHQ CC BY-NC, DNA-Rendering agreement-gated, MVHumanNet non-commercial ToU, THuman "non-commercial research purposes only", HuMMan S-Lab NC) [registry N-3 \| verified in the v7 planning pass across each dataset's terms], and the newest entrant VolHuMe has no published dataset license at all [registry LB2-6]. A commercially clean alternative with documented consent is the differentiator — it is also the only deliverable here that does not depreciate as models improve.
- **The model (demonstrator).** The best Variant-2 recipe re-trained at scale on the dataset; expected shelf life 12–18 months [estimate | typical generative-model generation cadence 2024-2026], after which the dataset trains its successor.
- **The distribution (funnel).** ComfyUI nodes (GPL, free — see LEGAL-LICENSING.md §3) + web viewer demos; monetization sits in the dataset, weights and API, never in the nodes.

## 3. Scale and budget (perimeters kept separate — the 6.24 §24.6 rule)

| Perimeter | Range | Basis |
|---|---|---|
| Data production: 100–300 subjects × 1-3 min multi-view | €25–90K | [estimate \| subject fees + releases €50-150/person, 15-40 studio days, ops; scale target itself is a hypothesis to be set by the 2a curve, not by round numbers] |
| Capture hardware upgrade | €10–60K | [estimate \| genlocked machine-vision rig vs GoPro fleet — see §4; wide range until §4 is decided] |
| Processing + storage (50–150 TB live) | €8–20K | [estimate \| COLMAP/fit compute on own GPU + cloud bursts; storage at C3-4-class cloud rates or local arrays] |
| Model training at scale | $30–60K | [registry N-8 \| 6.24 "good-quality" scenario] |
| ML engineer (contract, 6–12 mo) | €60–120K | [estimate \| EU contract rates; mandatory — the owner is a capture expert, not an ML researcher, and 6.13's solo plan covers fine-tune replication, not novel-scale training] |
| Legal (GDPR package, EULA, counsel) | €3–8K | [estimate \| includes recall-mechanism design and dataset EULA review] |
| **Total** | **≈ €130–330K, 9–15 months** | [estimate \| sum of the above; the legacy "$121K" was a from-scratch training figure with different perimeter — do not mix, per registry C3-6] |

Non-commercial Variant 3 is pointless (burning six figures with no right to sell); the only NC scenario is a research grant producing a citation-magnet CC BY-NC dataset [hypothesis | viable only with an academic partner carrying the grant].

## 4. The genlock line (honesty requirement inherited from the owner's own findings)

GoPro cannot genlock — the owner proved this in a dedicated project; QR-timecode resync bounds sync at ~±1 frame with measurable sub-frame residual [registry C1-3, N-5]. For a *premium* 4D dataset of fast human motion this is a quality ceiling. Variant 3 must choose explicitly and price it in §3:

- **(a) Genlocked machine-vision cameras** (hardware sync): removes the ceiling; multiplies camera cost roughly 5–10× vs GoPro per head [estimate | industrial GS cameras + lenses + sync + cabling vs consumer action cams].
- **(b) GoPro fleet + measured-offset declaration:** every take ships with its documented sub-frame offset; the dataset is positioned (and priced) for quasi-static/moderate motion. Cheaper, honest, but caps the premium tier.

Either is defensible; pretending (b) is (a) is exactly the "hardware-synchronized" fiction v7 removes [registry C1-3].

## 5. Sequencing and kill criteria

V1 (pipeline + capture SOP) → V2a (curve) → V2b (clean recipe + EULA) → gates → V3. Strictly sequential; a solo operator running these in parallel finishes none of them.

- **Kill during production:** dataset pre-sales < 2 paying customers after the first 50-subject tranche → stop scaling capture, sell the tranche, keep the model as a demonstrator.
- **Kill on quality:** if the scaled model fails to beat the 2b checkpoint on the frozen eval, do not ship it as an upsell — ship the dataset; the model failure is a finding, not a marketing problem.
- **Standing risk watch:** if ComfyUI core lands a native temporal-GAUSSIAN type, nodes migrate (design already thin-adapter, VARIANT-1 §3); if a frontier lab releases an ungated commercial video→4DGS model [registry N-11 \| watchlist from 6.18], the dataset REMAINS valuable (it becomes fine-tune/eval fuel) — the model line pivots to fine-tuning that release.
