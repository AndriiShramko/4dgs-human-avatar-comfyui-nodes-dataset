# DATASET-STATUS — the honest state of data, rigs and code

> **DRAFT — not legal advice; factual status snapshot as-of 2026-07-27.** Part of Research-docs v7. This file exists because the repository's facade docs described aspirations in the present tense [registry C1-1, C1-2, C1-3]. This is the single source of truth on what exists; update it whenever reality changes.

## 1. What exists today

| Asset | Status | Evidence |
|---|---|---|
| Published dataset (free or enterprise) | **NONE.** No data files in the repo, no HF link, `download_free.py` has an empty repo_id | [registry C1-2] |
| 90-camera studio rig | **NOT BUILT.** A future plan, not an asset | [registry C1-3 \| v6/6.10: "the 90-camera rig is not yet built"] |
| Actual capture kit | 2× GoPro HERO 13, QR-timecode resync, no genlock, ~5 available subjects, no green screen | [registry N-5] |
| Owner's broader hardware | large GoPro fleet and multicam volumetric experience exist as *capability* (separate projects), not as captured 4D data in this project | [internal \| owner context; no capture sessions for this project have been run yet] |
| ComfyUI nodes | 113-line placeholder skeletons; loader/generator contain TODOs and return inputs | [registry N-6, C1-2] |
| Trained models / checkpoints | none | [registry N-6] |
| Research docs | v5 (archived prompts/essays), v6 (26 files incl. audited plans + owner decisions), v7 (this set) | [internal \| Research-docs/] |

## 2. Corrections of record

- "Processing petabytes of volumetric data" and "hundreds of terabytes" cannot both describe the same plan; neither describes anything that exists. Correct statement: **zero bytes of project data captured to date**; the *planned* Variant-3 production scale is on the order of 50–150 TB live storage [registry C1-1; estimate range from VARIANT-3 §3].
- "Hardware-synchronized" describes no current equipment: GoPro offers no genlock; sync is QR-timecode resync with a measured sub-frame residual [registry C1-3, N-5]. A genlocked rig is a Variant-3 *decision*, priced in VARIANT-3 §4.
- "Thousands of high-quality human scans" (enterprise tier) is a target contingent on Variant-3 gates, not inventory [registry C1-2].

## 3. What the dataset will be, when it exists

Owner decisions bind the plan: own-rig data only, no purchased datasets, DNA-Rendering/ActorsHQ never used [registry N-1, N-3]. Combined with the market fact that every verified academic 4D-human dataset is non-commercial and the one 2026 newcomer has no dataset license at all [registry LB2-6], the differentiator is fixed: **consent-documented, commercially licensable multi-view human 4D capture** (see LEGAL-LICENSING.md §4 for the consent package and VARIANT-3 for gates/scale).

Dual-tier plan (inherited from the repo's original framing, now honestly labeled as future): a free CC BY-NC sample tier for community/citations + an enterprise tier under a negotiated EULA [hypothesis | tier split to be validated against the first LOI conversations].

## 4. Update protocol

Any capture session, any published sample, any trained checkpoint → update §1's table in the same PR, with an `[internal | path]` pointer to the artifact. The claims linter runs on this file; a number without a tag will fail CI-style checks. Facade docs (root README, dataset/*.md) must link here instead of restating quantities.
