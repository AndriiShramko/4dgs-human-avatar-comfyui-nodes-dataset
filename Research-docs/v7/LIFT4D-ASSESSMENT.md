# Lift4D — capability assessment for the capture → 4DGS pipeline

> Part of Research-docs v7. Research pass 2026-07-30, launched by the owner's questions: is Lift4D real 4DGS or "just a point cloud", at what pipeline stage can it help, and does it need fine-tuning? Every claim below carries a status tag; verdicts live in [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) section L. Method note: arxiv.org returns HTTP 403 to this session's proxy, so the abstract text was retrieved via a search snapshot of the abs page rather than a direct fetch; all other quotes are first-hand.

## 1. Verdict in one table

| Owner's question | Answer | Where |
|---|---|---|
| 4DGS or point cloud? | **4DGS** — a canonical set of 3D Gaussians plus a learned deformation field over time (deformable 3DGS, SC-GS lineage). Not a point cloud at any stage of its output [registry L-2, L-4] | §2 |
| Can the model be fine-tuned? | **There is no "Lift4D model" to fine-tune.** It is a per-video test-time optimizer; every run *is* the training. Its trainable parts are upstream third-party checkpoints (SAM 3D Objects, SAM3, Stable Zero123) [registry L-3] | §4 |
| Do WE need to fine-tune it? | **No, not now.** Use it as-is for a cheap monocular baseline experiment; adapting its upstream priors to humans is a separate research project that current evidence does not justify [registry L-3, L-7] | §4, §7 |
| Which stage does it help? | MVP-0 **baseline** (what does monocular-only buy?), Variant-1 **initialization + optimization recipes**, and nothing in Variants 2/3 [registry L-1..L-5] | §3 |
| Commercial track? | **NO — research/NC track only.** The repo ships no license file, its diffusion prior is non-commercial, and its vendored splatting stack inherits the Inria rasterizer [registry L-6, L-7, L-9] | §5 |
| Faithful-capture products? | Its diffusion back-fill of unseen regions is generated content — same rule as FLUX 3: allowed only labelled, never silently inside a faithful deliverable [registry L-12, N-2] | §6 |

## 2. What Lift4D actually is — and why it is 4DGS, not a point cloud

Lift4D (CMU: Litman, Ma, Shah, Ugrinovic, Kitani, De la Torre, Tulsiani) is a monocular-video → 4D-asset reconstruction method, arXiv 2606.23688, code at https://github.com/yehonathanlitman/Lift4D, project page https://lift4d.github.io/ [registry L-1]. The problem statement, from the abstract: "Reconstructing dynamic non-rigid objects from monocular video requires integrating visual cues from direct observations with data-driven priors over geometry and appearance" [verified | https://arxiv.org/abs/2606.23688 | as-of 2026-07-30 | "Reconstructing dynamic non-rigid objects from monocular video requires integrating visual cues from direct observations with data-driven priors over geometry and appearance"].

The pipeline has three stages [registry L-4]:

1. **Segment** — SAM3 isolates the target object in every frame from a text prompt.
2. **Per-frame 3D** — a single-view 3D reconstruction model (SAM 3D Objects) is adapted to produce "temporally consistent per-frame predictions via causal latent conditioning" [verified | https://arxiv.org/abs/2606.23688 | as-of 2026-07-30 | "temporally consistent per-frame predictions via causal latent conditioning"].
3. **4D fusion** — those per-frame estimates initialize "a deformable 3D Gaussian Splatting representation" [verified | https://arxiv.org/abs/2606.23688 | as-of 2026-07-30 | "a coherent initialization for a deformable 3D Gaussian Splatting representation"], which is then "sculpted" to match the video by an "occlusion-aware optimization that faithfully recovers visible surface details while completing unobserved regions using a view-conditioned diffusion prior" [verified | https://arxiv.org/abs/2606.23688 | as-of 2026-07-30 | "occlusion-aware optimization that faithfully recovers visible surface details while completing unobserved regions using a view-conditioned diffusion prior"].

So the answer to "is this 4DGS or just a point cloud" is unambiguous: the final representation is exactly our representation class — canonical Gaussians + deformation over time. The fusion stage is a vendored SC-GS derivative (the repo's `lift4d_scgs/` directory), so the output artifact is an SC-GS-style deformable Gaussian checkpoint [estimate | inferred from the repo's lift4d_scgs/ directory name plus SC-GS's architecture; confirm by opening a produced checkpoint in experiment E-L1]. The README's own one-line scope: "Recover complete 4D asset reconstructions from videos in the wild" [verified | https://raw.githubusercontent.com/yehonathanlitman/Lift4D/main/README.md | as-of 2026-07-30 | "Recover complete 4D asset reconstructions from videos in the wild"].

One scope caveat that matters for a human-avatar project: Lift4D is a **general-object** method. The abstract speaks of "dynamic non-rigid objects"; the README contains no SMPL, skeleton, or human-prior machinery [not-found-as-of 2026-07-30 | queries: full README text for "SMPL", "skeleton", "human", "body model"]. For us this cuts both ways: no research-only SMPL-X taint to inherit [registry LB2-1], and motion is carried by SC-GS sparse control points rather than kinematics — which matches Variant-1 "faithful replay" philosophy (no rigging promised) but gives no riggable avatar.

## 3. Where it can help our pipeline, stage by stage

| v7 stage | Lift4D role | Track |
|---|---|---|
| MVP-0 (cheap experiments) | **Monocular baseline** — run one GoPro clip through it end-to-end; this quantifies exactly what the second camera and our multi-view fit buy, before any real spend [§7] | NC sandbox only [registry L-6] |
| Variant 1 (per-capture 4D fit) | **Recipe donor, not a component.** Two transferable ideas: (a) temporally-consistent per-frame 3D initialization via causal conditioning — directly relevant to our sparse-view (2× GoPro) fitting, where per-frame init quality dominates; (b) the two-phase geometry-then-appearance optimization schedule with occlusion-aware supervision [registry L-4, L-10] | ideas transfer freely; code does not (no license) [registry L-6] |
| Variant 2 (fine-tune backbone) | **Not a backbone.** It is not a feed-forward model and produces no reusable weights; Wan 2.1 + PRoPE remains the plan [registry N-7, LB2-11]. The causal-conditioning trick is the only piece worth citing there | n/a |
| Variant 3 (flagship own-data) | No role — it neither generates training data we may use commercially nor scales beyond per-video optimization [registry L-3, L-6] | n/a |
| ComfyUI nodes | Its output should be convertible to a per-frame .ply sequence for the native GAUSSIAN type [hypothesis | export canonical Gaussians + per-frame deformed positions from the SC-GS checkpoint in E-L1; ComfyUI temporal support still absent per registry LB2-7, LB2-8] | NC demo only |
| Competitive landscape | Closest published shape to "faithful 4D from commodity video" — a watch item alongside CAT4D-class work [registry F-22]; it validates our core bet that per-capture optimization (not feed-forward generation) is the quality path today | — |

The honest summary: Lift4D is **evidence and a recipe book**, not a pipeline component. It is the strongest published argument that our Variant-1 architecture (initialize from priors, then optimize against real video evidence) is the right one for in-the-wild footage with occlusions.

## 4. The fine-tuning question, answered precisely

"Можно ли дотренировать эту модель?" decomposes into three different questions:

1. **Fine-tune Lift4D itself?** Meaningless — there are no Lift4D weights. It is "a test-time optimization framework": each video is optimized from scratch, geometry phase then appearance phase, resuming a checkpoint between them; the README's documented iteration budgets run to 19,999–29,999 optimization steps per video, tested on a single A100 40GB [verified | https://raw.githubusercontent.com/yehonathanlitman/Lift4D/main/README.md | as-of 2026-07-30 | "Tested on an A100 40GB GPU"] [registry L-3, L-10]. What we would "train" is each capture — which is exactly what Variant 1 already does with our own stack.

2. **Fine-tune its upstream priors on humans?** Technically the only real option: the per-frame model (SAM 3D Objects, gated Meta checkpoint) and the back-fill prior (Stable Zero123) are generic-object models, and a human-specialized prior would plausibly improve hands/faces/clothing. But this is a full research project with no published human variant to start from [not-found-as-of 2026-07-30 | queries: "Lift4D human", HF search for human-tuned SAM 3D Objects / Zero123 checkpoints], and the Zero123 branch is licensed non-commercially [registry L-7] — so any effort spent there is trapped on the NC track. Not justified before the E-L1 baseline even exists.

3. **Do we need any of that?** No. Our data plan is 2× synchronized GoPro, not monocular [registry N-5]; with real second-view supervision, the diffusion back-fill that fine-tuning would improve matters *less*, not more. Decision: use Lift4D as-is, in a sandbox, as a baseline and recipe source. Revisit only if E-L1 shows monocular quality within striking distance of the two-camera fit [hypothesis | decided by experiment E-L1, §7].

## 5. The license wall

| Component | Status | Evidence |
|---|---|---|
| Lift4D code | **No LICENSE file at all** → default copyright, no reuse/modification/redistribution rights granted; research courtesy only | [not-found-as-of 2026-07-30 | queries: root file listing of github.com/yehonathanlitman/Lift4D — .gitignore, README.md, lift4d_datasets.py, segment_video.py, lift4d_scgs/, sam3d/; no LICENSE; README has no license section] |
| SAM3 + SAM 3D Objects (Meta, gated) | commercial use permitted with restrictions under the custom "SAM License" | [verified | https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE | as-of 2026-07-30 | "a non-exclusive, worldwide, non-transferable and royalty-free limited license"] — restrictions: no military/weapons use, trade-control compliance, attribution on redistribution; both HF cards are gated with tag `license:other` [registry L-8] |
| Stable Zero123 (back-fill prior) | **non-commercial** — the blocking dependency | [verified | https://huggingface.co/stabilityai/stable-zero123 | as-of 2026-07-30 | "Stable Zero123 included some CC-BY-NC 3D objects, so it cannot be used commercially, but can be used for research purposes"] — the card also points commercial users to https://stability.ai/license and lists a second variant, Stable Zero123-C, with its own license link (`license_link_stable_zero123_c` in the card frontmatter) [registry L-7] |
| Vendored SC-GS (`lift4d_scgs/`) | SC-GS README declares MIT license, but its install includes the Inria-derived rasterizer — the same research-only inheritance as every unclean splatting stack we have audited | [verified | https://github.com/yihua7/SC-GS | as-of 2026-07-30 | "pip install ./submodules/diff-gaussian-rasterization"] [registry L-9, C4-7, N-12] |
| lucidrains/lift4d | independent reimplementation under MIT license, explicitly WIP ("lift4d (wip)"), 3 commits at time of check — a potential future clean-room escape hatch, currently unusable | [verified | https://github.com/lucidrains/lift4d | as-of 2026-07-30 | "Implementation of Lift4d, Harmonizing Single-View 3D Estimation for 4D Reconstruction In-the-Wild, from Litman et al. of Carnegie Mellon"] [registry L-11] |

Net result: the stack as shipped is research-track only, twice over (missing license on the glue code, NC on the prior), and would stay off the gsplat-only clean sub-path even if those were fixed, because of the Inria rasterizer inheritance [registry C4-7]. Rules of engagement: run it in the NC sandbox, never copy its code into our nodes, and treat its *methods* (which are published in the paper and not licensable) as the transferable asset.

## 6. Faithfulness caveat — the diffusion back-fill

Lift4D completes what the camera never saw: unobserved regions are filled by "a view-conditioned diffusion prior" [verified | https://arxiv.org/abs/2606.23688 | as-of 2026-07-30 | "completing unobserved regions using a view-conditioned diffusion prior"]. That is generated content, and the owner's product definition — faithful 4DGS of a real captured performance [registry N-2] — plus the EU AI Act Article 50 marking duties [registry F-18] impose the same rule we set for FLUX 3: generated regions may exist only in clearly-labelled outputs, never silently inside a faithful deliverable.

The important difference from FLUX 3, and the reason Lift4D is a watch item rather than a hard NO: generation here is *scoped and subordinated*. Visible surfaces are recovered from real evidence — "faithfully recovers visible surface details" [verified | https://arxiv.org/abs/2606.23688 | as-of 2026-07-30 | "faithfully recovers visible surface details"] — and the prior only touches what no camera observed. In a two-camera rig the unobserved set shrinks toward self-occlusions only. A future product option: ship faithful deliverables from multi-view evidence, and offer labelled "AI-completed 360°" as a separate tier [hypothesis | product decision, blocked on licensing per §5 regardless].

## 7. Proposed experiment E-L1 (MVP-0 addendum, cheap)

Goal: measure what monocular-only reconstruction achieves on our actual subject matter, so the two-camera Variant-1 fit has a quantified baseline.

1. NC sandbox on a rented A100/H100-class GPU at $2-3/hr [registry C3-4]; the documented footprint fits a single A100 40GB [registry L-10].
2. One 30–60 s clip of a clothed person from a single GoPro, SAM3 prompt "person" [estimate | clip length chosen to match the README's long-video config threshold].
3. Run all three stages; record wall-clock, VRAM peak, and per-stage failure modes [hypothesis | expected: hands and cloth self-occlusion stress the per-frame model].
4. Evaluate: temporal flicker on visible surfaces, geometry of the never-seen back side (expected: plausible invention, not reconstruction), and checkpoint format/exportability to .ply for the ComfyUI GAUSSIAN type [registry LB2-8].
5. Compare against the Variant-1 two-camera fit of the same performance once it exists; the delta is the measured value of the second camera [hypothesis | this number feeds the Variant-2 go/no-go reasoning].

Optional E-L2, only if E-L1 impresses: hack second-view supervision into the sculpting phase (SC-GS optimizes against rendered losses, so adding a second calibrated view is architecturally plausible) and see whether the diffusion prior's share of the surface collapses [hypothesis | requires code modification we may make privately for research; redistribution stays blocked by the missing license per §5].

## 8. What would change this verdict

Re-open this file if any of these becomes true: (a) the authors add a LICENSE file [registry L-6]; (b) a human-specialized variant or pretrained adaptation weights are published; (c) the Stable Zero123-C swap is validated and its commercial terms read in full [registry L-7]; (d) lucidrains/lift4d matures into a runnable MIT license implementation [registry L-11] — which would still need a clean rasterizer (gsplat) to reach the commercial sub-path [registry C4-7]; (e) E-L1 shows monocular quality close enough to two-camera to change the capture economics. Until then: Lift4D is a baseline, a recipe book, and the best published validation of the Variant-1 architecture — not a pipeline component.
