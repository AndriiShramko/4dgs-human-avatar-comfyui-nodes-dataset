# Marigold V2 — permissively-licensed dense priors (depth / normals / albedo) for the capture → 4DGS pipeline

> Part of Research-docs v7. Research pass **2026-09-10**, two days after the arXiv posting and one day after the checkpoint repository was last updated [registry MG-1]. Trigger: the owner asked what the Marigold V2 project page means for Shramko-Human-4D. Every claim below carries a status tag; verdicts live in [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) section MG.
>
> **Read this on the next research pass.** Marigold V2 is not a competitor to this project and not a 4DGS method. It is a *component candidate* that lands squarely on an open question v6 left unanswered, and it arrives with a licence chain this repo has been hunting for. §8 lists exactly what to re-check.

## 1. Verdict in one table

| Question | Answer | Why |
|---|---|---|
| Does it replace any part of the 4D fit? | **NO** | it is a per-image dense predictor, not a reconstruction or a camera front-end [registry MG-1] |
| Does it replace COLMAP / VGGT for poses? | **NO** | it outputs no camera parameters at all [registry MG-1, MG-6] |
| Is it a candidate for the depth/normal prior term in the fit loss? | **YES — leading candidate** | it is the first shortlisted depth prior whose whole chain (code, weights, base model) verifies as Apache-2.0 [registry MG-2, MG-3, MG-4] |
| Is it usable on the **clean commercial sub-path**? | **YES, subject to §5** | Apache-2.0 end to end; no non-commercial rail to trip over [registry MG-2, MG-3, MG-4] |
| Is it usable on video as-is? | **NOT ESTABLISHED — the main risk** | single-image model, no temporal handling published anywhere, and the encode step is seeded [registry MG-11, MG-12] |
| Is it validated on humans? | **NO** | trained on synthetic indoor and driving scenes; no human-domain evaluation published [registry MG-9, MG-10] |
| Does an official ComfyUI plugin exist? | **NO — open niche** | it is an unchecked item on the authors' own checklist [registry MG-13] |
| Does the fine-tuning recipe fit the owner's solo-hardware constraint? | **YES** | every released model was trained on a single 32 GB GPU [registry MG-7, MG-8] |

Short version: **the licence chain is the finding; the temporal behaviour is the risk.** Both are cheap to settle — see E-06 in [MVP-0-EXPERIMENT.md](MVP-0-EXPERIMENT.md).

## 2. What Marigold V2 actually is

- A family of **single-step dense predictors** obtained by fine-tuning a pretrained diffusion transformer: "Marigold V2 is a family of models and a cost-effective fine-tuning protocol that repurposes a pretrained diffusion transformer into single-step dense predictors: depth, see-through depth, surface normals, albedo, and other dense modalities" [verified | https://github.com/huawei-bayerlab/marigold-v2 | as-of 2026-09-10 | "repurposes a pretrained diffusion transformer into single-step dense predictors: depth, see-through depth, surface normals, albedo, and other dense modalities"].
- Published as ACM TOG 45(6), to be presented at SIGGRAPH Asia 2026; arXiv 2609.08084, DOI 10.1145/3842528 [registry MG-1]. Authors are EPFL / HUAWEI Bayer Lab / University of Bologna, including the Marigold V1 project lead [registry MG-1].
- The checkpoints are **rank-128 LoRA adapters over a 4-bit-quantized DiT**, plus a fine-tuned VAE decoder where trained; the frozen base model is downloaded separately [verified | https://huggingface.co/huawei-bayerlab/marigold-v2-0 | as-of 2026-09-10 | "Each checkpoint here is a set of rank-128 LoRA adapters for the 4-bit quantized DiT plus, where trained, the fine-tuned VAE decoder."].
- Headline quality claim, from the abstract: "16-26% improvement in AbsRel over the previous best on KITTI and ETH3D" and, qualitatively, "our model resolves fur, foliage, and hair-thin edges that have eluded prior models" [registry MG-5, MG-14].
- Inference needs **about 17 GB of GPU memory at 1024² and about 29 GB at 2048²** [registry MG-6]; outputs are float32 `.npy` per image plus PNG visualisations [verified | https://github.com/huawei-bayerlab/marigold-v2 | as-of 2026-09-10 | "predictions_npy/*.npy (depth: [H, W]; normals and albedo: [3, H, W])"].

## 3. Why this matters here: it answers an open v6 question

v6 left an explicit open question — *"Foundation model depth priors: Which monocular depth model (Depth Anything V2 vs Metric3D vs UniDepth) provides most stable gradients for our multi-camera setup?"* [internal | Research-docs/v6/6.1-loss-functions.md §"Open Questions"] — while the architecture already committed to using such priors: the training loss is "L1 + LPIPS + DISTS + depth priors + opacity entropy" [internal | Research-docs/v6/README.md] with "normal consistency loss and geometric depth priors from foundation models … as auxiliary supervision" [internal | Research-docs/v6/6.1-loss-functions.md].

The licence rail was the unresolved half of that question. This repo already records that for commercial work it must "prefer the permissive depth/pose stack" [internal | Research-docs/v6/6.15-world-tracing-analysis.md], that MoGe's "weights license not stated on repo — verify HF card" [internal | Research-docs/v6/6.12-noncommercial-mvp-track.md], and that a bundled Depth-Anything-V2 lineage was flagged as "most of them being incompatible with commercial use" in a neighbouring stack [internal | Research-docs/v6/6.22-4c4d-and-4dvai-competitive-landscape.md]. None of those verdicts is re-verified by this pass — they are cited only to show what the gap was.

Marigold V2 closes that gap for the first time on a **fully verified three-link chain**: code Apache-2.0 [registry MG-2], weights Apache-2.0 [registry MG-3], base model Qwen-Image-Edit-2509 Apache-2.0 [registry MG-4]. That is the specific property the clean sub-path of Variant 1 requires [internal | Research-docs/v7/VARIANT-1-BUDGET.md §6], and the wall rule that quarantines non-commercial outputs simply does not engage [internal | Research-docs/v7/README.md].

**Caveat that must travel with the finding:** Apache-2.0 covers the code, the LoRA weights and the base model. It does **not** cover the training datasets — "Qwen-Image-Edit-2509 and the datasets keep their own licenses" [verified | https://github.com/huawei-bayerlab/marigold-v2 | as-of 2026-09-10 | "Qwen-Image-Edit-2509 and the datasets keep their own licenses."]. Inference is therefore clean; *re-training on the released data mix* re-opens a dataset licence question that this pass did not investigate [not-found-as-of 2026-09-10 | queries: Hypersim, Virtual KITTI 2 and LayeredDepth-Syn licence terms were not read in this pass].

## 4. Where it could plug in

| Stage | Use | Status |
|---|---|---|
| Variant 1, 4D fit on the 2-camera rig | depth prior as the anti-collapse disparity regulariser, normals as the normal-consistency term | [hypothesis | E-06 must first show the per-frame prior is temporally stable enough to be supervision rather than noise] |
| Variant 1, hair and silhouette | the published strength is exactly hair-thin edges [registry MG-14]; hair is the classic human-4DGS failure | [hypothesis | eyes-on comparison against the current prep stack on one own capture] |
| Variant 1, metric lift | outputs are affine-invariant, so a per-frame scale+shift solve against COLMAP sparse points is required before the prior can constrain metric geometry [registry MG-6] | [hypothesis | the authors advertise "metric depth completion" as an application, but the depth-completion code is an unchecked checklist item — registry MG-13] |
| Albedo, augmentation and SH decomposition | a view-independent albedo channel is what the DC term of a Gaussian already represents [internal | Research-docs/v5/5.6.1.md], and HDRI relighting augmentation is already in the data plan [internal | Research-docs/v6/6.10-data-strategy.md] | [hypothesis | untested here; albedo quality is reported only on a synthetic indoor test split — registry MG-15] |
| Camera poses | none — it produces no extrinsics or intrinsics | ruled out [registry MG-1] |

Two cameras is a thin geometric constraint by construction [internal | Research-docs/v7/VARIANT-1-BUDGET.md §1], which is precisely the regime where a strong per-frame prior is worth the most — and also the regime where a *flickering* prior does the most damage, because there is no third view to out-vote it.

## 5. What is NOT established (the honest gaps)

1. **No temporal model, and a stochastic encode.** Nothing in the repository, the model card or the abstract discusses video, temporal consistency or frame-to-frame coherence [registry MG-11]. Worse, inference exposes a "seed for the VAE encoder sampling (default 2025)" [registry MG-12] — the encode step is sampled, so identical frames are not guaranteed identical predictions. Per-frame flicker in a supervision signal is worse than no supervision. **This is the gating unknown, not a footnote.**
2. **No human-domain validation.** Depth trains on Hypersim and Virtual KITTI 2 (about 74k images), normals and albedo on Hypersim, see-through depth on LayeredDepth-Syn [registry MG-9] — synthetic indoor scenes and synthetic driving. Every published number is on those same domains plus NYUv2 / ScanNet / iBims-1 / Sintel / DIODE [registry MG-5, MG-15]. No human-, portrait- or performance-capture benchmark appears anywhere [registry MG-10]. Generalisation to a GoPro-framed human is an assumption until measured here.
3. **Affine-invariant output.** "Outputs are affine-invariant, i.e. depth up to an unknown scale and shift per image" [registry MG-6]. Unknown *per image* means the alignment problem is itself per-frame, and therefore compounds gap 1.
4. **No runtime figure published.** Memory is documented, throughput is not [registry MG-16]. Cost per capture — frames × 2 cameras × one DiT step at 1024² — is unknown and must be measured, not assumed cheap because the model is "single-step". The 4-bit quantisation also happens on first load and "takes a few minutes" [registry MG-6].
5. **Re-training has a gated dependency.** The iREPA component needs DINOv3, which is gated on Hugging Face [registry MG-17]; its licence was not read in this pass and would have to be, before any fine-tune of ours is planned on this recipe [not-found-as-of 2026-09-10 | queries: DINOv3 licence text not fetched in this pass].

## 6. The strategic opening: no ComfyUI plugin exists

The authors' own checklist carries four unchecked items — depth completion code, see-through evaluation, Diffusers integration, and **ComfyUI plugin** [registry MG-13]. This repo's product is ComfyUI nodes, and v7 already records that the ComfyUI temporal niche is open [registry LB2-7, LB2-8].

Read this soberly: a plain "run Marigold V2 in ComfyUI" node is a **race against upstream**, who have it on their list and own the model. The defensible version is the part upstream explicitly does not do — a **video-oriented wrapper**: batched per-frame inference over a clip, per-frame affine alignment to a common reference, temporal smoothing / flicker suppression, and export in the shape the 4D fit consumes. That is adjacent to the `GAUSSIAN_SEQUENCE` design already chosen for Variant 1 [internal | Research-docs/v7/VARIANT-1-BUDGET.md §3], and it is only worth building if E-06 shows the flicker is both real and fixable [hypothesis | E-06 decides this].

## 7. The fine-tuning protocol as a Variant-2 datapoint

Binding owner constraint N-4 is that the project must be solo-realizable on one RTX 6000 Pro [registry N-4]. Marigold V2 is a published, reproducible existence proof for that class of ambition: "All released models were trained on a single 32 GB GPU with batch size 1. Stage 1 of the depth model (160k steps) takes about five days, every other config (30k steps) about one day" [registry MG-7], and the authors frame it as "within reach of individual practitioners and small labs" [registry MG-8]. The training framework is a YAML-composed registry with the surface-normal experiment released as the worked example of adding a new task [registry MG-18].

Implication, tagged as a hypothesis and **not** a plan change: a narrow fine-tune — a human-specialised depth/normal predictor trained on the project's own captures — is a far smaller bet than the Flex4DHuman / Wan 2.1 replication that Variant 2 currently centres on [internal | Research-docs/v7/VARIANT-2-SCALING.md], and could serve as a cheaper rung below the M0.5 pre-gate [hypothesis | only meaningful after E-06, after own captures exist, and after the DINOv3 and dataset licence questions in §5 are answered]. Variant 2's sequencing is unchanged by this pass.

## 8. What the next research pass must re-check

1. Has the **ComfyUI plugin** checkbox been ticked upstream? If yes, the §6 opening narrows to the temporal wrapper only [registry MG-13].
2. Has **Diffusers integration** landed? It would cut the integration cost sharply and change the node-build estimate [registry MG-13].
3. Has the **depth-completion code** landed? It is the route from affine-invariant to metric [registry MG-13].
4. Any **video / temporal follow-up** from the authors or third parties — the single decisive unknown of §5 [registry MG-11].
5. Any **human-domain evaluation** by anyone [registry MG-10].
6. **Per-image runtime** figures at 1024², from the authors or a third party [registry MG-16].
7. The **DINOv3 licence** and the training-dataset licences, if and only if a fine-tune of our own moves from hypothesis to plan [registry MG-17].
8. Whether E-06 has been run and its result note filed — if not, this whole page is still a candidate assessment, not a component decision.

## 9. Sources

- Code and training framework: https://github.com/huawei-bayerlab/marigold-v2 (Apache-2.0) [registry MG-2]
- Checkpoints: https://huggingface.co/huawei-bayerlab/marigold-v2-0 (Apache-2.0) [registry MG-3]
- Project page (the page the owner sent): https://hf.co/spaces/huawei-bayerlab/marigold-v2-web
- Interactive demo: https://huggingface.co/spaces/toshas/Marigold-V2
- Paper: https://arxiv.org/abs/2609.08084 · DOI https://doi.org/10.1145/3842528
- Base model: https://huggingface.co/Qwen/Qwen-Image-Edit-2509 (Apache-2.0) [registry MG-4]
