# AffineGS — geometry-prior assessment for the capture → 4DGS pipeline

> Part of Research-docs v7. Research pass 2026-11-12, one week after the paper appeared on arXiv and six days after the last push to its code repository [registry H-1, H-6]. Read first-hand: the arXiv HTML, the NeurIPS poster page, the project page, the GitHub repository, and the licence text of every depth model involved. Verdicts live in [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) section H. **AffineGS is a static-scene geometry regulariser — not a human, avatar or 4D method.** It matters to us as a component idea, and its most useful lesson is about licensing.

## 1. Verdict in one table

| Possible use | Verdict | Why |
|---|---|---|
| Adopt the upstream repository as a pipeline stage | **NO** | static scenes only, pinned old toolchain, per-scene hand-tuning, hours per scene [registry H-7, H-13, H-15, H-16] |
| Use its default depth prior (Depth-Anything-V2-Large) anywhere on a commercial track | **NO** | the weights are CC-BY-NC-4.0 while the code is Apache-2.0 [registry H-6, H-8] |
| Port its three loss terms into our own gsplat-based 2-camera fit, with a permissively licensed prior | **YES, as experiment E-06** (after E-01) | it targets exactly the sparse-view regime our kit lives in; evidence is on static scenes only [registry H-10, H-13] |
| Static studio / background reconstruction for compositing avatars into scenes | **YES, candidate** | the closest match to what the paper validated (static indoor); needs a metric scale anchor [registry H-11, H-15] |
| Rectified depth + normals as extra per-frame dataset channels | **HOLD** | only after E-06 passes and only with a clean-prior provenance field [registry H-8, H-9] |
| Treat it as a competitor or replacement for avatar methods | **N/A** | it supervises geometry; it produces no avatar, no motion model, no 4D [registry H-13] |

## 2. What it is (vendor-verified)

- Paper: AffineGS, arXiv 2611.07709, NeurIPS 2026 poster; Tse, Peng and Chen at dConstruct Robotics, Yao at NUS [registry H-1, H-2].
- Its finding: depth-guided 3DGS methods align a monocular depth map with one global scale-and-shift, but "modern monocular depth networks systematically exhibit region-dependent, piecewise-linear distortions", so the single transform warps the scene [registry H-3].
- Its fix is three loss terms added to the photometric loss: (1) per-pixel scale and shift fields obtained by comparing gradients of the monocular disparity with the rendered depth, then smoothed with an edge-aware guided filter; (2) a normal-aware term tying the rectified disparity to the rendered normals; (3) a global metric anchor on depth [registry H-4].
- The Gaussian representation and densification are untouched, so the gain comes from supervision alone [registry H-5]. That makes it a loss we can port, not a model we must adopt.
- Prior: Depth-Anything-V2-Large for the headline numbers; MoGe and Metric3D-V2 are analysed in an appendix [registry H-8].

## 3. What the evidence supports — and what it does not

- **Sparse views (the regime of our 2-camera kit [registry N-5]):** on NeRF-LLFF with 2–5 training views, mean PSNR is 18.79 / 20.95 / 22.37 / 23.32 dB against 17.31 / 18.75 / 20.06 / 20.78 dB for the global-alignment baseline DRGS [registry H-10]. LLFF is forward-facing, static, and contains no people.
- **Indoor depth and meshes:** over six MuSHRoom scenes the depth error (Abs Rel) is 0.055, versus 0.354 for global alignment and 0.023 for sensor depth — about 6.4× better than the former, about 2.4× worse than the latter [registry H-11 \| estimate: ratios of the README rows].
- **It does not always win:** on ScanNet++ its F-score is 0.321 against 0.766 for the sensor-supervised baseline [registry H-12].
- **Never tested on motion, video or people:** all evaluated data are static indoor or forward-facing scenes [registry H-13].
- **Its own stated limits:** quality is bounded by the depth prior (transparent surfaces, mirrors, domain gaps); the mesh has no absolute physical scale; the default loss weights are too strong for some scenes [registry H-15].
- **"Sensor-free" means sensor-free supervision:** the initial Gaussians are still seeded from back-projected sensor depth [registry H-14]. With two cameras the equivalent is a sparse SfM cloud — the very initialisation the paper says collapses competing methods at two views [hypothesis | run the init ablation inside E-06].

## 4. Licensing — clean code, tainted default prior

This is LEGAL-LICENSING §1 detection rule 2 (code and weights diverge) again, in a new place.

| Component | License | Commercial track? | Evidence |
|---|---|---|---|
| AffineGS code | Apache-2.0 | ✅ | [registry H-6] |
| Nerfstudio, gsplat, DN-Splatter (what it builds on) | Apache-2.0 — **no Inria rasterizer in the stack**, so it passes LEGAL-LICENSING §1 rule 1 | ✅ | [registry H-7, H-17] |
| Depth-Anything-V2-**Large** weights (the prior behind every headline number) | CC-BY-NC-4.0 | ❌ | [registry H-8] |
| Depth-Anything-V2-**Small** weights | Apache-2.0 | ✅ candidate | [registry H-8] |
| Depth Anything 3: Base, Small, Metric-Large, Mono-Large | Apache 2.0 (Large/Giant main series are CC BY-NC 4.0); predicts depth, not disparity, so the affine model needs re-deriving | ✅ candidate | [registry H-9] |
| MoGe-2 (code and the `vitl-normal` checkpoint) | MIT (code per README; weights per the Hugging Face tag only) | ✅ candidate, weights text still to read | [registry H-9] |
| Video-Depth-Anything-**Small** (temporally consistent) | Apache-2.0 (Base/Large are CC-BY-NC-4.0) | ✅ candidate | [registry H-9] |
| The paper's text and figures | CC BY-NC-SA 4.0 (arXiv licence label) — ideas are free, figures are not; reimplement from the equations | n/a | [registry H-1] |

**Policy until counsel answers:** depth maps generated by CC-BY-NC weights, and any splats or checkpoints fitted against them, are treated as NC-tainted under the wall rule [LEGAL-LICENSING §2 \| registry N-12]. Whether model *outputs* are restricted by a weights licence at all is a legal question, not a technical one [hypothesis | add to the single counsel review that LEGAL-LICENSING §4 already requires]. A commercially clean dataset cannot carry labels produced with an NC model — this is the repo's whole market position [registry C4-10b].

## 5. Where it could serve us (hypotheses, ranked)

1. **The 2-camera per-frame fit (Variant 1, clean sub-path).** Two GoPros is the extreme-sparsity regime the paper targets [registry N-5, H-10]. Without a green screen the subject and the background sit in different depth regimes, which is the situation an edge-aware field is built for and a single global alignment must compromise on [hypothesis | E-06 arm 2 vs arm 1]. Open risk: our baseline is wide and wraps a person, LLFF's is narrow and forward-facing.
2. **Static studio or environment capture** for placing an avatar into a scene: static, indoor, scene-scale — the validated regime — with a mesh available for occlusion and placement [registry H-11]. Needs a scale anchor (calibration target or known subject height) because the method returns none [registry H-15].
3. **Dataset channels (Variant 2b/3):** per-frame depth and normal maps shipped beside the RGB views would differentiate a clean dataset from the NC academic ones [registry C4-10b] [hypothesis | only if a clean prior matches the large-prior quality in E-06].
4. **Temporal use: nothing is offered.** The method is per-frame and static; a per-image prior flickers between frames. A temporally consistent prior (Video-Depth-Anything-Small) is the obvious thing to test, and the loss would have to live inside one 4D optimisation, not run per frame [hypothesis | E-06 temporal probe].

## 6. Why the upstream repository cannot simply be adopted

- Toolchain: Python 3.8, PyTorch 2.1.2, CUDA 12.1, gsplat 1.0.0, Nerfstudio 1.1.3 [registry H-7]. Our RTX 6000 Pro is sm_120, where stock toolchains emit sm_90 kernels [registry C3-8]; expect the repository to need the cloud fallback or a port [estimate | PyTorch 2.1.2 wheels predate Blackwell].
- Cost: about 1.5–2 hours per static scene on one RTX 3090 [registry H-16]; per frame this is impossible for a 4D clip.
- Per-scene tuning: guided-filter radius and epsilon and the loss weights differ per scene, and two scenes needed weaker weights to avoid floaters [registry H-15, H-18]. That collides with VARIANT-1 §5 acceptance criterion 1 (zero manual interventions) [hypothesis | E-06 counts the retunes our takes need].
- Resolution coupling: the filter settings were tuned for roughly 960×720 frames [registry H-18]; GoPro frames are larger, so the radius must be re-derived [estimate | radius is in pixels].
- Young repository: two commits (2026-11-05 and 2026-11-06), no releases, no open issues [registry H-6, H-7].

## 7. Next step

Experiment **E-06** in [MVP-0-EXPERIMENT.md](MVP-0-EXPERIMENT.md): same take as E-01, three arms (no prior, global alignment, spatially varying), NC-track diagnostic with the large prior against two clean priors, plus a half-day temporal probe. Until E-06 reports, no README, landing or dataset text may claim AffineGS-derived geometry [LANDING-SYNC.md \| registry N-6].
