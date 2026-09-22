# James C. Kane's low-cost dynamic splats, 4DAnyone and ComfyUI-SplatKit — assessment for Shramko-Human-4D

> Part of Research-docs v7. Research pass 2026-09-22, triggered by a LinkedIn post in which James C. Kane (Director, Emerging Technology) announced that his 4DGS research won an NVIDIA GTC Berlin Golden Ticket. Code was read first-hand from shallow clones (commit hashes in §7); pages the sandbox proxy blocked (LinkedIn, X, jamesckane.com, arxiv.org) are marked as such. Verdicts live in [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) section K.

## 1. Verdict in one table

| Question | Answer | Why |
|---|---|---|
| Is Kane's GTC entry itself published as code? | **Not found** | his public GitHub has no 4DGS repository; the entry is described only in a LinkedIn post [registry K-1, K-2] |
| What is the traceable technical core of his "low-cost dynamic splats"? | **4DAnyone** (monocular video → many synthetic views → 4DGS) + his own **web 4DGS player** + SparkJS | 4DAnyone's authors credit him for building its 4DGS player [registry K-3]; his shipped web work runs on SparkJS [registry K-4] |
| Can 4DAnyone enter our *commercial* faithful-capture product? | **NO** | generated views are invention [registry N-2]; its checkpoint is trained on DNA-Rendering [registry N-3, K-7]; its default path has non-commercial parts [registry K-8] |
| Is it useful to us at all? | **YES — four concrete ways** | §4: benchmark baseline, labelled back-fill research track, technique donor for the replay fitter, web/codec design |
| Did anything change for our market position? | **YES — important** | ComfyUI-SplatKit already ships a video → 4D-splat-sequence workflow inside ComfyUI; our "empty niche" claim is out of date [registry K-10] |

## 2. What Kane did (what can be verified)

- The post (screenshot supplied by the owner, 2026-09-22) says the research was made with NVIDIA hardware, NVIDIA open models and World Labs' SparkJS, and frames it as "ongoing testing of low-cost dynamic gaussian splats". Which NVIDIA open models were used is not stated anywhere reachable [not-found-as-of 2026-09-22 | queries: LinkedIn/X/jamesckane.com blocked by proxy; web search "James C. Kane" + 4DGS/Lyra/Cosmos/GEN3C/NVIDIA].
- The screenshot shows a subject filmed at night with a fan of virtual camera frustums around them — the visual signature of a generated-view ring as in 4DAnyone's pipeline visualisation [hypothesis | confirm by asking him directly, see §6].
- Hard evidence of his 4DGS work: the 4DAnyone project page thanks "James C. Kane for building a 4DGS player for 4DAnyone" [registry K-3].
- His background is web delivery, not training: open-sourced SparkJS noir game "Shadow" (splat morph/dissolve effects in `src/vfx/`), a MediaPipe → VRM face-mocap web app, a Weta fire-sim port to WebGPU [registry K-4] [internal | clones listed in §7].

So the honest reading: **Kane's contribution is the delivery layer** (browser playback of dynamic splats, cheap pipeline assembly); the generative heavy lifting is 4DAnyone by Ant Research / ZJU / HKUST.

## 3. What the code actually does

### 3.1 4DAnyone (ant-research/4DAnyone, SIGGRAPH Asia 2026)

- Input: one casual video of one person, no camera calibration; output: dozens of synchronized generated videos around the subject plus `cameras.json`, then a separate 4DGS fit [registry K-5].
- Architecture: fine-tuned Wan2.2-TI2V-5B; 3D skeleton from GVHMR (or SAM 3D Body in forks) rendered as a **depth-buffered** skeleton per target view; **Reference Context Packing** (earlier generated views packed at 2×/4× lower token resolution, O(1) context) and **Target Context Routing** (view groups of four rotated during high-noise steps, fixed during the last 20% of steps) [registry K-6].
- 4DGS stage in the paper: FreeTimeGS, space-carving init from foreground masks, ≈30 min per 16-camera 121-frame clip on an RTX 4090 [registry K-6]. The public repo does not ship that stage yet — only a nerfstudio 3DGS path [registry K-5].
- Cost: ≈2.1–2.7 min for a 6-view orbit and ≈16 min for 24 views on an RTX 4090 with Turbo [registry K-5]. Training was 3 days on 128 H20-3E GPUs — replication is out of reach for us [registry K-6, N-4].
- Stated limits: loose garments generate inconsistently; a wrong HMR pose (en-pointe read as flat feet) is copied into every view [registry K-6].

### 3.2 ComfyUI-SplatKit (mickmumpitz) — the part that matters most for us

- MIT node pack with six `SplatKit/4DAnyone` nodes and nine `SplatKit/Splatting` nodes: Generate Views → Export Frameset → Train Sequence → Sequence Player [registry K-10].
- Its trainer is **per-frame 3DGS with warm start**, and every rule in it is documented as "paid for by a failed run": rebuild optimisers per frame, reopen densification without opacity reset, cap the gaussian count (Turbo input once reached 21.7 M gaussians), and — the key idea — **move the previous frame's gaussians rigidly with the body parts** (Kabsch per part, blended by proximity) before optimising the next frame; plus a geometric cleaner for dust, needles and off-silhouette splats [registry K-11].
- Honest self-assessment in its own docs: Windows only for the backend, ~24 GB VRAM, and "A full real-person video run and visual browser playback still require validation" [registry K-10].
- The Linux port (RunPod template) measured an RTX 4090 OOM on 16 views and success on an RTX 5090 [registry K-12].

### 3.3 SparkJS (World Labs, MIT)

Kane's rendering stack. Relevant for us: `SplatSkinning` (experimental dual-quaternion skeletal animation, up to 4 bones per splat), `SplatPager` (streaming), SPZ I/O [registry K-4] [internal | spark clone a0cace3, `src/SplatSkinning.ts` header].

## 4. How this helps Shramko-Human-4D

**4.1 A free, honest benchmark baseline (MVP-0 add-on).** Film one take with our 2× GoPro [registry N-5]. Feed camera A alone into 4DAnyone; render its generated view at camera B's pose; compare with the real camera B footage (PSNR/LPIPS on the person mask). That number is the "cost of hallucination" and is exactly the argument for faithful multi-camera capture on the landing page — measured, not asserted [hypothesis | ≈1 GPU-hour on a 32 GB card or rented 5090; requires the non-Turbo path for anything published, §5].

**4.2 Technique donor for Variant 1 (commercially clean re-implementation of ideas, no code or weights copied) [internal | LEGAL-LICENSING.md license wall].**
- Skeleton-carried warm start (SplatKit) → directly applicable to our per-capture gsplat fitter; reduces motion streaks, which are our temporal-stability acceptance metric [hypothesis | A/B on one 2-cam take: flicker metric with/without carry].
- Gaussian cap + post-frame cleaner (dust / needles / off-silhouette) → same A/B.
- Space-carving init from masks → cheap visual-hull init with 2 cameras [hypothesis | compare against COLMAP-points init].
- Depth-buffered skeleton rendering → useful as our own debug overlay and as conditioning if Variant 2 ever trains a prior.

**4.3 Compact temporal files and web playback.** Per-frame PLY sequences are what SplatKit ships today (its player reads the saved PLY files) [registry K-10]; how Kane's player stores frames is not public [hypothesis | ask him, §6]. PLY-per-frame is the heavy format we promised to beat. Skeleton-bound gaussians enable a lighter design: one canonical splat + per-frame bone transforms (Spark's `SplatSkinning` already renders this in the browser) + sparse per-frame residuals for cloth and face [hypothesis | measure MB/frame vs PLY-seq on one take; compare with the ≈0.35–0.7 MB/frame codec class in README targets]. This is a natural collaboration point with Kane (§6).

**4.4 Labelled generative back-fill track.** For monocular or 2-camera captures, 4DAnyone is currently the strongest open candidate for the clearly-labelled back-fill research track. Its model was trained with 1, 4 and 8 source views packed into the reference context, but the public inference accepts one video only — adding our second real camera as a reference is an architecture-anticipated experiment, not a supported feature [registry K-6] [hypothesis | patch RCP inputs; expect needing a camera-to-skeleton alignment step]. Non-commercial and labelled synthetic by rule (§5).

**4.5 Competitive consequence.** Treat SplatKit as the reference competitor. Our differentiation must now be: faithful multi-camera replay (not generated), Linux + cloud API, temporal codec instead of PLY sequences, and measured stability — not "first 4DGS in ComfyUI".

## 5. License wall — where each piece may go

| Component | Status | Commercial track? |
|---|---|---|
| 4DAnyone code | Apache-2.0 [registry K-8] | ideas yes; the pipeline no (below) |
| 4DAnyone checkpoint | Apache-2.0, but trained on DNA-Rendering [registry K-7] | **NO** — owner ruling [registry N-3] |
| GVHMR (default pose path) | educational, research and non-profit only [registry K-8] | NO |
| Turbo LoRA | CC BY-NC-SA 4.0 [registry K-8] | NO — disable Turbo even for demos we sell |
| SAM 3D Body (fork path) | SAM License: commercial allowed, trade-control and litigation clauses [registry K-9] | allowed after legal read |
| ComfyUI-SplatKit | MIT [registry K-10] | ideas and interop yes |
| SparkJS | MIT [registry K-4] | yes |

Plus the product rule: generated pixels never enter a deliverable sold as faithful capture, and must be marked as synthetic (EU AI Act Art. 50) [registry N-2, F-18]; likeness consent must cover "views never filmed".

## 6. Actions

1. Reach out to James C. Kane — he will be at GTC Berlin, 20–22 Oct 2026 [registry K-13], a short trip from Poland; his attendance comes from his own post [registry K-1]. Offer: our faithful multi-camera captures for his player; his web-playback work for our landing/demo; joint write-up comparing generated vs real second view (4.1).
2. Add experiment 4.1 to [MVP-0-EXPERIMENT.md](MVP-0-EXPERIMENT.md) backlog; run once a 2-camera take exists [registry N-6].
3. Correct the "empty niche" wording in LEGAL-LICENSING and VARIANT-1 (done in this pass, pointing to K-10).
4. Watch items: 4DAnyone's promised open 4DGS stage; SplatKit's Linux support and ComfyUI Manager listing.

## 7. Provenance

Clones read on 2026-09-22 [internal | scratch clones, not vendored]: ant-research/4DAnyone e38f210 · mickmumpitz/ComfyUI-SplatKit 47c93ee · sparkjsdev/spark a0cace3 · jameskane05/shadow 0a818c5 · 4danyone/4danyone.github.io gh-pages (acknowledgements) · hmedj69-hub/runpod-4danyone-comfyui (README). Paper text read via the Hugging Face papers mirror of arXiv 2608.20335.
