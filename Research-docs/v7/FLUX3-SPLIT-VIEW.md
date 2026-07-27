# FLUX 3 "split view" — capability assessment for the capture → 4DGS pipeline

> Part of Research-docs v7. Research pass 2026-07-27, four days after the FLUX 3 announcement of 2026-07-23 [registry F-1]. Evidence is therefore thin by construction: the model is gated, unreleased and unbenchmarked. Every claim below carries a status tag; verdicts live in [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) section F.

## 1. Verdict in one table

| Pipeline stage | FLUX 3 today | Why |
|---|---|---|
| Generating extra camera angles fed into 4DGS fitting | **NO** | no camera-pose conditioning exists, no geometric-consistency evidence exists [registry F-7, F-12] |
| Back-filling the subject's unseen side | **NO** | same, plus it contradicts the product definition — faithful capture, not invention [registry N-2] |
| Pseudo-ground-truth for training (Variant 2) | **NO** | contractually blocked before it is technically judged [registry F-15, F-16] |
| Uploading our captured footage into it (any purpose) | **NO — hard stop** | the API takes a perpetual, sublicensable licence over Inputs, including training rights [registry F-17] |
| Pre-visualisation, storyboarding, camera blocking from text | **YES, with rules** | §7 |
| Marketing / trailer material with native audio | **YES, with rules** | §7 |

The short version: **"split view" is not a FLUX 3 feature — it is a user's prompt.** Even if it were a feature, licensing would block the useful applications before quality became the question.

## 2. What FLUX 3 actually is (vendor-verified)

- One multimodal model over images, video and audio: "FLUX 3 is our new multimodal foundation model. It jointly learns from images, videos, and audio within a unified architecture" [verified | https://bfl.ai/blog/flux-3 | as-of 2026-07-27 | "FLUX 3 is our new multimodal foundation model. It jointly learns from images, videos, and audio within a unified architecture"].
- Video length ceiling: "FLUX 3 can create highly diverse videos with audio up to 20 seconds in length in a single generation" [verified | https://bfl.ai/blog/flux-3 | as-of 2026-07-27 | "up to 20 seconds in length in a single generation"]. Longer content is handled by "Agentic chaining of individual clips into longer, multi-shot sequences" — sequential shots, not simultaneous views [verified | https://bfl.ai/blog/flux-3 | as-of 2026-07-27 | "Agentic chaining of individual clips into longer, multi-shot sequences."].
- The only resolution figure BFL states is an evaluation configuration, not a product spec: "For the preliminary analysis below, we generated 10-second text-to-video clips in 720p with audio" [verified | https://bfl.ai/blog/flux-3 | as-of 2026-07-27 | "we generated 10-second text-to-video clips in 720p with audio"]. No fps, codec, bit depth or seed-determinism figures are published anywhere official [registry F-6].
- Status: early access by application; results are explicitly provisional — "these results are preliminary, and we expect further improvements during the early access phase" [verified | https://bfl.ai/blog/flux-3 | as-of 2026-07-27 | "these results are preliminary, and we expect further improvements during the early access phase"]. An open-weight tier is announced but undated and unlicensed: "Open-weight access to a multimodal backbone" ("FLUX 3 Dev") [verified | https://bfl.ai/blog/flux-3 | as-of 2026-07-27 | "Open-weight access to a multimodal backbone"].

## 3. What "split view" really is

The viral demo is a single split-screen clip of a jumping cat, generated from one text prompt by an X user and amplified by BFL's co-founder with a two-word reaction, "very cool 🤓" [verified | https://digg.com/tech/bx4xbvj1 | as-of 2026-07-27 | "very cool" — Robin Rombach, 3:49 AM · Jul 24, 2026].

The decisive detail, read directly from the source post: the properties everyone quotes as FLUX 3's capability are **instructions inside the prompt**:

> "Split-screen test on Flux 3! Prompt : Split-screen video showing the same real-time action from two different camera angles … **Both sides must be precisely synchronized, showing the exact same event from different angles with consistent lighting, physics, and timing.**"
> [verified | https://x.com/umesh_ai/status/2080585387144274290 | as-of 2026-07-27 | "Both sides must be precisely synchronized, showing the exact same event from different angles with consistent lighting, physics, and timing." — post title/meta, read first-hand]

A request is not a measurement. The aggregator that spread the story reports sentiment, not evaluation: its own byline says the read is "Based on 32 visible X reactions from 21 accounts" [verified | https://digg.com/tech/bx4xbvj1 | as-of 2026-07-27 | "Based on 32 visible X reactions from 21 accounts."].

On the vendor side there is no such feature at all: no official BFL surface names split view, split screen, multi-view, multi-angle, viewpoint, orbit or camera control for FLUX 3 [not-found-as-of 2026-07-27 | queries: full-text grep of bfl.ai/blog/flux-3, bfl.ai/models/flux-3, bfl.ai/announcements/flux-3, docs.bfl.ai, api.bfl.ai/openapi.json for "camera|angle|multi-view|split|viewpoint|360|orbit"]. What BFL does claim is appearance-level: "visual references help ensure that the characters remain consistent across all scenes" [verified | https://bfl.ai/blog/flux-3 | as-of 2026-07-27 | "visual references help ensure that the characters remain consistent across all scenes"] — identity matching, not multi-view photometric or geometric agreement.

## 4. Stability evidence — what exists, what does not

**Does not exist:** any geometric (3D-metric) consistency evidence for FLUX 3 — no camera-pose recovery test, no reprojection error, no reconstruction benchmark, published by BFL or anyone else between launch and this pass [not-found-as-of 2026-07-27 | queries: "FLUX 3" + COLMAP / photogrammetry / gaussian splatting / novel view synthesis / camera pose / 3D reconstruction / multi-view consistency].

**Exists, and points the wrong way:** a hands-on comparison against Seedance 2.0 and LTX 2.3 reports FLUX 3 losing specifically on character-reference preservation, i.e. identity drift [reported | third-party hands-on write-ups collected 2026-07-27; no primary vendor confirmation]; a published prompting guide advises restricting each generation to a single camera concept and flags multi-location prompts as risky [reported | prompting guides published during launch week]. Both are consistent with a model whose "views" are independently imagined rather than jointly constrained.

**Our own observation, deliberately not upgraded:** inspecting the demo's poster frame suggests the two halves disagree about the room's geometry (the shelf differs in depth between panels) [hypothesis | single-frame eyeball inspection is an observation, not a measurement — see the decisive test below].

**The decisive test, cheap, not yet runnable:** pull the frames of a split-view generation and run COLMAP/SfM across the two halves. If the panels depict one 3D scene, feature matching finds a consistent two-camera solution; if they are two independent hallucinations, reconstruction fails or yields incoherent poses. This costs an afternoon and settles the question empirically — it cannot be run today because access is gated [registry F-5].

## 5. Why this class of tool is structurally unsafe as 4DGS input

Not a FLUX-specific complaint — it is the documented failure mode of video generators without a 3D substrate. NVIDIA's own GEN3C paper motivates its 3D cache precisely by this: prior video models "tend to leverage little 3D information, leading to inconsistencies, such as objects popping in and out of existence" [verified | https://arxiv.org/abs/2503.03751 | as-of 2026-07-27 | "they tend to leverage little 3D information, leading to inconsistencies, such as objects popping in and out of existence"].

Downstream that becomes: SfM finding no consistent correspondences across generated views, scale ambiguity between independently sampled views, identity drift and temporal flicker in the fitted splats. v6 already recorded the same conclusion from the other direction: "Temporal stability across generated views — any diffusion-densified path flickers" [internal | v6/6.16 §16.4], and DimensionX was flagged there as "research toy, not core" for exactly this reason [internal | v6/6.16 §16.2 Path B].

And the product-level objection outranks all of it: the owner's definition is faithful 4DGS of a real captured performance [registry N-2]. Generated angles are, by construction, invention. They may be legitimate in a clearly-labelled generative product line; they are illegitimate inside a deliverable sold as a faithful capture.

## 6. Legal and licensing blockers (independent of quality)

1. **Nothing to integrate against.** No FLUX 3 weights are published, no FLUX 3 licence text exists, no API endpoint carries it, and no price is published; access is an application form [registry F-3, F-4, F-13, F-14].
2. **Outputs may not train competing models.** The consumer terms state "You may not use Output to train, distill or fine tune any other AI models that compete with the Flux AI model(s)" [verified | https://bfl.ai/legal/terms-of-service | as-of 2026-07-27 | "You may not use Output to train, distill or fine tune any other AI models that compete with the Flux AI model(s)."]; the developer terms use a wider object — "competes with us or any of our products or services, including using any Output to train, distill or fine tune any other AI models" [verified | https://bfl.ai/legal/developer-terms-of-service | as-of 2026-07-27 | "competes with us or any of our products or services, including using any Output to train, distill or fine tune any other AI models"]. Both are competition-qualified; the risk is that a 4D human generative model plausibly falls inside "competes with … our products or services". That BFL sells a separate Synthetic Data licence to grant training rights confirms the default grant does not include them [registry F-16].
3. **The input licence is the harder blocker.** API use grants BFL a perpetual, irrevocable, sublicensable licence over Inputs as well as Outputs, expressly including training rights [registry F-17]. Uploading a client's captured performance — a real, identifiable person under signed release to *us* — would breach both the confidentiality we owe them and the GDPR basis under which they consented. This alone removes FLUX 3 from any workflow that touches capture footage, regardless of geometry.
4. **Open question worth a written answer:** does fitting Gaussians count as "training another AI model" under those terms? Splat optimisation is gradient descent over parameters, and the wording is broad enough to argue either way [hypothesis | obtain written clarification or the Synthetic Data licence before relying on any FLUX output in the pipeline].
5. **EU AI Act Article 50 lands in days.** "Article 50 of the AI Act applies as from 2 August 2026" [verified | https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act | as-of 2026-07-27 | "Article 50 of the AI Act applies as from 2 August 2026."], requiring that "AI-generated or manipulated content are marked in a machine-readable format and detectable as artificially generated or manipulated" [verified | https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act | as-of 2026-07-27 | "AI-generated or manipulated content are marked in a machine-readable format and detectable as artificially generated or manipulated."] and that "Deployers must disclose deepfake content to a natural person upon first exposure at the latest" [verified | https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act | as-of 2026-07-27 | "Deployers must disclose deepfake content to a natural person upon first exposure at the latest."]. A deliverable of a real person containing generated views is squarely in scope — a second, regulatory reason to keep generated pixels out of "faithful capture" products.
6. **Provenance policy is an unfilled hole.** Whether FLUX 3 outputs carry C2PA metadata or pixel watermarks — and what happens when such marks propagate into, or are stripped from, a capture deliverable — is unresolved [not-found-as-of 2026-07-27 | queries: docs.bfl.ai full text for C2PA/watermark; BFL legal pages].
7. **Procurement hazard.** Third-party resellers already advertise "FLUX 3 API" with hard specs (resolutions, fps) that BFL has never published [reported | reseller sites collected 2026-07-27]. Any purchase outside bfl.ai should be treated as unverified until BFL's own surface lists it.

## 7. Where FLUX 3 may be used today — and the rules

Allowed, once access lands: pre-visualisation and camera blocking generated purely from text before a shoot; storyboards and pitch clips; marketing and trailer material, where its native audio is a genuine advantage; lighting and look-dev references.

Rules, non-negotiable: never feed client or subject capture footage into it (§6.3); keep outputs in a separate asset bucket that never enters a reconstruction or training path; label them synthetic and keep the labelling through delivery (§6.5); never quote a generated frame as a product result.

## 8. What to use instead, when the job really is "more angles"

| Tool | What it is | Licence/availability status |
|---|---|---|
| Uni3C (Alibaba DAMO) | camera + human-motion control with point-cloud conditioning, closest match to human capture | Apache-2.0, weights published (a ControlNet checkpoint, not a full standalone model) [registry F-19] |
| GEN3C (NVIDIA) | camera-controlled video with an explicit 3D cache — built to avoid the failure in §5 | source released; model licence must be read before commercial use [registry F-20] |
| Diffuman4D (ZJU3DV) | sparse-view human 4D novel-view synthesis, expands 4 cameras to 44 | code and models released; 4DGS reconstruction scripts are a future-tense promise, only 3DGS is documented [registry F-21] |
| CAT4D-class (Google DeepMind) | monocular video → multi-view video → deformable 3DGS, the canonical shape of this pipeline | research; no public weights [registry F-22] |
| MV-Performer, Flex4DHuman | human-specific 360° multi-view diffusion; the latter is already our Variant 2 backbone plan | research-gated / replication path [registry N-7] |

Ranking for our job: real cameras first, purpose-built camera-conditioned models second, general video generators nowhere near the reconstruction path.

## 9. What would change this verdict

Re-open this file if any of these becomes true: (a) FLUX 3 Dev weights ship with a licence permitting commercial use and training [registry F-4]; (b) BFL exposes camera-pose or extrinsics conditioning as an API feature [registry F-7]; (c) anyone publishes a geometric-consistency evaluation — pose recovery, reprojection error, or a successful reconstruction from generated views [registry F-12]; (d) we obtain access and the COLMAP test in §4 succeeds on split-view output. Until then FLUX 3 is a previz and marketing tool for this project, and a watch item — not a pipeline component.
