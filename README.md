# Preparing to release the ultimate 4DGS module for ComfyUI. Take your seats.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Nodes-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Status](https://img.shields.io/badge/Status-Coming%20soon-00D4FF.svg)](https://comfyui-4dgs-volumetric-node.eu/)

**Faithful volumetric video of real performances — surround coverage from multi-camera capture, engineered for measurable stability.**

---

<p align="center">
  <a href="https://comfyui-4dgs-volumetric-node.eu/#waitlist">
    <img src="https://comfyui-4dgs-volumetric-node.eu/video/hero-poster.jpg" alt="4DGS Volumetric — Shramko 4D" width="720" />
  </a>
</p>

<p align="center">
  <strong>
    <a href="https://comfyui-4dgs-volumetric-node.eu/">Watch the demo & join the waitlist →</a>
  </strong>
</p>

---

## There’s nothing to run here yet — and that’s why we need you

This repo will host **ComfyUI nodes** (Shramko-Volumetric-Bridge) and **human scan datasets** for 4D Gaussian Splatting. The module and data are being prepared. **There is no downloadable build or dataset yet.** If you want to be first in line when we ship, add yourself to the **closed waitlist** on our landing page.

---

## We’re all tired of noisy, heavy 4DGS and non-commercial licenses

Most of today’s 4D Gaussian Splatting stack is research code under non-commercial licenses, with no stable end-to-end path from a real captured performance to a playable volumetric clip. Feed-forward single-image tools (e.g. Apple SHARP — a genuinely impressive *static* 3DGS generator whose released weights are research-only) solve a different problem: they generate a scene from one photo, they don’t faithfully replay a real performance. We’re building the missing piece: **faithful 4DGS of a real captured performance**, delivered through ComfyUI, on a commercially clean stack.

**Our target characteristics** (targets, not shipped results — derivations and verification status for every number live in [Research-docs/v7](Research-docs/v7/README.md)):

| Target | Honest fine print |
|--------|-------------------|
| Faithful surround view from multi-camera capture | single-camera captures are *front-faithful*; nobody can observe an unseen back — learned back-fill is a separate, clearly-labeled research track |
| Measurable temporal stability | acceptance metrics (held-out-camera PSNR/LPIPS, flicker checks) are defined in Research-docs/v7 — no “absolute” claims |
| Commercially clean pipeline | built on Apache-2.0 components (gsplat, Wan 2.1) with a strict license wall; details in [LEGAL-LICENSING](Research-docs/v7/LEGAL-LICENSING.md) |
| Compact temporal files | published research codecs already reach ≈0.35–0.7 MB/frame (DualGS, NVIDIA QUEEN) — our target is that class of size on a commercially clean codepath |

---

## Why the waitlist?

Building this module — and capturing, processing and legally clearing large volumes of volumetric data — is a huge investment (the honest current status of data and code is tracked in [DATASET-STATUS](Research-docs/v7/DATASET-STATUS.md)). Before the final push, we need to see real interest. **Your signup is a signal** that we’re building what the market actually needs. We also need to know: will you run nodes locally, or do you need a cloud API? Leave your details on the landing page — when we’re ready, you’ll get access first.

---

## Reserve early access. Launch is around the corner.

Your waitlist signup is a vote for this product and a ticket to the front row. Everyone who joins at this stage gets **exclusive terms right after launch** — enough credits for plenty of tests.

<p align="center">
  <a href="https://comfyui-4dgs-volumetric-node.eu/#waitlist">
    <img src="https://img.shields.io/badge/Join_the_closed_Waitlist-00D4FF?style=for-the-badge&labelColor=0A0A0A" alt="Join the closed Waitlist" />
  </a>
</p>

**→ [Go to the landing page and add yourself to the waitlist](https://comfyui-4dgs-volumetric-node.eu/#waitlist)**

---

## What will be in this repo (when it’s ready)

- **ComfyUI nodes** — Shramko-Volumetric-Bridge: Ingest, Processing, Visualization, Export. 4DGS creation as intuitive as regular video generation.
- **Free dataset** — Human scan data for **non-commercial** use and research (CC BY-NC-4.0), hosted on Hugging Face.
- **Enterprise dataset** — planned: large-scale, consent-documented human capture for training your own AI models under a commercial license (production gates and scale are defined in [Research-docs/v7/VARIANT-3](Research-docs/v7/VARIANT-3-FLAGSHIP.md); current status: [DATASET-STATUS](Research-docs/v7/DATASET-STATUS.md)).

Until then, **star this repo** and **[join the waitlist](https://comfyui-4dgs-volumetric-node.eu/#waitlist)** so we can notify you at launch.

---

## Development & Research Documentation

Research and implementation docs (architecture, loss functions, kinematics, deployment, legal strategy, etc.) are in the **[Research-docs](Research-docs/)** folder. **Start from [Research-docs/v7](Research-docs/v7/README.md)** — the audited, three-variant plan where every claim carries a verification tag (see its [CLAIMS_REGISTRY](Research-docs/v7/CLAIMS_REGISTRY.md)). v5/v6 remain for provenance and implementation detail. *This documentation was produced with AI assistance; v7 claims are individually verified and tagged, earlier versions may contain inaccuracies.*

---

## Links

- **Landing page (demo, video, waitlist):** [comfyui-4dgs-volumetric-node.eu](https://comfyui-4dgs-volumetric-node.eu/)
- **Waitlist form:** [comfyui-4dgs-volumetric-node.eu/#waitlist](https://comfyui-4dgs-volumetric-node.eu/#waitlist)

**Andrii Shramko** · Engineer · Inventor · 4DGS Researcher · [LinkedIn](https://www.linkedin.com/in/andriishramko/) · [Book a video call](https://comfyui-4dgs-volumetric-node.eu/)

---

## Repository structure (for later)

```
├── Research-docs/       # Development & research documentation (v5, v6, v7 — start at v7)
├── comfyui_nodes/       # Shramko-Volumetric-Bridge nodes (when released)
├── models/              # Model weights & instructions
├── dataset/             # Free & enterprise dataset docs and scripts
├── docs/
└── examples/
```

**Licenses:** Code — [Apache 2.0](LICENSE). Free dataset — [CC BY-NC-4.0](DATASET_LICENSE_FREE.md). Enterprise — [see terms](DATASET_LICENSE_ENTERPRISE.md).

---

Development and research documentation in [Research-docs](Research-docs/) was produced with AI assistance and is shared for reference only; it may contain inaccuracies or be outdated.
