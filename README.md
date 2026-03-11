# Preparing to release the ultimate 4DGS module for ComfyUI. Take your seats.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Nodes-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Status](https://img.shields.io/badge/Status-Coming%20soon-00D4FF.svg)](https://comfyui-4dgs-volumetric-node.eu/)

**True 360°. Volumetric video generation without flicker or artifacts.**

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

This repo will host **ComfyUI nodes** (Shramko-Volumetric-Bridge) and **human scan datasets** for 4D Gaussian Splatting. Right now we’re in the final stretch: the module and data are being prepared. **There is no downloadable build or dataset yet.** If you want to be first in line when we ship, add yourself to the **closed waitlist** on our landing page.

---

## We’re all tired of noisy, heavy 4DGS and non-commercial licenses

You’ve probably already tried tools like **Apple SHARP**. It’s a cool toy, but not fit for real work: 10–20° parallax, the back of the subject ignored, temporal “boiling” in animations, and **commercial use strictly forbidden**. We need a tool that’s stable, great-looking, and fully legal for commercial work. That’s what we’re building.

| Apple SHARP | **Shramko-Human-4D** (ours) |
|-------------|------------------------------|
| Limited 10–20° parallax | **Full 360° volumetric view** |
| Ignores subject’s back | **Accurate back-side reconstruction** |
| Temporal “boiling” in animations | **Absolute temporal stability** |
| Commercial use forbidden | **100% legal for commercial use** |
| ~100 GB per minute of video | **Ultra-optimized — less than 1 GB per minute** |

---

## Why the waitlist?

Building this module and processing petabytes of volumetric data is a huge investment. Before the final push, we need to see real interest. **Your signup is a signal** that we’re building what the market actually needs. We also need to know: will you run nodes locally, or do you need a cloud API? Leave your details on the landing page — when we’re ready, you’ll get access first.

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
- **Enterprise dataset** — Large-scale scans (thousands of subjects, hundreds of TB) for training your own AI models, under a commercial license.

Until then, **star this repo** and **[join the waitlist](https://comfyui-4dgs-volumetric-node.eu/#waitlist)** so we can notify you at launch.

---

## Links

- **Landing page (demo, video, waitlist):** [comfyui-4dgs-volumetric-node.eu](https://comfyui-4dgs-volumetric-node.eu/)
- **Waitlist form:** [comfyui-4dgs-volumetric-node.eu/#waitlist](https://comfyui-4dgs-volumetric-node.eu/#waitlist)

**Andrii Shramko** · Engineer · Inventor · 4DGS Researcher · [LinkedIn](https://www.linkedin.com/in/andriishramko/) · [Book a video call](https://comfyui-4dgs-volumetric-node.eu/)

---

## Repository structure (for later)

```
├── comfyui_nodes/       # Shramko-Volumetric-Bridge nodes (when released)
├── models/              # Model weights & instructions
├── dataset/             # Free & enterprise dataset docs and scripts
├── docs/
└── examples/
```

**Licenses:** Code — [Apache 2.0](LICENSE). Free dataset — [CC BY-NC-4.0](DATASET_LICENSE_FREE.md). Enterprise — [see terms](DATASET_LICENSE_ENTERPRISE.md).
