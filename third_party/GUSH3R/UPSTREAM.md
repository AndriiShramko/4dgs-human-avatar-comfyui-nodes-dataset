# UPSTREAM — provenance of this vendored copy

**This directory is an archival preservation snapshot of a third-party research repository, kept for internal research on the Shramko-Human-4D project. It is NOT our code and is NOT covered by this repository's Apache-2.0 license.**

| Field | Value |
|---|---|
| Project | GUSH3R: Everyone Everywhere All at Once as Gaussians |
| Upstream repo | https://github.com/abkeito/GUSH3R |
| Snapshot commit | `bac8d88405ff62453b033c5a2b5709f42fcf50be` ("feat: finalizaion", 2026-07-07 14:26:22 +0900) |
| Snapshot taken | 2026-08-14 (git clone --depth 1, `.git` stripped) |
| Project page | https://abkeito.github.io/gush3r-page/ |
| Paper | arXiv:2607.05243 — https://arxiv.org/abs/2607.05243 |
| Weights | https://huggingface.co/abkeito/GUSH3R — single file `checkpoints/gush3r.pth`, **4.87 GB** (NOT mirrored here; too large for git and license-unclear) |
| Authors | Keito Abe, Kaede Shiohara (project lead), Takashi Otonari, Toshihiko Yamasaki — The University of Tokyo |

## Licensing status (important — read before using anything from here)

- Upstream root `LICENSE` file is **empty (0 bytes)** as of the snapshot commit — the authors have not granted an explicit license. Default copyright law applies: **all rights reserved by the authors** until clarified.
- Upstream README states the `src/` code is adopted from [Human3R](https://github.com/fanegg/Human3R) (MIT), **but** the bundled `src/croco/` carries its own `LICENSE`: CroCo, © Naver Corporation, **CC BY-NC-SA 4.0 — non-commercial only**. `src/dust3r/` derives from DUSt3R (Naver, also CC BY-NC-SA 4.0 upstream).
- SMPL / SMPL-X body models are NOT included upstream or here; they require separate registration and carry their own restrictive licenses.
- Practical conclusion: treat this snapshot as **research-only, non-commercial** material until the authors publish an explicit license. Do not merge any of this code into the commercial pipeline (see `Research-docs/v7/LEGAL-LICENSING.md` — "the wall").

This snapshot is kept in the spirit of a GitHub fork: to preserve publicly published research code against upstream deletion, with full attribution. If the upstream authors object to this copy, it will be removed on request (zmei116@gmail.com).

## Citation (from upstream README)

```bibtex
@article{abe2026gush3r,
  title   = {GUSH3R: Everyone Everywhere All at Once as Gaussians},
  author  = {Abe, Keito and Shiohara, Kaede and Otonari, Takashi and Yamasaki, Toshihiko},
  journal = {arXiv preprint arXiv:2607.05243},
  year    = {2026}
}
```

## Research notes

See `Research-docs/intake/2026-08-14-gush3r.md` for the intake analysis and open research questions on how GUSH3R could be used in Shramko-Human-4D.
