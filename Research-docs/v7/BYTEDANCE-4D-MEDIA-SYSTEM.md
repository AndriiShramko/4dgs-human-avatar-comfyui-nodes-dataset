# ByteDance Multimedia Lab's 4D media system — competitive pass (2026-09-05)

> Part of Research-docs v7. Triggered by the owner sharing [radiancefields.com's coverage](https://radiancefields.com/bytedance-multimedia-lab-builds-a-4d-media-system-for-live-and-on-demand-6dof-video) of ByteDance Multimedia Lab's "4D media system" for live and on-demand 6DoF video. Companion to [6.22 — 4C4D and 4DV.ai/FreeTimeGS competitive landscape](../v6/6.22-4c4d-and-4dvai-competitive-landscape.md) (v6, audited); kept as a separate v7 file rather than a Part C there, per the v7 rule that audited v6 files stay unchanged and get pointer notices only.

## 0. Verification-method caveat (read this before trusting any tag below)

In this session, direct `WebFetch` was **`EGRESS_BLOCKED` by the network proxy** for `radiancefields.com`, `arxiv.org`, `huggingface.co`, `dl.acm.org`, `researchgate.net`, `nowheretrix.github.io`, and `webcache.googleusercontent.com` — every primary source for the radiancefields article, the LiveGS/Director papers, and the DualGS project page. Only `github.com` was directly fetchable this pass (confirmed working, see §3–4).

This document therefore uses the v7 bracket-tag taxonomy with one explicit, deliberate extension: claims sourced only from a WebSearch-indexed snippet of a blocked domain (not a raw page read) are tagged `[inherited-unverified | WebSearch snippet only, <domain> EGRESS_BLOCKED this session]` — repurposing the existing `inherited-unverified` keyword (originally scoped to legacy v5/v6 claims) for the same underlying meaning: *secondhand, not confirmed against the primary source in this pass*. Claims where `github.com` fetched successfully carry a normal `[verified | url | as-of date | "quote"]` tag. **Before citing any `inherited-unverified` line below in a pitch, budget, or legal decision, re-fetch the URL directly** — this is the same discipline §22.0 of 6.22 applied when arxiv.org was blocked there. <!-- lint-skip -->

---

## 1. What was actually announced

The article's own framing (WebSearch-indexed, radiancefields.com; not read first-hand this pass): ByteDance MultiMedia Lab presented an "intelligent 3D video stack" bundling three components — **high-fidelity multi-camera 4DGS**, **real-time LiveGS**, and **monocular conversion for existing 2D content** — under a "4D media system for live and on-demand 6DoF video" [inherited-unverified | WebSearch snippet only, radiancefields.com EGRESS_BLOCKED this session]. 6DoF here means the standard volumetric-video sense already used elsewhere in this project: free choice of viewing angle *and* position (not just head rotation), demoed against a sports-broadcast use case [inherited-unverified | WebSearch snippet only, radiancefields.com EGRESS_BLOCKED this session].

This is a **marketing/demo bundling of at least two, probably three, separately-published research lines** by an overlapping ByteDance + ShanghaiTech (HiFi-Human lab, Prof. Lan Xu) + Max-Planck-Institute-for-Informatics author group — not one paper. §2 lays out the lineage; §3–5 take the three named components in turn.

## 2. The research lineage behind the branding (this pass's main finding)

Cross-referencing author lists across five independent papers (WebSearch synthesis, cross-checked directly on GitHub for two of them — see §3) shows one continuous, multi-year collaboration, not disconnected work:

| Paper | Venue / arXiv | Key authors | Institutions |
|---|---|---|---|
| HiFi4G | (predecessor, not re-verified this pass) | Yuheng Jiang, Lan Xu et al. | ShanghaiTech |
| DualGS | SIGGRAPH Asia 2024, arXiv 2409.08353 | Yuheng Jiang, Zhehao Shen, Yu Hong, Chengcheng Guo, Yize Wu, **Yingliang Zhang**, Jingyi Yu, Lan Xu | ShanghaiTech [verified | https://github.com/HiFi-Human/DualGS | as-of 2026-09-05 | repo fetched directly, see §3] |
| Director | arXiv 2604.01678 | Yuheng Jiang, Yiwen Cai, Zihao Wang, Yize Wu, Sicheng Li, Zhuo Su, **Shaohui Jiao**, Lan Xu | ShanghaiTech **and ByteDance** [inherited-unverified | WebSearch snippet only, arxiv.org EGRESS_BLOCKED this session] |
| TaoGS | SIGGRAPH Asia 2025, arXiv 2509.07653 | Yuheng Jiang, Chengcheng Guo, Yize Wu, Yu Hong, Shengkun Zhu, Zhehao Shen, Yingliang Zhang, **Shaohui Jiao**, Zhuo Su, Lan Xu, Marc Habermann, Christian Theobalt | ShanghaiTech + ByteDance + MPI Informatics [verified | https://github.com/HiFi-Human/TaoGS | as-of 2026-09-05 | repo README fetched directly, see §3] |
| LiveGS | SIGGRAPH 2025 Emerging Technologies, DOI 10.1145/3721257.3734033 | Yuzhong Chen, Yuqin Liang, Zihao Wang, Danying Wang, Cong Xie, **Shaohui Jiao**, Li Zhang | ByteDance (+ possibly ShanghaiTech) [inherited-unverified | WebSearch snippet only, dl.acm.org EGRESS_BLOCKED this session] |

**Shaohui Jiao** (PhD, Institute of Software, Chinese Academy of Sciences) is confirmed as a ByteDance researcher working on "next-generation 3D video including free-viewpoint video, volumetric video, and AI-generated video content" [inherited-unverified | WebSearch snippet only, source is a bio page not re-fetched directly this session] and appears on **Director, TaoGS, and LiveGS** — the three most recent papers in the table. **Zihao Wang** and **Yize Wu** appear on both LiveGS and Director/TaoGS. This is strong circumstantial evidence that ByteDance's "4D media system" branding is a product-facing wrapper around this exact ShanghaiTech-HiFi-Human + ByteDance + (for TaoGS) MPI-Informatics pipeline, not a separate internal system [hypothesis | confirm by reading the radiancefields article and the LiveGS ACM page directly once dl.acm.org/radiancefields.com are reachable — neither was fetchable this session].

## 3. Component 1 — "high-fidelity multi-camera 4DGS": almost certainly this DualGS/TaoGS lineage

`DualGS` is **already tracked in this registry**: [registry LB2-5] (code exists, dual-licensed) and [registry LB2-13b] (≈350KB/frame compression, verified against arXiv directly in the prosecutor pass). This pass re-fetched `github.com/HiFi-Human/DualGS` directly and confirms LB2-5/LB2-13b still hold: license is **Gaussian Splatting Research License** (Inria-derived, for the inherited rasterizer code) **+ MIT** (for DualGS's own new components) [verified | https://github.com/HiFi-Human/DualGS | as-of 2026-09-05 | repo license section states original GS code under "Gaussian Splatting Research License", modifications under "MIT License"], and the repo ships a **complete, runnable** training/rendering pipeline (`train.py`, `render.py`, `metrics.py`, conda setup) — not a stub [verified | https://github.com/HiFi-Human/DualGS | as-of 2026-09-05 | repo directly browsed, files present].

**TaoGS is the newer, likely-successor system** — same lab, adds topology-aware tracking for scenes with topology changes (clothing, object handoff) that break DualGS's fixed-topology assumption. Directly fetched this pass:
- Title/venue: "Topology-Aware Optimization of Gaussian Primitives for Human-Centric Volumetric Videos," SIGGRAPH Asia 2025, arXiv 2509.07653 [verified | https://github.com/HiFi-Human/TaoGS | as-of 2026-09-05 | README fetched directly].
- **Code status: "Coming soon~" — NOT yet released**, despite the repo existing and being titled "Official implementation" [verified | https://github.com/HiFi-Human/TaoGS | as-of 2026-09-05 | README states "Coming soon~" for the implementation]. This is the same pattern already flagged for FreeTimeGS in [6.22 §22.12](../v6/6.22-4c4d-and-4dvai-competitive-landscape.md#2212-freetimegs-code--license-status) — a repo/README existing is not evidence that code exists. **Do not plan around TaoGS code being available.**
- No license file was visible in the repo at this pass [not-found-as-of 2026-09-05 | queries: direct browse of github.com/HiFi-Human/TaoGS root]; given it explicitly builds on DualGS and "Reperformer," expect the same Inria-derived non-commercial trap once code lands.

Whether the radiancefields article specifically names DualGS, TaoGS, or an unpublished internal variant as its "multi-camera 4DGS" component could not be confirmed (article itself unreadable this pass) [not-found-as-of 2026-09-05 | queries: radiancefields.com article text via WebSearch, 6 distinct query phrasings tried]. For our purposes the distinction barely matters: **both are the same research lineage, both carry the same license shape** (permissive new code, Inria-derived non-commercial rasterizer underneath).

## 4. Component 2 — real-time LiveGS

"LiveGS: Live Free-Viewpoint Video via High-Performance Gaussian Splatting for Mobile Devices," ACM SIGGRAPH 2025 Emerging Technologies, DOI 10.1145/3721257.3734033 [inherited-unverified | WebSearch snippet only, dl.acm.org EGRESS_BLOCKED this session]. Three claimed innovations, per WebSearch-indexed abstract text:

1. A feed-forward view-synthesis framework producing high-fidelity volumetric human 3DGS **without per-scene optimization** — i.e. generalizable/real-time, not a per-capture fit.
2. A compression scheme mapping temporal 3DGS onto **2D video planes**, so transmission can ride standard video codecs with minimal quality loss.
3. **Region-based Gaussian modeling** — coarse granularity on low-frequency regions, fine granularity on high-frequency regions — to cut compute for mobile playback.

Performance claims: **>30 FPS on an iPhone 15, <20 Mbps bitrate, <1 second latency** [inherited-unverified | WebSearch snippet only, dl.acm.org EGRESS_BLOCKED this session — re-verify these three figures against the ACM page or paper PDF before citing in any external material].

**No public code repository was found for LiveGS.** A direct fetch of ByteDance's own GitHub org confirms **zero repositories match "gaussian"** [verified | https://github.com/orgs/bytedance/repositories?q=gaussian | as-of 2026-09-05 | search returned "0 repositories... No repositories matched your search"], and no independent search turned up a third-party mirror or fork. Treat LiveGS as **closed / paper-only**.

## 5. Component 3 — "monocular conversion for existing 2D content": not identified

No specific paper, project name, or code could be traced for this component in any query tried this pass [not-found-as-of 2026-09-05 | queries: "ByteDance monocular video to volumetric 6DoF conversion 2D to 3D Gaussian paper 2026"; "ByteDance MultiMedia Lab monocular 4DGS 2D conversion"; "ByteDance Multimedia Lab LiveGS 4DGS system ACM Multimedia 2026 demo"]. It may be: an unpublished/product-internal ByteDance capability (plausible — the other two components are academic-paper-backed but this framing sounds product-facing, closer to "take a normal video and give it a navigable 3D feel"); a paper whose title doesn't contain any of the obvious keyword combinations tried; or a rebranding of an existing monocular-to-4DGS technique from elsewhere. **Flagging honestly rather than guessing** — if this component matters strategically, it needs a pass with unblocked access to the actual article and ByteDance's own research-publication pages.

## 6. Open-source status summary — answering "can we get any of this?"

| Component | Public code? | License (if any) | Where to look |
|---|---|---|---|
| LiveGS | **No** — not found anywhere, confirmed absent from ByteDance's own GitHub org | n/a | Paper only: ACM DOI 10.1145/3721257.3734033 |
| Multi-camera 4DGS (≈ DualGS) | **Yes, and already usable for research today** | Gaussian Splatting Research License (Inria, non-commercial) + MIT for new code [registry LB2-5] | `github.com/HiFi-Human/DualGS`, dataset `github.com/xyi1023/DualGS_Dataset` (non-commercial; commercial use needs permission from Yuheng Jiang / Lan Xu directly, per that repo's terms [inherited-unverified | WebSearch snippet only, github.com/xyi1023/DualGS_Dataset not re-fetched directly this pass]) |
| Multi-camera 4DGS (≈ TaoGS, likely successor) | **Repo exists, code not released ("Coming soon~")** | Unknown — no LICENSE file visible yet | `github.com/HiFi-Human/TaoGS` — watch for updates |
| Monocular conversion | **Unknown — component itself unidentified** | n/a | Not found this pass |

**Bottom line for "опенсорс или только теория": mixed, not a clean yes/no.** The system as *branded* in the article has no single open-source release — it's a demo bundle. But its most load-bearing named piece (multi-camera human 4DGS) traces to a lab that **does** publish working, cloneable code (DualGS, right now) and has a clear successor in the pipeline (TaoGS, code pending) — same license shape as everything else this project has already flagged as research-only-until-cleaned (§6.7's Clean-Room policy, [registry N-12]). LiveGS and the monocular-conversion piece are theory/paper-or-less, with no code trail at all.

**Concrete ways to get more, in order of effort:**
1. **Lowest effort, available now:** clone and read `HiFi-Human/DualGS` — it's real, runnable code, same lab. Use it to study the dual-Gaussian (motion/appearance) architecture, not to import into Track A (license, per §6.7).
2. **Watch, don't chase:** star/watch `HiFi-Human/TaoGS` for the "coming soon" code drop, and watch arXiv (2509.07653, 2604.01678) for revisions.
3. **Ask directly, precedent exists:** [6.22 §22.12](../v6/6.22-4c4d-and-4dvai-competitive-landscape.md#2212-freetimegs-code--license-status) already shows the move that works elsewhere in this exact research area — an open GitHub issue asking the authors about an open-source timeline (`zju3dv/EasyVolcap#47` for FreeTimeGS). The same move is available on `HiFi-Human/TaoGS` (ask about the "coming soon" code) and, if a repo ever appears, on any future LiveGS release.
4. **No realistic path to LiveGS or the monocular-conversion component specifically** beyond reading the paper (once available) and re-implementing from the method section — there is no code to request. A SIGGRAPH 2025 Emerging Technologies listing may carry author contact info if a direct research conversation is ever worth pursuing, but that is a cold-outreach, not a download.

## 7. Relevance to our project — where this actually changes anything

- **Compression (§6.6):** LiveGS's "map temporal 3D Gaussians onto 2D video planes, ride a standard codec" idea is a distinct approach from DualGS/QUEEN's direct Gaussian-attribute compression (already our reference class per the README's "≈0.35–0.7 MB/frame (DualGS, NVIDIA QUEEN)" target [registry LB2-13a, LB2-13b]). It's paper-only with no code, so it's a **candidate technique to study and clean-room-evaluate**, not a dependency — worth a line item in §6.6/§6.19 next time that module gets revisited, flagged as unverified-numbers/no-code.
- **Feed-forward / no-per-scene-optimization (§6.2):** LiveGS's headline claim (generalizable reconstruction, no per-capture fit) sits in the same research direction already tracked there (LRM+1-step refinement, DiffusionGS [registry LB1-6, LB1-7]) — another data point that the field is converging on feed-forward human reconstruction, reinforcing rather than changing that section's direction.
- **Kinematics / temporal coherence (§6.3):** DualGS's joint-Gaussian (motion) vs. skin-Gaussian (appearance) split is architecturally interesting for our temporal-regularizer work, alongside 4C4D's Neural Decaying Function already flagged in [6.22 §22.15](../v6/6.22-4c4d-and-4dvai-competitive-landscape.md#2215-relevance-to-our-project) as a clean-room candidate. Same treatment applies here: study the mechanism from the paper, do not import DualGS/TaoGS code, for the same Inria-license reason.
- **Legal strategy (§6.7):** No new trap, but a **reconfirmation** of the existing pattern — permissive-looking new code (MIT) sitting on top of a non-commercial Inria rasterizer [registry LB2-4, LB2-5, N-12]. TaoGS is very likely to repeat this exactly once its code lands; assume non-commercial until proven otherwise.
- **Product scoping ([registry N-1], [registry N-2]):** ByteDance shipping "multi-camera 4DGS" (faithful capture) and "monocular conversion" (generative/hallucinated) as two *separate* named components — not one model — is an external, well-funded validation of this project's own owner-decision to keep faithful-capture and avatar/monocular-hallucination as clearly separated tracks rather than blurring them into one product claim.
- **Competitive landscape (§6.22):** ByteDance is a materially different threat profile than 4C4D (academic, license-trapped) or 4DV.ai (funded but small, opaque ToS): it has **CapCut/TikTok-scale distribution**, meaning a shipped consumer feature could reach end-users directly rather than needing a B2B sales motion. Nothing in this pass suggests a shipped ByteDance consumer product yet (LiveGS is an Emerging Technologies demo, not a product announcement), so **no roadmap change is indicated today** — but this is the one competitor in the current landscape worth a standing watch item given its distribution reach, more so than 4DV.ai's enterprise/XR focus.

## Sources

**Directly fetched this pass (primary):**
1. `github.com/HiFi-Human/DualGS` — README, license section, file listing
2. `github.com/HiFi-Human/TaoGS` — README, code status, author list
3. `github.com/orgs/bytedance/repositories?q=gaussian` — live search, 0 results

**WebSearch-indexed only, primary source blocked this session (`inherited-unverified` throughout):**
4. radiancefields.com article (URL in header) — EGRESS_BLOCKED, never read directly
5. `dl.acm.org/doi/10.1145/3721257.3734033` (LiveGS) — EGRESS_BLOCKED
6. `arxiv.org/abs/2604.01678` (Director) — EGRESS_BLOCKED
7. `arxiv.org/abs/2509.07653` (TaoGS paper, distinct from its GitHub repo) — EGRESS_BLOCKED
8. `github.com/xyi1023/DualGS_Dataset` — not re-fetched this pass, cited via earlier registry knowledge + WebSearch
9. Shaohui Jiao ByteDance bio — via WebSearch synthesis, source page not re-fetched directly

**Internal:** [registry LB2-5, LB2-13b] · [6.22 competitive landscape](../v6/6.22-4c4d-and-4dvai-competitive-landscape.md) · [6.2 sparse-view](../v6/6.2-sparse-view.md) · [6.3 kinematics](../v6/6.3-kinematics.md) · [6.6 compression-deploy](../v6/6.6-compression-deploy.md) · [6.7 legal strategy](../v6/6.7-legal-strategy.md) · [CLAIMS_REGISTRY.md](CLAIMS_REGISTRY.md) Section BD
