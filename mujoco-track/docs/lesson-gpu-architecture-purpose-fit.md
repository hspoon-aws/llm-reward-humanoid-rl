# Lesson: GPU architecture vs. RT cores — why your simulator picks your GPU

**Date:** 2026-06-12 (MuJoCo track, dual-run A/B in flight)
**Where it came from:** untangling "RTX vs Blackwell" and "B200 vs the g6 RTX card" while
explaining the `nvidia-smi` C/G process types during the dual-track runs.
**Blog section:** "Choosing hardware for physical-AI" → the comparison that frames the whole
two-track project.

## TL;DR

"RTX vs Blackwell" is a category error — they're different axes. **Blackwell is an architecture
(generation); RTX is a feature brand (has RT cores).** The real question is *"does this
Blackwell chip have RT cores?"* — and the answer differs between the data-center B200 (no) and
workstation Blackwell (yes). The decisive, non-obvious lesson of this project:
**your simulator choice dictates your GPU class, not your budget or your compute needs.**
Isaac Sim needs RT cores → you must use an RT-core GPU (even an older/cheaper one). MuJoCo MJX
needs only compute → you're free to use the compute-optimal GPU.

## The two axes people conflate

| Axis | Examples | What it determines |
|---|---|---|
| **Architecture (generation)** | Blackwell, Hopper, Ada Lovelace | compute capability, CUDA version, raw speed, VRAM tech |
| **Has RT cores? (RTX brand)** | RTX-branded = yes; data-center = no | whether it can do hardware graphics / ray tracing at all |

These are independent. A chip can be newest-generation AND lack RT cores (B200). A chip can be
a generation behind BUT have RT cores (L40S).

## Blackwell comes in two flavors (the trap)

| | Data-center Blackwell | Workstation/consumer Blackwell |
|---|---|---|
| Examples | **B200**, B100, GB200 | RTX PRO 6000 Blackwell, RTX 50-series |
| RT cores | **No** | **Yes** |
| NVENC video encode | No | Yes |
| Display output | No | Yes |
| Branded "RTX"? | No | Yes |
| Built for | LLM train/infer, HPC — pure compute | graphics, rendering, simulation, viz |

So the **B200 is Blackwell but NOT RTX** — RT cores were stripped out to pack in more compute
and HBM for AI. When NVIDIA docs say "Isaac Sim supports Blackwell," they mean the **RTX**
Blackwell. True in the letter, false for the B200. (This is the exact trap
`../../docs/lesson-isaac-lab-bringup.md` learning 4 documented: the unwatched risk was not "is
Blackwell supported" but "is *data-center* Blackwell supported.")

## B200 vs the AWS g6/g6e RTX card — head to head

(AWS **g6** = NVIDIA **L4** 24 GB; **g6e** = **L40S** 48 GB. Both are **Ada Lovelace**,
both have RT cores.)

| | **B200** (p6-b200) | **L40S / L4** (g6e / g6) |
|---|---|---|
| Architecture | Blackwell (data-center) | Ada Lovelace |
| RT cores (ray tracing) | **No** | **Yes** |
| NVENC | No | Yes |
| VRAM | ~180 GB HBM3e | 48 GB (L40S) / 24 GB (L4) GDDR6 |
| Mem bandwidth | ~8 TB/s | ~0.86 TB/s (L40S) |
| Raw AI compute | top-tier | modest |
| ~On-demand cost | ~$110+/hr (8-GPU node) | ~$3/hr (g6.4xlarge) |
| Runs Isaac Sim? | **No** (no RT cores) | **Yes** |
| Runs MuJoCo MJX? | **Yes**, fast | Yes, slower |
| Runs a 30B LLM well? | **Yes** (180 GB, 8 TB/s) | Tight/slow (48 GB) |

## Purpose-fit: the two halves want opposite things

- **LLM half (vLLM, Qwen3-Coder-30B):** wants raw compute + huge VRAM + bandwidth → **B200
  wins decisively**. The L40S's 48 GB is tight for a 30B model + KV cache.
- **Simulation half:**
  - *Isaac Sim:* needs RT cores → **only the L40S/L4 can run it at all**; the B200 literally
    cannot launch it. Here the older, ~30× cheaper card is the *only* viable one.
  - *MuJoCo MJX:* needs only compute → **B200 is far faster** (more envs, more bandwidth); the
    L40S works but slower.

## The irony (the blog's money quote)

A generation-newer, ~30× more expensive B200 **cannot run the simulator that a humble L40S runs
fine** — because for Isaac Sim, *having RT cores beats being faster*. Conversely, the L40S is a
poor home for the 30B LLM. **Neither card is "better." They're fit for different jobs.**

The `nvidia-smi` process types make this visible:
- **C (compute):** CUDA kernels — vLLM, JAX/XLA training, PhysX. Every modern NVIDIA GPU has this.
- **G (graphics):** OpenGL/Vulkan/EGL contexts — rendering. Needs the graphics path; on
  data-center cards this is offscreen/EGL only.
- **C+G:** both. **Isaac Sim is inherently C+G** (rendering *is* its core loop, RT-core-bound),
  which is exactly why it can't init on a B200 (no G). **MuJoCo MJX is C during training** and
  only briefly **G** at demo-render time via EGL (no RT cores needed).

## How the two tracks proved the fit rule

- **Isaac track → forced onto L40S (g6):** the simulator's RT-core dependency dictated the
  hardware regardless of cost or compute. Pay ~$3/hr, accept slower physics, because it's the
  only thing that *runs*.
- **MuJoCo track → fits the B200:** MJX needs only compute, so it co-locates with vLLM on one
  B200 and exploits all the compute/bandwidth — one expensive box does both halves.

## The reusable decision rule

> **Pick your simulator first; it dictates your GPU class.**
> - Need photorealistic / ray-traced rendering (Isaac Sim, Omniverse)? → you need an **RT-core
>   GPU** (RTX workstation / L40S / L4). Raw compute is secondary; a data-center card without RT
>   cores is a non-starter no matter how fast.
> - Need only physics + light offscreen rendering (MuJoCo MJX)? → use the **compute-optimal
>   GPU** (B200/H100), and it can co-host your LLM too.
> Architecture generation (Blackwell vs Ada) matters for *speed*; RT-core presence matters for
> *feasibility*. Feasibility wins.

## Cross-references
- `../../docs/lesson-isaac-lab-bringup.md` (learning 3 & 4: the B200/Isaac RT-core wall).
- `docs/lesson-mjx-b200-bringup.md` (MJX physics + EGL render both pass on the B200).
- `docs/lesson-problems-and-resolutions.md` §2 (the two hardware gates).
- `docs/lesson-self-host-vllm-vs-bedrock.md` (the GPU-cost angle of the same B200).
