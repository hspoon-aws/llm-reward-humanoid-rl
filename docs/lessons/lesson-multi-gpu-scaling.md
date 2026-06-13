# Lesson: GPU scaling for the Eureka loop (Isaac/RSL-RL vs MJX/Brax)

**Date:** 2026-06-12
**Where it bites:** deciding how many GPUs to provision for Phase 2 training, and how the loop
should use them.
**Blog section:** "Running the loop on EC2" → GPU sizing / cost.

## TL;DR

- **One GPU = one training run** is the Isaac Lab model. Parallelism on a GPU comes from
  **thousands of vmap'd/parallel envs**, NOT from packing multiple training jobs onto it.
- **Scale UP before you scale OUT.** Push `num_envs` as high as one GPU's VRAM allows
  (4096/8192/…) first. A single A100/L40S at 4096 envs already saturates the GPU. Multi-GPU is
  for when one GPU is genuinely the bottleneck, not the default.
- The two projects differ entirely because of the **RL framework**:
  - **MJX + Brax (mujoco track): multi-GPU for free.** JAX `pmap`/sharding splits the env batch
    across every visible device in a **single process** — just set `CUDA_VISIBLE_DEVICES` + cap
    `XLA_PYTHON_CLIENT_MEM_FRACTION` and Brax does the rest (`num_envs % device_count == 0`).
  - **Isaac Lab + RSL-RL (isaac track): multi-GPU is explicit.** PyTorch **DDP under `torchrun`**,
    **one process per GPU**, gradients all-reduced. Not wired in our code today (the trainer pins
    `cuda:0`), so a multi-GPU box would train on only one GPU.

## "Only one training per GPU?" — yes, and here's the precise model

For Isaac Lab / RSL-RL the recommended pattern is:

> **One training process per GPU**, each running its **own large env batch**, synchronized via
> DDP all-reduce. You do NOT run multiple training jobs on one GPU, and you do NOT split a single
> env batch finer than per-GPU.

The mental model: **GPU = one training run.** Within that GPU, parallelism is the thousands of
envs, not multiple jobs.

## Isaac Lab has TWO multi-GPU axes (often conflated)

### Axis 1 — scale UP (single GPU, more envs) — the primary lever
Isaac Lab's whole point is massive on-GPU parallelism. First best practice is to maximize
`num_envs` on ONE GPU before adding GPUs. This is what we validated (smoke at 64 envs; real runs
go to thousands). "One GPU running thousands of envs, one training run" is the *normal* setup.

### Axis 2 — scale OUT (multi-GPU) — when one GPU isn't enough
PyTorch **DDP** via `torchrun`:
- one process per GPU (`--nproc_per_node=N`), each owns one GPU and its own full env set;
- **data-parallel**: every GPU runs the SAME policy on DIFFERENT env batches; gradients
  all-reduced each update, so effective batch = `num_envs × num_GPUs`;
- rank 0 owns logging/checkpointing;
- launch: `python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 train.py --distributed`.

## The distinction that matters for Eureka

There are **two different** multi-GPU strategies, mutually exclusive per GPU:

| Strategy | What it does | When |
|---|---|---|
| **DDP data-parallel** | Trains ONE policy faster / bigger batch across N GPUs | A single candidate's training is too slow even at max envs (rare for H1 flat goal-reaching) |
| **Embarrassingly parallel candidates** | Trains N reward candidates **simultaneously**, each pinned to its own GPU as an independent single-GPU job (no DDP) | The Eureka-faithful pattern — K× more reward search per wall-clock hour |

For a from-scratch Eureka loop, **parallel candidates is usually the better use of GPUs**: it
multiplies reward *search* throughput (the thing that drives improvement), rather than marginally
speeding one candidate. The original Eureka paper samples K rewards per iteration and trains them
concurrently across GPUs, then picks the best.

- **DDP** = make one candidate's training faster (1 policy across N GPUs).
- **Parallel candidates** = train N candidates at once (N independent 1-GPU jobs).

## Current state of our two codebases

- **mujoco track:** genuinely multi-GPU today via Brax/JAX sharding — single process, set the GPU
  list + mem fraction, Brax shards `num_envs` across visible devices. (`src/train/mjx_trainer.py`
  `_apply_device_pinning`.)
- **isaac track:** multi-GPU-*configurable* but single-GPU-*executing*. The config models a GPU
  list (`training_gpus`) and `select_cuda_visible_devices` masks correctly, but
  `_train_device_from_visible` returns `cuda:0` and one `OnPolicyRunner` is built — so only the
  first GPU trains. No `torchrun`/`LOCAL_RANK`/`init_process_group` wiring exists. The code's own
  docstring notes multi-GPU "uses torchrun distributed launch" — an intention, not an
  implementation.

## Recommendations (for when this is picked up)

1. **Single GPU, max envs is the correct default.** Don't add multi-GPU complexity until env
   count on one GPU is the binding constraint. This is what's validated.
2. If scaling out for Eureka, prefer **one candidate per GPU, run concurrently** (independent
   single-GPU jobs) over DDP-on-one-candidate. That's an **Orchestrator** change (dispatch K
   candidates to K GPUs + collect/select), not a trainer device change.
3. Reserve **DDP-on-one-candidate** for the case where a single policy's training is too slow at
   max envs — uncommon for H1 flat-ground goal-reaching.
4. Whichever path: keep GPU 0 reserved for the co-resident LLM only when vLLM is loopback; the
   cross-host topology frees all GPUs for training (already handled by the loopback check).

## Lessons

- **Framework choice decides the multi-GPU story.** JAX/Brax gives transparent single-process
  sharding; PyTorch/RSL-RL needs explicit multi-process DDP. Neither is "better" — but the cost of
  multi-GPU is very different, and it's worth knowing before provisioning.
- **For Eureka specifically, "more GPUs" should usually mean "more reward candidates in flight,"
  not "one candidate trained faster."** Match the parallelism to what actually moves the metric.
- **`num_envs` is the first knob, not GPU count.** Saturate one GPU before adding hardware.
