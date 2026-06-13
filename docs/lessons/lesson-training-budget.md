# Lesson: training-budget semantics (Isaac/RSL-RL) — "epochs" is not a budget

**Date:** 2026-06-12
**Where it bites:** sizing per-iteration training so the Eureka loop actually *refines* rewards
on a single GPU, rather than over-training one reward or under-training all of them.
**Companion:** the MuJoCo track's `humanoid-mujoco-from-scratch/docs/lesson-problems-and-resolutions.md`
§7C (the "epochs was a 1500× landmine" finding) — this is the same audit applied to Isaac/RSL-RL.

## TL;DR

- **Always compute total env-steps per iteration, not raw epochs:**
  `env_steps_per_iter = num_envs × num_steps_per_env × epochs`.
  Sanity-check it against the task's published convergence budget BEFORE a long run.
- **Isaac differs from MuJoCo by the steps-per-epoch constant.** RSL-RL's `num_steps_per_env`
  is **24**; MJX/Brax's `episode_length` is **1000**. Same `epochs=1500` therefore means very
  different budgets:
  - Isaac 8-GPU (4096 envs): 4096 × 24 × 1500 ≈ **147M** — inside the ~50-200M H1 band. **Correct.**
  - MuJoCo (2048 envs): 2048 × 1000 × 1500 ≈ **3.07B** — 15-60× overkill (the landmine).
- So Isaac's `1500` was **not** the same bug — but two MuJoCo lessons transfer verbatim:
  1. the steps-per-epoch constant (`num_steps_per_env`) was **hardcoded in code, not in config** —
     a silent budget multiplier (exactly MuJoCo's `episode_length` mistake);
  2. **prefer more iterations × a sane per-iteration budget** over few iterations × a giant one —
     Eureka's value is reward *refinement*, not over-training a single reward.

## What was wrong / risky

1. **`num_steps_per_env` (=24) lived only in `_IsaacLabTrainer._build_train_cfg`**, invisible to
   anyone reading the config. You could not tell the real env-step budget from `run_config.yaml`.
2. **The single-GPU g6 profile copied `epochs: 1500` from the 8-GPU config.** On one L4 at
   1024 envs that is 1024 × 24 × 1500 ≈ **37M steps/iter** — *below* the convergence floor AND
   slow (one L4 is ~7× less throughput than the 7-GPU split). It would spend a long time
   per iteration yet still under-train, starving the Eureka refinement loop of iterations.
3. `checkpoint_interval: 500` on a sub-500-epoch search run would mean **mid-iteration
   checkpoints never fire** (same sub-bug MuJoCo hit).

## Fix (aligned with the MuJoCo conclusion)

- **Surfaced `num_steps_per_env` as a first-class config field** (`training.num_steps_per_env`,
  default 24) threaded through `Config` → `TrainConfig` → `_build_train_cfg`. No more silent
  multiplier; every budget knob is now in the config.
- **`config/run_config.yaml` (8-GPU sprint): kept `epochs: 1500`** — 4096 × 24 × 1500 ≈ 147M is
  the correct convergence budget; added an inline note showing the env-step math.
- **`config/run_config.g6.yaml` (single L4): `epochs: 1500 → 600`** (search-phase budget:
  1024 × 24 × 600 ≈ **15M steps/iter**, ~tens of minutes on an L4) so the loop can afford many
  reward-refinement iterations and each produces a *differentiated* learning signal for Qwen.
  Also `checkpoint_interval: 500 → 200` (≤ epochs, so mid-iteration checkpoints fire).

The split mirrors the Eureka method: a **search-phase** budget per candidate (enough to rank a
reward and feed metrics back), with the **winning reward** trained long separately for the final
policy. On 8 GPUs the per-iteration budget is already the convergence budget; on one L4 it must
be the smaller search budget.

## Why the smoke value (5 epochs) is fine for wiring but useless for learning

`--smoke-epochs 5` = 1024 × 24 × 5 ≈ 123K steps — a policy that just falls (the Isaac analog of
MuJoCo's "fall_rate 1.0 at 40 epochs"). The smoke gate only proves the pipeline wires end-to-end;
it produces **no learning signal**, so identical near-zero metrics across iterations would give
Qwen nothing to refine against. Never interpret a smoke run as a learning run.

## Lessons

- **"Epochs" is meaningless without the steps-per-epoch mapping.** Compute
  `num_envs × num_steps_per_env × epochs` and compare to the literature before committing GPU time.
- **Every budget multiplier belongs in the config**, never as a code default. A hidden constant
  (Isaac `num_steps_per_env`, MuJoCo `episode_length`) silently scales the run by 10-100×.
- **Single-GPU is a first-class profile, not the 8-GPU config with one GPU.** Re-size the budget
  for the throughput you actually have; copying the cluster budget under-trains AND runs slow.
- **For Eureka, more iterations beats a bigger single train.** The metric is moved by reward
  search, so spend the budget on refinement iterations at a sane per-iteration size.
