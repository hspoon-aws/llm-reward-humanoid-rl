# Project Plan — Humanoid goal-reaching LLM→Reward→RL loop on MuJoCo (MJX)

**Status:** PLAN / not yet scaffolded. This document is for review before any code is written.
**Origin:** sibling track to `deepracer-from-scratch` (Isaac Lab) after confirming Isaac Sim
cannot run on the `p6-b200.48xlarge` (data-center Blackwell has no RT cores — see
`../docs/lessons/lesson-isaac-lab-bringup.md`, confirmed learning 4).
**Goal:** keep the *same system behavior and the same 22 requirements*, swapping only the
simulator + RL stack from Isaac Lab / RSL-RL / PyTorch to **MuJoCo MJX / Brax PPO / JAX**.

## Dual-track decision (both projects run, in parallel, on different hardware)

We are running **both** simulators concurrently, each on the hardware that suits it:

| Track | Simulator | Instance | GPU class | Status |
|---|---|---|---|---|
| Isaac (existing) | Isaac Sim / Isaac Lab + RSL-RL | new **RTX** EC2 (e.g. `g6.4xlarge` / L40S, Ada — has RT cores) | RTX/workstation | moves off the B200 per the bring-up lesson |
| **MuJoCo (this)** | MJX + Brax PPO | the existing **`p6-b200.48xlarge`** | data-center (no RT cores) | this project |

Implications of running them side by side:

- **No contention.** The two tracks are on separate instances, so neither competes for GPU,
  VRAM, or the simulator's hardware requirements. The B200 is now *dedicated* to MuJoCo (vLLM on
  GPU 0, MJX/Brax on the rest); the RTX box owns Isaac.
- **LLM hosting is per-box.** Each instance hosts its own vLLM on its own GPU 0 (simplest, keeps
  the validated single-host loopback design on both). Alternative — one shared vLLM reached
  cross-host — adds a private network path + auth and breaks the loopback assumption, so default
  to per-box. The Bedrock backup (`../docs/lessons/lesson-self-host-vllm-vs-bedrock.md`) is attractive
  here precisely because it removes the "two vLLMs to babysit" problem: both tracks could call
  one managed endpoint instead.
- **The comparison becomes a deliverable.** Same task (Unitree H1 goal-reaching), same LLM-driven
  Eureka loop, two physics engines on two hardware classes. This is a genuine
  **Isaac-vs-MuJoCo / B200-vs-RTX** head-to-head — wall-clock to first walking policy, env
  throughput (steps/s), $/successful-policy, and reward-code portability (PyTorch vs JAX). Both
  projects' blogs (Req 21) should cross-reference and a joint comparison section should be
  planned. See "Comparison metrics to capture" below.
- **Shared assets.** Model weights in S3, prompt-engineering lessons, and the
  framework-agnostic core all transfer between tracks; only the simulator adapter + reward
  language differ.

> Note on location: this folder lives *inside* `deepracer-from-scratch/` only because file
> access is currently restricted to that workspace. It is intended as an independent sibling
> project (`projects/humanoid-mujoco-from-scratch/`) and should be moved out once a workspace
> that contains it is opened. Relative links to the Isaac project use `../`.

## Why MuJoCo on this exact instance

The B200 killed Isaac Sim at RTX/Vulkan init (RT cores required, B200 has none). MJX runs
physics as **JAX/XLA on CUDA compute** — the same path vLLM already uses successfully on this
host. No RT cores, no NVENC, no Vulkan rasterizer in the training loop. So:

- The LLM half (vLLM + Qwen, or the Bedrock backup) is **unchanged** and stays on GPU 0.
- The sim half now maps onto the hardware we already paid for. The B200 is **dedicated** to this
  MuJoCo track (Isaac moves to its own RTX box), so there is no cross-track contention.
- vLLM (CUDA compute) and MJX (CUDA compute) co-host cleanly on one B200 box; the validated
  `--network host` loopback topology carries over.

## Robot + environment selection

**Decision: Unitree H1 via `H1JoystickGaitTracking`, reframed to point-to-point goal-reaching.**

Confirmed against the live MuJoCo Playground locomotion registry
(`mujoco_playground/_src/locomotion/__init__.py`):

| Robot | Registered envs | Notes |
|---|---|---|
| **Unitree H1** | `H1InplaceGaitTracking`, `H1JoystickGaitTracking` | Same robot as the Isaac project. No goal-reaching task exists — same gap we reframed before. |
| Unitree G1 | `G1JoystickFlatTerrain`, `G1JoystickRoughTerrain` | Better maintained (has domain randomizer), flat-terrain joystick is the cleanest base. Different robot (~23 DOF). |
| Berkeley Humanoid, Booster T1, Apollo, OP3 | various Joystick | Other bipeds, further from the original. |

- **Primary: `H1JoystickGaitTracking`** keeps robot parity with the Isaac project (Unitree H1)
  and is the direct analog of the old `Isaac-Velocity-Flat-H1-v0` velocity-tracking task. We
  reframe it to goal-reaching exactly as the design already specifies: store a Goal on the env,
  append a robot-frame Goal_Observation, zero the stock gait/velocity-tracking reward, and let
  the generated reward own the objective.
- **Documented alternative: `G1JoystickFlatTerrain`** if H1's gait-tracking base proves awkward
  to reframe (it tracks a gait phase, not just a base velocity). G1 ships a flat-terrain
  joystick task + domain randomizer and is the better-supported humanoid in Playground. Keeping
  the env id a **config field** (as the Isaac project learned the hard way) makes H1↔G1 a
  one-line change, not a code change.

> Reframing mechanism is identical to the Isaac design's "primary mechanism": goal-conditioned
> reward on the stock locomotion env. The two reframing options (goal-conditioned reward vs.
> derive a velocity command from goal direction) both still apply; MJX changes the *how*, not
> the *what*.

## Stack mapping (what changes under the adapter)

| Concern | Isaac project | MuJoCo project |
|---|---|---|
| Simulator | Isaac Sim / Isaac Lab manager-based env | MuJoCo **MJX** (`mujoco_playground`) |
| Physics device | RTX/Vulkan + CUDA | **CUDA compute only** (JAX/XLA) |
| RL algorithm | RSL-RL `OnPolicyRunner` PPO | **Brax PPO** (`brax.training.agents.ppo`) |
| Tensor framework | PyTorch (`torch.Tensor`) | **JAX** (`jax.numpy`, functional, jit-traced) |
| Reward signature | `fn(env, ...) -> torch.Tensor (num_envs,)` | reward over `mjx.Data` state → `jax.Array (num_envs,)` |
| Parallelism | `num_envs` envs across GPUs 1–7 | **vmap'd** batch on GPU(s); MJX scales thousands of envs on one GPU |
| Env id source | `training.task` config field | `training.env_name` config field (default `H1JoystickGaitTracking`) |
| Demo video | Isaac external world cameras (RTX) | MuJoCo `Renderer` headless (EGL/OSMesa, **no RT cores**) — must verify on B200 |

## What carries over UNCHANGED (the payoff of the DI-clean design)

Per the Isaac design's component→requirement map, everything except the two simulator-facing
adapters is pure and host-testable. Target: reuse with near-zero change.

- **Qwen_Client** (`src/llm/qwen_client.py`) + prompt-template loading + retry/extraction/
  `ast` validation + `ServiceUnavailableError` classification — unchanged. (Optional Bedrock
  backend per `../docs/lessons/lesson-self-host-vllm-vs-bedrock.md` can come along.)
- **Orchestrator** loop + all recovery edges (skip, revert+lr-cut, OOM fallback, wait-and-
  resume), best-policy tracking, capture coordination, blog assembly — unchanged control flow.
- **Evaluator** metric math, staged goal gates, JSON serialization — unchanged (operates on
  rollout arrays, framework-agnostic once the rollout adapter feeds it).
- **Config loader**, **S3_Store**, **data models**, **exceptions**, **blog**, **capture**
  scaffolding — unchanged.
- The **lessons** from `../docs/lessons/lesson-prompting-qwen-for-reward-code.md` (specify the exact
  interface, grade against machine-checkable rules, guard the env surface) transfer fully.

## What actually changes (the real work)

1. **New trainer adapter `build_mjx_trainer`** replacing `build_isaaclab_trainer` /
   `_IsaacLabTrainer`. Wraps MJX env construction + Brax PPO. This is the bulk of the work, and
   it replaces (not ports) the highest-risk Isaac piece (the `OnPolicyRunner` cfg schema).
2. **Reward contract: PyTorch → JAX.** Biggest ripple. Generated `compute_reward` must be
   `jax.jit`-compatible: `jnp` ops, no Python branching on traced values, no in-place mutation,
   functional reads of `mjx.Data`. This rewrites:
   - `prompts/initial_reward.txt`, `prompts/refine_reward.txt` (the STATE API + signature +
     hard-rules blocks),
   - the Reward_Executor's allowlist (`jax`, `jnp` instead of `torch`) and the `_EnvAccessProxy`
     accessor surface (MJX `data.qpos`/`qvel`/`xpos`/contacts instead of
     `env.scene["robot"].data.root_pos_w`),
   - `scripts/probe_*_capability.py` grader rules (jit-compat checks instead of torch checks).
3. **Goal injection on an MJX env.** Functional/stateless: Goal lives in env state (or a config
   constant broadcast over the batch), Goal_Observation computed in the robot frame each step
   and concatenated to the proprio obs. Point A = reset pose; record A→Goal straight-line
   distance at reset for path-efficiency.
4. **Evaluation rollout adapter** producing the per-episode arrays the Evaluator already
   consumes (positions, upright time, torques) from a Brax/MJX rollout.
5. **Demo-video renderer** using MuJoCo's headless `Renderer` (EGL) instead of Isaac cameras —
   **must be smoke-verified on B200 before depending on it** (same cheap-gate discipline that
   caught the Isaac issue; rendering is the one MJX surface that touches GL).
6. **GPU allocation model.** JAX preallocates GPU memory aggressively; set
   `XLA_PYTHON_CLIENT_MEM_FRACTION` and pin JAX to the training GPU(s) while vLLM keeps GPU 0.
   Note: MJX often saturates a single GPU with thousands of envs, so we may not need 7 training
   GPUs — revisit whether the B200 is even the right size vs. cost.

## Requirements deltas (same behavior, updated mechanism wording)

The 22 requirements are preserved. Wording that names the old stack needs editing in a new spec:

- **Req 6, 8** ("Isaac Lab `RewTerm`", "RSL-RL") → MJX reward callable + Brax PPO.
- **Req 8.2 / 18.5** ("exclude GPU 0") → still valid: GPU 0 stays vLLM; JAX pinned off it.
  (If we drop to a single training GPU this becomes "JAX uses GPU N, vLLM uses GPU 0".)
- **Req 10** (external world-frame cameras) → MuJoCo headless renderer cameras; same intent
  (eval-only, small batch, never a policy input).
- **Req 16** (vLLM wait-and-resume) → unchanged for vLLM; behaves differently on Bedrock
  (throttling, not a dead socket) per the existing lesson.
- **Req 1.7 / obs space** → still "no image modality"; obs dims differ (H1 in Playground, not
  the ~69-dim Isaac layout) — re-derive the exact obs vector from the env's `default_config`.

Everything else (loop behavior, recovery, metrics, gates, best-policy, S3, blog, security) is
behavior-preserving and copies over.

## Proposed project structure (mirrors the Isaac project)

```
humanoid-mujoco-from-scratch/
├── PROJECT-PLAN.md                 # this file
├── README.md
├── pyproject.toml / requirements   # jax[cuda], mujoco, mujoco_mjx, mujoco_playground, brax, boto3
├── Dockerfile                      # JAX-CUDA base (no Isaac image); + project
├── config/run_config.yaml          # env_name, goal, training (Brax PPO knobs), s3, capture, blog
├── prompts/                        # JAX-rewritten initial/refine/analyze templates
├── src/
│   ├── llm/qwen_client.py          # UNCHANGED (carry over, optional Bedrock backend)
│   ├── rewards/reward_executor.py  # JAX allowlist + MJX _EnvAccessProxy
│   ├── envs/goal_env.py            # MJX goal wrapper (Goal + Goal_Observation, reward zeroing)
│   ├── train/ppo_runner.py         # build_mjx_trainer (Brax PPO) behind the same interface
│   ├── eval/evaluator.py           # UNCHANGED math + MJX rollout adapter
│   ├── sensors/camera_cfg.py       # MuJoCo headless renderer cameras
│   ├── storage/s3_store.py         # UNCHANGED
│   ├── config.py, data_models.py, exceptions.py, orchestrator.py, capture.py, blog.py, run_loop.py
├── scripts/
│   ├── probe_qwen_capability.py    # JAX-compat grader
│   ├── smoke_test_mjx.py           # headless MJX import + H1 env load + 1 step (cheap gate)
│   └── smoke_test_render.py        # headless EGL renderer on B200 (the render risk gate)
├── tests/                          # port the off-GPU suite; swap torch fakes for jax fakes
├── docs/                           # bring-up + decision lessons
└── .kiro/specs/humanoid-mujoco-llm-rl/   # requirements.md / design.md / tasks.md (edited copies)
```

## Phased execution plan

- **Phase 0 — Plan + spec (this).** Approve robot/env choice and structure. Copy the spec into
  `.kiro/specs/humanoid-mujoco-llm-rl/` and edit the stack-specific wording.
- **Phase 1 — Carry over the framework-agnostic core.** Port Qwen_Client, Orchestrator, Config,
  S3_Store, data models, exceptions, blog, capture + their off-GPU tests. Green test suite with
  fakes, zero MJX/JAX dependency at import (preserve the DI seam — the single best decision from
  the Isaac project).
- **Phase 2 — MJX env + Goal reframing.** `goal_env.py`: load `H1JoystickGaitTracking`, inject
  Goal + Goal_Observation, zero stock reward. Smoke: env loads headless, one vmap'd step.
- **Phase 3 — Reward contract (JAX).** Rewrite prompts + Reward_Executor allowlist + env proxy;
  rebuild the capability probe with jit-compat grading. This is where prompt-engineering lessons
  get re-validated against `jnp`.
- **Phase 4 — Brax PPO trainer adapter.** `build_mjx_trainer` behind the existing PPO_Runner
  interface; divergence/OOM/checkpoint behavior preserved.
- **Phase 5 — Evaluator rollout adapter + headless renderer.** Feed the unchanged metric math;
  verify EGL rendering on B200 (gate).
- **Phase 6 — End-to-end smoke on the box.** Tiny budget (few envs, few PPO steps, 1 iteration)
  → checkpoint + EvalMetrics + artifacts in S3, exactly like the Isaac Stage-C gate.
- **Phase 7 — Scale + unattended run + blog.**

## Open decisions (need your call before scaffolding)

1. **Robot/env:** confirm **H1 (`H1JoystickGaitTracking`)** for parity, or switch to
   **G1 (`G1JoystickFlatTerrain`)** which is better-supported in Playground. (Recommend: start
   H1 for parity — it keeps the two tracks comparing the *same robot* — and keep the env id a
   config field so G1 is a one-line fallback.)
2. **New spec vs. shared spec:** create a sibling spec
   `.kiro/specs/humanoid-mujoco-llm-rl/` (recommended — the stack wording differs), or reuse the
   Isaac spec and track deltas only here.
3. **Reward language:** generated reward in **JAX/`jnp`** (jit-compatible, fast, the MJX-native
   path — recommended) vs. a slower `jax.pure_callback` to numpy (simpler prompts, kills GPU
   throughput). Recommend JAX-native. (Note: this is the one place reward code is *not* portable
   between tracks — Isaac is torch, MuJoCo is jnp — which itself is a comparison finding.)
4. **B200 sizing — RESOLVED.** The B200 is dedicated to this track and Isaac moves to its own
   RTX box, so we keep the B200 here. Still worth measuring whether MJX saturates one GPU (it
   often does) so the *next* run can right-size; capture it as a comparison metric, not a
   blocker.
5. **vLLM hosting across the two tracks:** per-box vLLM (each instance runs its own on GPU 0 —
   recommended, keeps the loopback design on both) vs. one shared vLLM cross-host vs. **Bedrock
   for both** (removes two-vLLM babysitting; see the cost lesson). Recommend per-box for the
   sprint, Bedrock as the backup that also simplifies the dual-track case.
6. **How much to copy vs. import:** duplicate the carry-over modules into the new project
   (clean, independent) vs. share a common package across both projects (DRY, more coupling).
   Recommend duplicate — the two projects diverge at the adapter and independence is worth more
   than DRY here, *but* a shared core would make the head-to-head comparison strictly
   apples-to-apples. Flagging the tension explicitly given the dual-track goal.

## Comparison metrics to capture (Isaac/RTX vs MuJoCo/B200 head-to-head)

Since both tracks run the same task with the same Eureka loop, plan to record these for the
joint blog section:

- **Env throughput** — simulation steps/second at the training env count on each engine/GPU.
- **Wall-clock to first milestone** — iterations and hours to first makes-progress, first
  reaches-goal, first efficient-goal (the same staged gates, Req 17).
- **Cost per successful policy** — instance $/hr × hours to a reaches-goal policy, B200 vs RTX.
- **Reward-code portability** — how much the generated reward differs between torch (Isaac) and
  jnp (MuJoCo); how often each engine's generated reward fails to compile/run.
- **GPU utilization profile** — does MJX saturate one GPU vs. Isaac spreading across many.
- **Sim fidelity / sim-to-real notes** — qualitative gait differences between the two engines.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Headless render on B200 (EGL, no RT cores) for demo video (Req 10) | Medium | Phase-5 cheap `smoke_test_render.py` gate; fall back to CPU/OSMesa render for eval only |
| Generated reward not jit-compatible (Python control flow on traced vals) | High | Strong prompt rules + probe grader that compiles the fn under `jax.jit`; re-prompt on trace error |
| H1 gait-tracking base awkward to reframe to goal-reaching | Medium | G1 flat-terrain joystick fallback (config-swappable env id) |
| Brax PPO config/schema drift | Low-Med | Brax PPO is a stable, documented API; smaller surface than RSL-RL `OnPolicyRunner` |
| JAX GPU memory preallocation collides with vLLM on a shared box | Medium | `XLA_PYTHON_CLIENT_MEM_FRACTION`, pin JAX off GPU 0; verify with `nvidia-smi` |
| Reward math differences (torch vs jnp numerics) change behavior subtly | Low | Property tests on the executor; same gates/metrics catch regressions |

## Definition of done (parity with the Isaac project)

- Off-GPU test suite green with zero MJX/JAX import at module load (DI seam preserved).
- Hour-0 smoke on the box: one generated reward → validated/wrapped (JAX) → Brax PPO trains a
  few steps on the H1 MJX env → checkpoint → EvalMetrics → artifacts in S3.
- At least one full iteration's artifacts under `s3://.../runs/`.
- All 22 requirements satisfied with MuJoCo/Brax mechanisms; deltas documented in the new spec.
```
